"""Asynchronous pinned-CPU activation offloader with 1-block-ahead H2D prefetch for MTT S5000.

Uses dedicated DMA async copy engines (separate torch.musa.Stream) and static per-slot
GPU staging buffers (zero dynamic allocation / caching-allocator thrashing during backward)
to overlap D2H/H2D with transformer block compute while preserving exact bitwise activations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class AsyncOffloadPool:
    """Manages reusable pinned-CPU buffers, static GPU staging buffers, and overlapped DMA."""

    def __init__(self, device: torch.device):
        self.device = device
        self.stream = torch.musa.Stream(device=device)
        # Pre-allocated pinned CPU buffers keyed by (blk_idx, kind) -> Tuple[Tensor, ...]
        self._cpu_bufs: Dict[Tuple[int, str], Tuple[torch.Tensor, ...]] = {}
        # Static reusable GPU staging buffers keyed by kind -> Tuple[Tensor, ...]
        # Only one block consumes a given kind at a time during backward, so 1 staging slot per
        # kind suffices and eliminates all per-block aten::empty() allocator stalls in backward.
        self._staging_bufs: Dict[str, Tuple[torch.Tensor, ...]] = {}
        # Events tracking when compute_stream finished reading staging_bufs[kind]
        self._done_events: Dict[str, torch.musa.Event] = {}
        # Events tracking D2H completion per slot
        self._d2h_events: Dict[Tuple[int, str], torch.musa.Event] = {}
        # In-flight / prefetched GPU tensors + H2D completion events
        self._prefetched: Dict[Tuple[int, str], Tuple[Tuple[torch.Tensor, ...], torch.musa.Event]] = {}
        self._offloaded_slots: set[Tuple[int, str]] = set()
        self._scratch: Dict[str, torch.Tensor] = {}
        # Static multi-slot GPU ring buffers for giant forward activations (fc1 754MiB, vsa 720MiB)
        # Eliminates per-block aten::empty() allocator stalls and record_stream fragmentation.
        self._ring_bufs: Dict[Tuple[str, int], Tuple[torch.Tensor, ...]] = {}
        self._ring_d2h_events: Dict[Tuple[str, int], torch.musa.Event] = {}

    def get_scratch(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Return a static reusable GPU scratch buffer (avoids repeated 754 MiB alloc stalls)."""
        buf = self._scratch.get(key)
        if buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._scratch[key] = buf
        return buf

    def get_ring_slot(
        self,
        kind: str,
        blk_idx: int,
        specs: Tuple[Tuple[Tuple[int, ...], torch.dtype], ...],
        num_slots: int = 3,
    ) -> Tuple[torch.Tensor, ...]:
        """Return a static GPU ring buffer slot, waiting asynchronously on prior D2H if needed."""
        rkey = (kind, blk_idx % num_slots)
        evt = self._ring_d2h_events.pop(rkey, None)
        if evt is not None:
            torch.musa.current_stream(self.device).wait_event(evt)
        bufs = self._ring_bufs.get(rkey)
        if bufs is None or len(bufs) != len(specs) or any(
            tuple(b.shape) != tuple(sh) or b.dtype != dt for b, (sh, dt) in zip(bufs, specs)
        ):
            bufs = tuple(torch.empty(sh, dtype=dt, device=self.device) for sh, dt in specs)
            self._ring_bufs[rkey] = bufs
        return bufs

    def reset_step(self) -> None:
        self._prefetched.clear()
        self._d2h_events.clear()
        self._offloaded_slots.clear()
        self._ring_d2h_events.clear()

    def offload(
        self,
        slot_key: Tuple[int, str],
        tensors: Tuple[torch.Tensor, ...],
        is_ring: bool = False,
        num_slots: int = 3,
    ) -> None:
        """Asynchronously copy GPU tensors to reusable pinned CPU memory on the DMA stream."""
        # Keep at most 3 blocks of non-ring D2H in flight so deferred record_stream HBM stays bounded.
        # Ring-buffered slots use non-blocking GPU-side wait_event in get_ring_slot instead of CPU sync.
        blk_idx, kind = slot_key
        if not is_ring and blk_idx >= 3:
            old_evt = self._d2h_events.pop((blk_idx - 3, kind), None)
            if old_evt is not None:
                old_evt.synchronize()

        compute_stream = torch.musa.current_stream(self.device)
        ready_evt = torch.musa.Event()
        ready_evt.record(compute_stream)

        cpu_bufs = self._cpu_bufs.get(slot_key)
        if cpu_bufs is None or len(cpu_bufs) != len(tensors) or any(
            cb.shape != t.shape or cb.dtype != t.dtype for cb, t in zip(cpu_bufs, tensors)
        ):
            cpu_bufs = tuple(
                torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                for t in tensors
            )
            self._cpu_bufs[slot_key] = cpu_bufs

        d2h_evt = torch.musa.Event()
        with torch.musa.stream(self.stream):
            self.stream.wait_event(ready_evt)
            for cb, t in zip(cpu_bufs, tensors):
                cb.copy_(t, non_blocking=True)
            d2h_evt.record(self.stream)

        if is_ring:
            self._ring_d2h_events[(kind, blk_idx % num_slots)] = d2h_evt
        else:
            for t in tensors:
                t.record_stream(self.stream)

        self._d2h_events[slot_key] = d2h_evt
        self._offloaded_slots.add(slot_key)

    def _get_staging(self, kind: str, cpu_bufs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        bufs = self._staging_bufs.get(kind)
        if bufs is None or len(bufs) != len(cpu_bufs) or any(
            gb.shape != cb.shape or gb.dtype != cb.dtype for gb, cb in zip(bufs, cpu_bufs)
        ):
            bufs = tuple(
                torch.empty(cb.shape, dtype=cb.dtype, device=self.device)
                for cb in cpu_bufs
            )
            self._staging_bufs[kind] = bufs
        return bufs

    def prefetch(self, slot_key: Tuple[int, str]) -> None:
        """Asynchronously start H2D copy into the static staging buffer for `kind`."""
        if slot_key not in self._offloaded_slots or slot_key in self._prefetched:
            return
        _, kind = slot_key
        cpu_bufs = self._cpu_bufs[slot_key]
        d2h_evt = self._d2h_events.get(slot_key)
        done_evt = self._done_events.get(kind)
        h2d_evt = torch.musa.Event()
        with torch.musa.stream(self.stream):
            if d2h_evt is not None:
                self.stream.wait_event(d2h_evt)
            if done_evt is not None:
                self.stream.wait_event(done_evt)
            gpu_bufs = self._get_staging(kind, cpu_bufs)
            for gb, cb in zip(gpu_bufs, cpu_bufs):
                gb.copy_(cb, non_blocking=True)
            h2d_evt.record(self.stream)
        self._prefetched[slot_key] = (gpu_bufs, h2d_evt)

    def fetch(self, slot_key: Tuple[int, str]) -> Tuple[torch.Tensor, ...]:
        """Wait for H2D transfer into staging buffer and return GPU tensors."""
        if slot_key not in self._prefetched:
            self.prefetch(slot_key)
        gpu_bufs, h2d_evt = self._prefetched.pop(slot_key)
        self._offloaded_slots.discard(slot_key)
        self._d2h_events.pop(slot_key, None)
        compute_stream = torch.musa.current_stream(self.device)
        compute_stream.wait_event(h2d_evt)
        return gpu_bufs

    def release(self, kind: str) -> None:
        """Mark staging buffer for `kind` as finished on compute_stream so next prefetch can reuse it."""
        compute_stream = torch.musa.current_stream(self.device)
        evt = torch.musa.Event()
        evt.record(compute_stream)
        self._done_events[kind] = evt
