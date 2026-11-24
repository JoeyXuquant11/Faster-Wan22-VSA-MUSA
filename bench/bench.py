"""Frozen ResearchBoost-Agent judge for Wan2.2-TI2V-5B + VSA full finetuning on 8x MTT S5000.

Subcommands:
  check [--quick | --skip-timing | --n1]   Gates 0 -> 1 -> 2 -> 3
  time [--freeze]                          Steady-state timing vs frozen reference
  profile [--iters 1] [--bytes-model F]    Torch-MUSA profiler + kernel_util.py report
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
CFG = json.loads((BENCH / "bench.json").read_text())

sys.path.insert(0, str(ROOT))
from bench import recipe  # noqa: E402

TIER_ORDER = {"N0": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 4, "N5": 5}
GPU_SECONDS = 0.0


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[bench {ts}] {msg}", file=sys.stderr, flush=True)


def _expand(v: str) -> str:
    return os.path.expandvars(v)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def evaluator_sha256() -> str:
    h = hashlib.sha256()
    for base in (ROOT / "bench", ROOT / "numerics"):
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc" or p.name.startswith("._"):
                continue
            rel = str(p.relative_to(ROOT))
            h.update(rel.encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


class GpuLock:
    def __init__(self, path: str):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.path, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None


@contextlib.contextmanager
def scratch_dir(prefix: str):
    base = Path(_expand(CFG["scratch"]))
    base.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(prefix=prefix, dir=str(base))
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def candidate_environment(config_file: str | None) -> dict[str, str]:
    if not config_file:
        return {}
    raw = json.loads(Path(config_file).read_text())
    env = raw.get("environment", {})
    if not isinstance(env, dict):
        raise SystemExit(f"config {config_file}: environment must be a JSON object")
    reserved = set(CFG.get("reserved_env", []))
    bad = sorted(k for k in env if k in reserved)
    if bad:
        raise SystemExit(f"config {config_file}: reserved environment variables forbidden: {bad}")
    return {str(k): str(v) for k, v in env.items()}


def base_env(tmp: Path, cand_env: dict[str, str] | None = None, extra_pp: list[str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    tl_cache = tmp / "tilelang_cache"
    tmp_sub = tmp / "tmp"
    tl_cache.mkdir(parents=True, exist_ok=True)
    tmp_sub.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_sub)
    env["TMP"] = str(tmp_sub)
    env["TEMP"] = str(tmp_sub)
    env["TILELANG_CACHE_DIR"] = str(tl_cache)
    env["XDG_CACHE_HOME"] = str(tmp / "xdg_cache")
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["PYTORCH_MUSA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    pp = list(extra_pp or []) + [str(ROOT)]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pp)
    if cand_env:
        env.update(cand_env)
    return env


def run_gpu(cmd: list[str], cwd: Path, env: dict[str, str], log_file: Path, what: str, timeout: int):
    global GPU_SECONDS
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"start {what}: {cmd[0]} ...")
    t0 = time.perf_counter()
    with log_file.open("w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            rc = -9
    wall = time.perf_counter() - t0
    GPU_SECONDS += wall * int(CFG["world_size"])
    log(f"done {what}: rc={rc} wall={wall:.1f}s")
    return rc, wall, timed_out


def scan_signatures(log_file: Path) -> list[dict]:
    if not log_file.exists():
        return []
    txt = log_file.read_text(errors="replace")
    hits = []
    for sig in CFG.get("invalid_run_signatures", []):
        m = re.search(sig["regex"], txt)
        if m:
            hits.append({"regex": sig["regex"], "reason": sig["reason"], "match": m.group(0)})
    return hits


def tree_digest(tree: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(tree.rglob("*")):
        if not p.is_file():
            continue
        if any(x in p.parts for x in (".git", "__pycache__", ".pytest_cache")) or p.suffix == ".pyc":
            continue
        rel = str(p.relative_to(tree))
        h.update(rel.encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def materialize(src: Path, dest: Path) -> Path:
    def _ignore(d, names):
        return [n for n in names if n in (".git", "__pycache__", ".pytest_cache") or n.endswith(".pyc")]

    shutil.copytree(src, dest, ignore=_ignore)
    return dest


def check_reference() -> Path:
    ref = ROOT / CFG["reference"]
    if not ref.is_dir():
        raise SystemExit(f"missing reference tree: {ref}")
    head = subprocess.check_output(["git", "-C", str(ref), "rev-parse", "HEAD"], text=True).strip()
    expected_commit = CFG["reference_commit"]
    if head != expected_commit:
        raise SystemExit(f"reference HEAD {head} != expected {expected_commit}")
    dirty = subprocess.check_output(["git", "-C", str(ref), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise SystemExit(f"reference tree is dirty: {dirty[:200]}")
    return ref


def file_fingerprint(p: Path) -> dict:
    p = p.resolve()
    if p.is_file():
        return {"path": str(p), "size": p.stat().st_size, "sha256": sha256_file(p)}
    entries = []
    for f in sorted(p.rglob("*")):
        if not f.is_file() or f.name.startswith("._"):
            continue
        rel = str(f.relative_to(p))
        st = f.stat()
        item = {"rel": rel, "size": st.st_size}
        if st.st_size < (2 << 20):
            item["sha256"] = sha256_file(f)
        entries.append(item)
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"path": str(p), "files": len(entries), "digest": digest}


def verify_inputs(when: str) -> dict:
    fp_file = ROOT / "task" / "formulation.json"
    if not fp_file.exists():
        return {"when": when, "ok": True, "note": "formulation.json not frozen yet"}
    form = json.loads(fp_file.read_text())
    mismatches = []
    for k, expected in form.get("inputs", {}).items():
        cur = file_fingerprint(Path(expected["path"]))
        if cur != expected:
            mismatches.append({"name": k, "expected": expected, "actual": cur})
    if form.get("evaluator_sha256") and form["evaluator_sha256"] != evaluator_sha256():
        mismatches.append({
            "name": "evaluator_sha256",
            "expected": form["evaluator_sha256"],
            "actual": evaluator_sha256(),
        })
    return {"when": when, "ok": not mismatches, "mismatches": mismatches}


# ---------------- Gate 0: Structure (AST) & DataLoader ----------------


def _fn_ast_dump(src_path: Path, fn_name: str) -> str:
    tree = ast.parse(src_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.dump(node, include_attributes=False)
    raise RuntimeError(f"missing function {fn_name} in {src_path}")


def _trainer_train_header_dump(src_path: Path) -> str:
    tree = ast.parse(src_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "train":
                    for st in item.body:
                        if isinstance(st, ast.For):
                            return ast.dump(st.target) + "|" + ast.dump(st.iter)
    raise RuntimeError(f"missing Trainer.train loop in {src_path}")


def setup_guard(ref_tree: Path, cand_tree: Path) -> list[str]:
    errs = []
    ref_entry = ref_tree / "train.py"
    cand_entry = cand_tree / "train.py"
    if not cand_entry.exists():
        return ["missing train.py in candidate"]
    for fn in ("parse_args", "main"):
        try:
            if _fn_ast_dump(ref_entry, fn) != _fn_ast_dump(cand_entry, fn):
                errs.append(f"train.py:{fn}() AST differs from reference")
        except Exception as e:
            errs.append(f"train.py:{fn}() AST error: {e}")
    try:
        if _trainer_train_header_dump(ref_entry) != _trainer_train_header_dump(cand_entry):
            errs.append("train.py:Trainer.train() loop header differs from reference")
    except Exception as e:
        errs.append(f"train.py:Trainer.train() AST error: {e}")
    return errs


def compute_loader_digests(tree: Path, n_batches: int = 4) -> list[str]:
    bs = recipe.RECIPE["batch_size"]
    ws = CFG["world_size"]
    sd = recipe.RECIPE["seed"]
    nt = recipe.RECIPE["num_latent_t"]
    lh = recipe.RECIPE["latent_height"]
    lw = recipe.RECIPE["latent_width"]
    script = f"""
import hashlib, json, sys, torch
sys.path.insert(0, {str(tree)!r})
from wan22_vsa.data import build_dataloader
dl = build_dataloader(
    data_dir={str(recipe.DATA_DIR)!r},
    batch_size={bs},
    num_workers=0,
    rank=0,
    world_size={ws},
    seed={sd},
    num_latent_t={nt},
    latent_height={lh},
    latent_width={lw},
)
it = iter(dl)
out = []
for _ in range({n_batches}):
    b = next(it)
    h = hashlib.sha256()
    for k in sorted(b):
        v = b[k]
        h.update(k.encode() + str(v.dtype).encode() + str(tuple(v.shape)).encode())
        h.update(v.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    out.append(h.hexdigest()[:16])
print(json.dumps(out))
"""
    env = os.environ.copy()
    env["MKL_NUM_THREADS"] = "1"
    raw = subprocess.check_output([CFG["python"], "-c", script], env=env, text=True).strip()
    return json.loads(raw.splitlines()[-1])


def gate0_structure(tree: Path) -> tuple[str, dict] | None:
    ref = check_reference()
    errs = setup_guard(ref, tree)
    if errs:
        return "ERROR", {"passed": False, "stage": "structure", "errors": errs}
    loader_ref_file = ROOT / "task" / "loader_ref.json"
    if loader_ref_file.exists():
        expected = json.loads(loader_ref_file.read_text())
        try:
            actual = compute_loader_digests(tree, n_batches=len(expected))
        except Exception as e:
            return "ERROR", {"passed": False, "stage": "loader", "errors": [str(e)]}
        if actual != expected:
            return "INCORRECT", {
                "passed": False,
                "stage": "loader",
                "errors": [f"dataloader batches differ: expected={expected}, actual={actual}"],
            }
    return None


# ---------------- Gate 1: Timing & Profiling ----------------


def versions() -> dict:
    import torch
    import torch_musa

    return {
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "world_size": CFG["world_size"],
    }


def time_tree(
    tree: Path,
    out: Path,
    label: str,
    cand_env: dict[str, str] | None = None,
    profile: str = "none",
    profile_iters: int = 1,
    warmup: int | None = None,
    window: int | None = None,
    repeats: int | None = None,
) -> dict:
    tcfg = CFG["timing"]
    wu = tcfg["warmup"] if warmup is None else warmup
    win = tcfg["window"] if window is None else window
    rep = tcfg["repeats"] if repeats is None else repeats
    total_steps = wu + (profile_iters if profile == "torch" else win * rep) + 2
    ws = CFG["world_size"]

    prefix = out / f"timing_{label}"
    log_file = out / f"timing_{label}.log"

    with scratch_dir(f"wan22-time-{label}-") as tmp:
        train_argv = recipe.build_argv(max_train_steps=total_steps, output_dir=tmp / "train_out")
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={ws}",
            f"--master_port={free_port()}",
            str(BENCH / "timing.py"),
            "--out", str(prefix),
            "--warmup", str(wu),
            "--window", str(win),
            "--repeats", str(rep),
            "--profile", profile,
            "--profile-iters", str(profile_iters),
            "--",
            *train_argv,
        ]
        env = base_env(tmp, cand_env=cand_env, extra_pp=[str(tree)])
        timeout = CFG["timeouts_s"]["profile" if profile == "torch" else "timing"]
        rc, wall, timed_out = run_gpu(cmd, tree, env, log_file, f"timing({label})", timeout)

    sigs = scan_signatures(log_file)
    if timed_out:
        return {"status": "TIMEOUT", "label": label, "wall_s": round(wall, 3), "signatures": sigs}
    if rc != 0:
        return {"status": "ERROR", "label": label, "rc": rc, "wall_s": round(wall, 3), "signatures": sigs}
    if sigs:
        return {"status": "INVALID", "label": label, "wall_s": round(wall, 3), "signatures": sigs}

    ranks = []
    for r in range(int(ws)):
        rf = Path(f"{prefix}.r{r}.json")
        if not rf.exists():
            return {"status": "ERROR", "label": label, "reason": f"missing rank {r} timing file"}
        ranks.append(json.loads(rf.read_text()))

    n_win = len(ranks[0]["window_s"])
    if n_win == 0:
        return {"status": "ERROR", "label": label, "reason": "no timing windows recorded"}
    window_s = [round(max(r["window_s"][i] for r in ranks), 6) for i in range(n_win)]
    total_s = round(sum(window_s), 6)
    iters = int(ranks[0]["iterations_timed"])
    s_per_iter = round(total_s / iters, 6)
    win_spread = round((max(window_s) - min(window_s)) / statistics.median(window_s), 4) if n_win > 1 else 0.0
    peak_alloc = round(max(r["peak_memory_allocated_gib"] for r in ranks), 3)
    peak_resv = round(max(r["peak_memory_reserved_gib"] for r in ranks), 3)
    compiles_in_win = max(r["compiles_in_window"] for r in ranks)
    if compiles_in_win > 0:
        return {"status": "INVALID", "label": label, "reason": f"recompilation inside timing window ({compiles_in_win})"}

    res = {
        "status": "ok",
        "label": label,
        "wall_s": round(wall, 3),
        "startup_s": round(max(r["startup_s"] for r in ranks), 3),
        "warmup": wu,
        "window": win,
        "repeats": rep,
        "iterations_timed": iters,
        "window_s": window_s,
        "total_s": total_s,
        "s_per_iteration": s_per_iter,
        "window_spread": win_spread,
        "peak_memory_allocated_gib": peak_alloc,
        "peak_memory_reserved_gib": peak_resv,
        "rng_r0": ranks[0]["rng"],
        "fingerprint": {
            "versions": versions(),
            "world_size": ws,
            "device": ranks[0].get("device_name"),
            "warmup": wu,
            "window": win,
            "repeats": rep,
            "candidate_environment": cand_env or {},
        },
    }
    if profile == "torch":
        res["profile_json"] = str(Path(f"{prefix}.profile.r0.json"))
        res["trace_gz"] = str(Path(f"{prefix}.trace.r0.json.gz"))
    return res


def gate1_benefit(tree: Path, out: Path, cand_env: dict[str, str], final_mode: bool = False):
    tcfg = CFG["timing"]
    frozen_file = ROOT / "task" / "timing_ref.json"
    if not frozen_file.exists():
        return "ERROR", {"reason": "missing task/timing_ref.json (freeze baseline first)"}
    frozen = json.loads(frozen_file.read_text())

    if not final_mode:
        cand = time_tree(tree, out, "cand", cand_env=cand_env)
        if cand["status"] != "ok":
            return cand["status"], {"speedup": None, "runs": {"cand": cand}}
        ref_total = float(frozen["total_s"])
        speedup = round(ref_total / cand["total_s"], 4)
        ref_spread = float(frozen.get("reference_spread", 0.0))
        bar = round(max(float(CFG["benefit"]["min_speedup"]), 1.0 + 2.0 * ref_spread), 4)
        notes = []
        wbc = tcfg["window_band_candidate"]
        if cand["window_spread"] > wbc:
            notes.append(f"candidate window spread {cand['window_spread']:.1%} > {wbc:.0%} (note only)")
        perf = {
            "mode": "vs_frozen_reference",
            "frozen_reference_s_per_iteration": frozen["s_per_iteration"],
            "candidate_s_per_iteration": cand["s_per_iteration"],
            "speedup": speedup,
            "speedup_bar": bar,
            "reference_spread": ref_spread,
            "peak_memory_allocated_gib": cand["peak_memory_allocated_gib"],
            "reference_peak_memory_allocated_gib": frozen.get("peak_memory_allocated_gib"),
            "notes": notes,
            "runs": {"cand": cand},
            "passed": speedup > bar,
        }
        return ("ok" if perf["passed"] else "UNAVAILABLE"), perf

    ref_src = check_reference()
    with scratch_dir("wan22-ref-") as rtmp:
        ref_tree = materialize(ref_src, rtmp / "reference")
        ref_a = time_tree(ref_tree, out, "ref_a")
        cand = time_tree(tree, out, "cand", cand_env=cand_env) if ref_a["status"] == "ok" else None
        ref_b = time_tree(ref_tree, out, "ref_b") if cand and cand["status"] == "ok" else None
    runs = {"ref_a": ref_a, "cand": cand, "ref_b": ref_b}
    if any(r is None or r["status"] != "ok" for r in runs.values()):
        st = "INVALID" if any(r and r["status"] in ("INVALID", "TIMEOUT") for r in runs.values()) else "ERROR"
        return st, {"speedup": None, "runs": runs}

    fp_a = {k: v for k, v in ref_a["fingerprint"].items() if k != "candidate_environment"}
    for name in ("cand", "ref_b"):
        fp_x = {k: v for k, v in runs[name]["fingerprint"].items() if k != "candidate_environment"}
        if fp_x != fp_a:
            return "ERROR", {"speedup": None, "runs": runs, "reason": f"fingerprint mismatch in {name}"}

    ra, rb, c = ref_a["total_s"], ref_b["total_s"], cand["total_s"]
    ref_mean = (ra + rb) / 2.0
    ref_spread = abs(ra - rb) / ref_mean
    speedup = round(ref_mean / c, 4)
    problems = []
    fb, wb, wbc, pb = tcfg["frozen_band"], tcfg["window_band"], tcfg["window_band_candidate"], tcfg["pair_band"]
    for name, run in (("ref_a", ref_a), ("ref_b", ref_b)):
        off = run["total_s"] / frozen["total_s"] - 1.0
        if abs(off) > fb:
            problems.append(f"{name} {off:+.1%} off frozen reference (> {fb:.0%})")
        if run["window_spread"] > wb:
            problems.append(f"{name} window spread {run['window_spread']:.1%} > {wb:.0%}")
    if cand["window_spread"] > wbc:
        problems.append(f"cand window spread {cand['window_spread']:.1%} > {wbc:.0%}")
    if ref_spread > pb:
        problems.append(f"ref_a vs ref_b spread {ref_spread:.1%} > {pb:.0%}")

    bar = round(max(float(CFG["benefit"]["min_speedup"]), 1.0 + 2.0 * ref_spread), 4)
    perf = {
        "mode": "final_ref_cand_ref",
        "reference_s_per_iteration": round(ref_mean / ref_a["iterations_timed"], 6),
        "frozen_reference_s_per_iteration": frozen["s_per_iteration"],
        "candidate_s_per_iteration": cand["s_per_iteration"],
        "speedup": speedup,
        "speedup_bar": bar,
        "reference_spread": round(ref_spread, 4),
        "peak_memory_allocated_gib": cand["peak_memory_allocated_gib"],
        "reference_peak_memory_allocated_gib": round(
            (ref_a["peak_memory_allocated_gib"] + ref_b["peak_memory_allocated_gib"]) / 2.0, 3
        ),
        "problems": problems,
        "runs": runs,
        "passed": (not problems) and (speedup > bar),
    }
    if problems:
        return "INVALID", perf
    return ("ok" if perf["passed"] else "UNAVAILABLE"), perf


# ---------------- Gate 2: Bitwise Numerics ----------------


def record_cases(
    tree: Path,
    rec_dir: Path,
    cases_file: Path,
    log_file: Path,
    cand_env: dict[str, str] | None = None,
    losses_only: bool = False,
    lr_scale: float = 1.0,
) -> tuple[int, list[dict]]:
    rec_dir.mkdir(parents=True, exist_ok=True)
    ws = CFG["world_size"]
    with scratch_dir("wan22-rec-") as tmp:
        adapter_kwargs = json.dumps({"tree": str(tree), "scratch": str(tmp), "lr_scale": lr_scale})
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={ws}",
            f"--master_port={free_port()}",
            "-m", "numerics", "record",
            "--adapter", CFG["numerics"]["adapter"],
            "--adapter-kwargs", adapter_kwargs,
            "--cases", str(cases_file),
            "--out", str(rec_dir),
            "--deterministic",
            "--math-mode", CFG["math_mode"],
            "--sketch-dim", str(CFG["numerics"]["sketch_dim"]),
        ]
        if losses_only:
            cmd.append("--losses-only")
        env = base_env(tmp, cand_env=cand_env, extra_pp=[str(tree)])
        timeout = CFG["timeouts_s"]["trajectory" if losses_only else "numerics"]
        rc, _wall, timed_out = run_gpu(cmd, tree, env, log_file, f"record({rec_dir.name})", timeout)
        if timed_out:
            return -9, [{"reason": "timeout"}]
        return rc, scan_signatures(log_file)


def gate2_bitwise(tree: Path, out: Path, cand_env: dict[str, str], lr_scale: float = 1.0) -> dict:
    from numerics.judge import compare as num_compare

    rec_dir = out / "records_cand"
    log_file = out / "numerics_record.log"
    rc, sigs = record_cases(tree, rec_dir, ROOT / CFG["numerics"]["cases"], log_file, cand_env=cand_env, lr_scale=lr_scale)
    if rc != 0 or sigs:
        return {"status": "INVALID" if sigs else "ERROR", "rc": rc, "signatures": sigs}

    ref_dir = ROOT / CFG["numerics"]["ref"]
    th_file = ROOT / CFG["numerics"]["thresholds"]
    thresholds = json.loads(th_file.read_text())
    rep = num_compare(str(ref_dir), str(rec_dir), thresholds, exclude=[])
    (out / "numerics_compare.json").write_text(json.dumps(rep, indent=1) + "\n")
    if not CFG["numerics"].get("keep_records", False):
        shutil.rmtree(rec_dir, ignore_errors=True)
    return {
        "status": rep["status"],
        "measured_tier": rep["measured_tier"],
        "bitwise": rep["bitwise_identical"] == rep["checked"] and rep["checked"] > 0 and not rep["exact_failures"],
        "checked": rep["checked"],
        "bitwise_identical": rep["bitwise_identical"],
        "exact_failures": rep["exact_failures"][:20],
        "violations": len(rep["violations"]),
        "unresolved": len(rep["unresolved"]),
        "errors": rep.get("errors", [])[:20],
        "compare_json": str(out / "numerics_compare.json"),
    }


# ---------------- Gate 3: Statistical Trajectory ----------------


def gate3_statistical(tree: Path, out: Path, cand_env: dict[str, str], lr_scale: float = 1.0) -> dict:
    from numerics.trajectory import bound as traj_bound

    tcfg = CFG["trajectory"]
    frozen_file = ROOT / tcfg["frozen"]
    if not frozen_file.exists():
        return {"verdict": "error", "error": f"missing {frozen_file}"}
    frozen = json.loads(frozen_file.read_text())
    seeds = frozen["reference_seeds"][: int(tcfg.get("seeds", 1))]
    steps = int(tcfg["steps"])

    cand_traj_dir = out / "traj_cand"
    cand_traj_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        case_file = out / f"traj_case_s{seed}.json"
        case_file.write_text(json.dumps([{"name": f"traj_s{seed}", "seed": seed, "steps": steps, "phase": "train"}]))
        log_file = out / f"traj_s{seed}.log"
        rc, sigs = record_cases(
            tree, cand_traj_dir, case_file, log_file, cand_env=cand_env, losses_only=True, lr_scale=lr_scale
        )
        if rc != 0 or sigs:
            return {"verdict": "error", "status": "INVALID" if sigs else "ERROR", "rc": rc, "signatures": sigs}

    rep = traj_bound(frozen, str(cand_traj_dir), k=float(tcfg.get("sigma_k", 3.0)))
    (out / "traj_compare.json").write_text(json.dumps(rep, indent=1) + "\n")
    return rep


# ---------------- Unified Evaluation ----------------


def evaluate(
    tree: Path,
    out: Path,
    cand_env: dict[str, str],
    quick: bool = False,
    skip_timing: bool = False,
    n1_only: bool = False,
    final_mode: bool = False,
    lr_scale: float = 1.0,
):
    g0 = gate0_structure(tree)
    if g0 is not None:
        return g0[0], g0[1], None

    if n1_only:
        traj = gate3_statistical(tree, out, cand_env, lr_scale=lr_scale)
        st = "PASS" if traj.get("verdict") == "equivalent" else ("INCORRECT" if traj.get("verdict") == "different" else "ERROR")
        return st, {"passed": st == "PASS", "stage": "statistical (--n1)", "trajectory": traj}, None

    if quick:
        num = gate2_bitwise(tree, out, cand_env, lr_scale=lr_scale)
        if num.get("exact_failures"):
            return "INCORRECT", {"passed": False, "stage": "bitwise (--quick)", "numerics": num}, None
        if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
            return "ERROR", {"passed": False, "stage": "bitwise (--quick)", "numerics": num}, None
        return "PASS", {
            "passed": True,
            "stage": "quick (gates 1 and 3 skipped)",
            "bitwise": num["bitwise"],
            "measured_tier": num["measured_tier"],
            "numerics": num,
        }, None

    perf = None
    if not skip_timing:
        g1_st, perf = gate1_benefit(tree, out, cand_env, final_mode=final_mode)
        if g1_st != "ok":
            return g1_st, {"passed": False, "stage": "benefit", "note": "did not pass benefit gate"}, perf

    num = gate2_bitwise(tree, out, cand_env, lr_scale=lr_scale)
    if num.get("exact_failures"):
        return "INCORRECT", {"passed": False, "stage": "bitwise", "verdict": "exact_failure", "numerics": num}, perf
    if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
        return num.get("status", "ERROR"), {"passed": False, "stage": "bitwise", "numerics": num}, perf
    tier = num["measured_tier"]
    if TIER_ORDER.get(tier, 99) > TIER_ORDER[CFG["agent_max_tier"]]:
        return "INCORRECT", {
            "passed": False,
            "stage": "bitwise",
            "verdict": "tier_exceeded",
            "measured_tier": tier,
            "numerics": num,
        }, perf
    if num["bitwise"]:
        return "PASS", {
            "passed": True,
            "stage": "bitwise",
            "verdict": "bitwise",
            "measured_tier": tier,
            "numerics": num,
        }, perf

    traj = gate3_statistical(tree, out, cand_env, lr_scale=lr_scale)
    if traj.get("verdict") == "equivalent":
        return "PASS", {
            "passed": True,
            "stage": "statistical",
            "verdict": "equivalent",
            "measured_tier": tier,
            "numerics": num,
            "trajectory": traj,
        }, perf
    if traj.get("verdict") == "different":
        return "INCORRECT", {
            "passed": False,
            "stage": "statistical",
            "verdict": "different",
            "measured_tier": tier,
            "numerics": num,
            "trajectory": traj,
        }, perf
    return traj.get("status", "ERROR"), {
        "passed": False,
        "stage": "statistical",
        "verdict": "error",
        "numerics": num,
        "trajectory": traj,
    }, perf


def guarded(out: Path, fn, tree_src: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    with GpuLock(_expand(CFG["gpu_lock"])):
        start_inp = verify_inputs("start")
        if not start_inp["ok"]:
            return {"status": "ERROR", "stage": "inputs", "inputs": [start_inp], "note": "frozen inputs changed"}
        if tree_src is None:
            res = fn(None)
        else:
            with scratch_dir("wan22-src-") as tmp:
                tree = materialize(tree_src, tmp / "tree")
                dig = tree_digest(tree)
                res = fn(tree)
                res["source_digest"] = dig
                if tree_digest(tree) != dig:
                    res["status"] = "ERROR"
                    res["note"] = "candidate tree mutated itself during evaluation"
        end_inp = verify_inputs("end")
        res["inputs"] = [start_inp, end_inp]
        if not end_inp["ok"]:
            res["status"] = "ERROR"
            res["note"] = "frozen inputs changed during run"
        return res


def finish(out: Path, fname: str, res: dict) -> int:
    res["cost"] = {"gpu_hours": round(GPU_SECONDS / 3600.0, 4)}
    res["evaluator_sha256"] = evaluator_sha256()
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / fname).write_text(json.dumps(res, indent=1, default=str) + "\n")
    summary = {"status": res.get("status")}
    if isinstance(res.get("performance"), dict):
        summary["speedup"] = res["performance"].get("speedup")
        summary["s_per_iteration"] = res["performance"].get("candidate_s_per_iteration")
    for k in ("speedup_vs_frozen_reference", "s_per_iteration"):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get("status") in ("PASS", "ok") else 1


# ---------------- CLI Commands ----------------


def cmd_check(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    final_mode = bool(getattr(a, "final", False))
    if final_mode:
        marker = ROOT / "task" / "final-evaluation"
        if not marker.exists():
            raise SystemExit("check --final is reserved for the harness (missing task/final-evaluation)")

    def run(tree):
        st, corr, perf = evaluate(
            tree,
            out,
            cand_env=cand_env,
            quick=a.quick,
            skip_timing=a.skip_timing,
            n1_only=a.n1,
            final_mode=final_mode,
            lr_scale=a.lr_scale,
        )
        return {
            "status": st,
            "final": final_mode,
            "n1": a.n1,
            "tree": str(Path(a.tree).resolve()),
            "config": a.config,
            "environment": cand_env,
            "correctness": corr,
            "performance": perf,
        }

    return finish(out, "check.json", guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        if a.freeze:
            n_runs = int(CFG["timing"].get("reference_runs", 2))
            runs = []
            for i in range(n_runs):
                r = time_tree(tree, out, f"ref_freeze_{i}", cand_env=cand_env)
                if r["status"] != "ok":
                    return {"status": r["status"], "runs": runs + [r]}
                runs.append(r)
            totals = [r["total_s"] for r in runs]
            mean_total = statistics.fmean(totals)
            spread = (max(totals) - min(totals)) / mean_total if len(totals) > 1 else 0.0
            iters = runs[0]["iterations_timed"]
            frozen = {
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "reference_commit": CFG["reference_commit"],
                "runs": len(runs),
                "iterations_timed": iters,
                "total_s": round(mean_total, 6),
                "s_per_iteration": round(mean_total / iters, 6),
                "reference_spread": round(spread, 4),
                "peak_memory_allocated_gib": round(statistics.fmean(r["peak_memory_allocated_gib"] for r in runs), 3),
                "per_run_s_per_iteration": [r["s_per_iteration"] for r in runs],
                "per_run_window_s": [r["window_s"] for r in runs],
            }
            (ROOT / "task").mkdir(parents=True, exist_ok=True)
            (ROOT / "task" / "timing_ref.json").write_text(json.dumps(frozen, indent=1) + "\n")
            return {"status": "ok", "frozen": frozen}

        res = time_tree(tree, out, "tree", cand_env=cand_env)
        ref_file = ROOT / "task" / "timing_ref.json"
        if ref_file.exists() and res["status"] == "ok":
            frozen = json.loads(ref_file.read_text())
            res["speedup_vs_frozen_reference"] = round(frozen["total_s"] / res["total_s"], 4)
            res["frozen_reference_s_per_iteration"] = frozen["s_per_iteration"]
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)

    return finish(out, "time.json", guarded(out, run, Path(a.tree).resolve()))


CATEGORIES = [
    ("mccl", r"mccl|nccl|allgather|reducescatter|allreduce"),
    ("attention", r"flash|fmha|sdpa|attention|vsa|bsa|mate"),
    ("gemm", r"gemm|mublas|wmma|sgemm|hgemm|bf16.*mma|matmul|dot"),
    ("norm", r"layer_norm|rms_norm|norm"),
    ("optimizer", r"adam|foreach|multi_tensor"),
    ("memcpy/cast", r"copy|memcpy|memset|cast|convert|cat|permute|transpose|index|scatter|gather"),
    ("elementwise/reduce", r"elementwise|reduce|vectorized|gelu|silu|softmax|mul|add|topk"),
]


def categorize_kernels(kernels: list[dict]) -> dict[str, float]:
    buckets = {k: 0.0 for k, _ in CATEGORIES}
    buckets["other"] = 0.0
    for k in kernels:
        nm = k["name"].lower()
        ms = float(k["total_ms"])
        matched = False
        for cat, pat in CATEGORIES:
            if re.search(pat, nm):
                buckets[cat] += ms
                matched = True
                break
        if not matched:
            buckets["other"] += ms
    return {k: round(v, 3) for k, v in buckets.items() if v > 0}


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = time_tree(
            tree,
            out,
            "profile",
            cand_env=cand_env,
            profile="torch",
            profile_iters=a.iters,
        )
        if res["status"] != "ok":
            return res
        prof_file = Path(res["profile_json"])
        trace_gz = Path(res["trace_gz"])
        prof_data = json.loads(prof_file.read_text())
        res["category_ms"] = categorize_kernels(prof_data.get("kernels", []))
        res["annotations"] = prof_data.get("annotations", {})
        res["top_kernels"] = prof_data.get("kernels", [])[:25]

        if trace_gz.exists() and (BENCH / "kernel_util.py").exists():
            ku_cmd = [
                CFG["python"],
                str(BENCH / "kernel_util.py"),
                str(trace_gz),
                "--out", str(out / "kernels.md"),
                "--json", str(out / "kernels.json"),
            ]
            if CFG.get("peak_bw_gbs"):
                ku_cmd.extend(["--peak-bw-gbs", str(CFG["peak_bw_gbs"])])
            if a.bytes_model:
                ku_cmd.extend(["--bytes-model", str(Path(a.bytes_model).resolve())])
            p = subprocess.run(ku_cmd, capture_output=True, text=True)
            res["kernel_util_rc"] = p.returncode
            if p.returncode == 0:
                res["kernels_md"] = str(out / "kernels.md")
                res["kernels_json"] = str(out / "kernels.json")
            else:
                res["kernel_util_stderr"] = p.stderr[:1000]
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config)

    return finish(out, "profile.json", guarded(out, run, Path(a.tree).resolve()))


def cmd_freeze_inputs(a):
    task_dir = ROOT / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    ref = check_reference()
    loader_digs = compute_loader_digests(ref, n_batches=4)
    (task_dir / "loader_ref.json").write_text(json.dumps(loader_digs, indent=1) + "\n")

    form = {
        "task": CFG["task"],
        "reference_commit": CFG["reference_commit"],
        "evaluator_sha256": evaluator_sha256(),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "model_dir": file_fingerprint(recipe.MODEL_DIR),
            "data_dir": file_fingerprint(recipe.DATA_DIR),
            "cases": file_fingerprint(ROOT / CFG["numerics"]["cases"]),
            "loader_ref": file_fingerprint(task_dir / "loader_ref.json"),
        },
    }
    for key, rel in (
        ("timing_ref", "task/timing_ref.json"),
        ("thresholds", CFG["numerics"]["thresholds"]),
        ("traj_frozen", CFG["trajectory"]["frozen"]),
    ):
        p = ROOT / rel
        if p.exists():
            form["inputs"][key] = file_fingerprint(p)

    (task_dir / "formulation.json").write_text(json.dumps(form, indent=1) + "\n")
    print(json.dumps({"status": "ok", "evaluator_sha256": form["evaluator_sha256"], "loader": loader_digs}))
    return 0


def cmd_freeze_ref_bitwise(a):
    from numerics.judge import calibrate, compare

    out = Path(a.out).resolve()
    ref_src = check_reference()

    def run(tree):
        r1 = ROOT / CFG["numerics"]["ref"]
        r2 = out / "ref_run2"
        if r1.exists():
            shutil.rmtree(r1)
        rc1, sig1 = record_cases(tree, r1, ROOT / CFG["numerics"]["cases"], out / "ref1.log")
        if rc1 != 0 or sig1:
            return {"status": "ERROR", "step": "ref1", "rc": rc1, "signatures": sig1}
        rc2, sig2 = record_cases(tree, r2, ROOT / CFG["numerics"]["cases"], out / "ref2.log")
        if rc2 != 0 or sig2:
            return {"status": "ERROR", "step": "ref2", "rc": rc2, "signatures": sig2}

        th = calibrate(str(r1), [str(r2)])
        th_path = ROOT / CFG["numerics"]["thresholds"]
        th_path.parent.mkdir(parents=True, exist_ok=True)
        th_path.write_text(json.dumps(th, indent=1, sort_keys=True) + "\n")

        rep = compare(str(r1), str(r2), th)
        (out / "ref_self_compare.json").write_text(json.dumps(rep, indent=1) + "\n")
        bitwise_ok = (
            rep["status"] == "PASS"
            and rep["bitwise_identical"] == rep["checked"]
            and rep["checked"] > 0
            and not rep["exact_failures"]
        )
        return {
            "status": "PASS" if bitwise_ok else "ERROR",
            "checked": rep["checked"],
            "bitwise_identical": rep["bitwise_identical"],
            "exact_failures": rep["exact_failures"][:10],
            "thresholds": str(th_path),
            "ref_run2": str(r2),
        }

    return finish(out, "freeze_bitwise.json", guarded(out, run, ref_src))


def cmd_freeze_ref_traj(a):
    from numerics.trajectory import freeze as traj_freeze

    out = Path(a.out).resolve()
    ref_src = check_reference()
    tcfg = CFG["trajectory"]

    def run(tree):
        traj_ref_dir = ROOT / tcfg["ref"]
        traj_ref_dir.mkdir(parents=True, exist_ok=True)
        steps = int(tcfg["steps"])
        seeds = list(tcfg["reference_seeds"])
        spec = {
            "endpoints": [
                {"name": "loss_tail", "loss": "loss", "steps": [max(0, steps - 4), steps]},
                {"name": "gate_top10_step1", "loss": "gate_top10_abs_mean", "steps": [1, 2]},
            ]
        }
        spec_path = ROOT / tcfg["spec"]
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=1) + "\n")

        for seed in seeds:
            if (traj_ref_dir / f"traj_s{seed}.r0.json").exists() and not a.force:
                log(f"reuse existing ref trajectory seed {seed}")
                continue
            cf = out / f"ref_traj_s{seed}.json"
            cf.write_text(json.dumps([{"name": f"traj_s{seed}", "seed": seed, "steps": steps, "phase": "train"}]))
            rc, sigs = record_cases(tree, traj_ref_dir, cf, out / f"ref_traj_s{seed}.log", losses_only=True)
            if rc != 0 or sigs:
                return {"status": "ERROR", "seed": seed, "rc": rc, "signatures": sigs}

        frozen = traj_freeze(str(traj_ref_dir), spec, k=2.0)
        frozen_path = ROOT / tcfg["frozen"]
        frozen_path.write_text(json.dumps(frozen, indent=1) + "\n")
        return {"status": "ok", "frozen": frozen}

    return finish(out, "freeze_traj.json", guarded(out, run, ref_src))


def main():
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="Run gates 0 -> 1 -> 2 -> 3 on a candidate tree")
    c.add_argument("--tree", required=True)
    c.add_argument("--config", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--quick", action="store_true")
    c.add_argument("--skip-timing", action="store_true")
    c.add_argument("--n1", action="store_true")
    c.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    c.add_argument("--lr-scale", type=float, default=1.0, help=argparse.SUPPRESS)

    t = sub.add_parser("time", help="Time a tree against frozen reference")
    t.add_argument("--tree", required=True)
    t.add_argument("--config", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--freeze", action="store_true", help=argparse.SUPPRESS)

    pr = sub.add_parser("profile", help="Profile a tree with Torch-MUSA profiler + kernel_util")
    pr.add_argument("--tree", required=True)
    pr.add_argument("--config", default=None)
    pr.add_argument("--out", required=True)
    pr.add_argument("--iters", type=int, default=1)
    pr.add_argument("--bytes-model", default=None)

    sub.add_parser("freeze-inputs", help=argparse.SUPPRESS)
    fb = sub.add_parser("freeze-bitwise", help=argparse.SUPPRESS)
    fb.add_argument("--out", required=True)
    ft = sub.add_parser("freeze-traj", help=argparse.SUPPRESS)
    ft.add_argument("--out", required=True)
    ft.add_argument("--force", action="store_true")

    a = p.parse_args()
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "time":
        return cmd_time(a)
    if a.cmd == "profile":
        return cmd_profile(a)
    if a.cmd == "freeze-inputs":
        return cmd_freeze_inputs(a)
    if a.cmd == "freeze-bitwise":
        return cmd_freeze_ref_bitwise(a)
    if a.cmd == "freeze-traj":
        return cmd_freeze_ref_traj(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
