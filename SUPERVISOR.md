# Wan2.2-TI2V-5B + VSA (8× MTT S5000): supervision record

*Archived note: internal GPU microarchitecture identifiers, hardware peak/bandwidth parameters, and proprietary vendor kernel symbol names have been redacted; timing, speedup, memory, and numerics records are unchanged.*

One autonomous optimization session (`10-hour` wall-clock budget, `8 × Moore Threads MTT S5000`) optimized full-parameter finetuning of **Wan2.2-TI2V-5B + Video Sparse Attention (VSA)** (`5.28B` params, `704×1280×121f`, `S=27,280`, `vsa_sparsity=0.80`, `topk=96`, FSDP2 BF16 + FP32 master/AdamW); one supervisor formulated the frozen judge, enforced strict sandbox isolation, and oversaw infrastructure without supplying optimization hints. Target `4.0×` (`<= 6.32 s/iter`), minimum `2.5×` (`<= 10.12 s/iter`). Workspace layout: candidate `wan22_vsa_s5000/` (branch `opt`), reference `targets/reference` (commit `6e832aa`).

## Results

Final outer-harness certification (`bench/bench.py check --final`, candidate `3258fd7`, 12 accepted rounds within the 10-hour budget; full record in [`task/final_check.json`](task/final_check.json)):

| Item | Value |
|---|---|
| status | **PASS** (`final=true`, `accepted=true`, `met_target=true`) |
| performance.speedup | **`5.2816×`** (`25.2660 → 4.7837 s/iteration`; `ref_a 25.2639 s`, `cand 4.7837 s`, `ref_b 25.2682 s`; reference spread `0.02%`, candidate window spread `0.15%`) |
| Effective MFU (`@500T` BF16) | **`7.65% → 40.39%`** (`38.23 → 201.93 TFLOPS/GPU`, `~966 TFLOPs/GPU/step` effective FLOPs) |
| Peak HBM | **`68.60 GiB`** allocated / **`75.43 GiB`** reserved per GPU (`56.19 / 56.70 GiB` reference) |
| Gate 0 | AST match of `parse_args`/`main`/`Trainer.train` loop + first 4 batches bitwise identical |
| Gate 1 | `cand` `4.7837 s/iter`, RNG hashes (`rng_r0`) identical across `ref_a`, `cand`, `ref_b`; `problems = []` |
| Gate 2 | Tier **`N1`**; `15,992` of `71,008` tensors bitwise identical, **`0` exact-item failures**, `0` errors (not bitwise, so on to Gate 3) |
| Gate 3 | **`equivalent`**, paired deterministic 6-step trajectory on seed `10`: `loss_tail = -0.000227` (`-0.0342σ` vs `±3σ` bound `0.0199`), `gate_top10_step1 = -1.188e-8` (`-0.3499σ` vs `±3σ` bound `1.018e-7`) |
| Integrity | `evaluator_sha256` (`2c7ad67e...`) and `judge.sha256` (`96/96` files) verified intact; frozen inputs match at start and end; source digest `bb3314da...` |
| Cost | `2.7039 GPU-hours` for `check --final` |

**Target exceeded (`5.2816×` vs `4.0×` target) with every custom TileLang-MUSA and MUSA C++ kernel written from scratch.** Key progression on branch `opt` (detailed in [`TRACE.md`](TRACE.md) and [`WORK_LOG.md`](WORK_LOG.md)): TileLang-MUSA block-sparse VSA (`2.03×`), pure-scalar MUSA C++ fused norms/RoPE/VSA pre-post (`3.17×`), async pinned-CPU activation offload + GELU-only recompute + bucketed FSDP2 ZeRO-2 (`3.92×`), LPT load balancing + full-`K=128` VSA backward + parallel KV-CSR (`4.61×`), Cross-Attention offload + fused QKVG/AdamW (`4.75×`), L2-coherent 4-quantile scheduling (`4.96×`), warp-ballot Top-K + 3-block offload (`5.00×`), GELU+7-bias fusion (`5.03×`), 1-warp/row forward & shared-memory-accumulated backward norms (`5.10×`), static 3-slot GPU ring buffers (`5.11×`), VSA `BN=256` + register-hoisted `dQ` (`5.18×`), and branchless 4-tile VBS + clamp-safe `dQ` (`5.28×`).

## Formulation

- **Workload**: Official `Wan2.2-TI2V-5B` (30 DiT blocks, `dim=3072`, `ffn_dim=14336`, `24` heads × `128`, `5,282,995,392` trainable parameters including 30 zero-initialized VSA compression gates) finetuned on pre-encoded `704×1280×121f` video latents (`S=27,280`, 480 `(4,4,4)` tiles, `vsa_sparsity=0.80`, `topk=96`), global batch 8 (`batch_size=1` × 8 GPUs), Rectified Flow Matching velocity MSE loss, FSDP2 BF16 compute + FP32 master weights, FP32 gradient reduction, and FP32 AdamW states. Details in [`GOAL.md`](GOAL.md) and [`task/baseline.md`](task/baseline.md).
- **Hardware & Baseline**: `8 × MTT S5000` (`80 GiB` HBM per GPU). Frozen reference timing across two independent cold-cache runs: **`25.2887 s/iter`** (`reference_spread = 0.002%`), peak `56.19 GiB` alloc.
- **Deterministic mode & MUSA adaptation**: MUSA's stock `F.scaled_dot_product_attention` and gather-based VSA backward are nondeterministic under default flags. In deterministic judge mode, [`bench/wan22_vsa_adapter.py`](bench/wan22_vsa_adapter.py) routes those two stock PyTorch paths to same-precision deterministic implementations so the reference itself is `71,008 / 71,008` bitwise identical across two runs; custom candidate kernels are **not** bypassed and must provide their own deterministic path when `torch.are_deterministic_algorithms_enabled()` is True.
- **Judge validation**: `numerics selftest` passed `12/12` mutation checks; 6-seed trajectory reference (`seeds 10..15`, 6 steps each) established `sigma_seed = 0.006635` on `loss_tail` and `sigma_seed = 3.395e-8` (`0.172%`) on `gate_top10_step1`.

## Strict Isolation & Sandbox Setup

- The optimizing agent ran inside a dedicated container with `targets/reference/`, `bench/`, `numerics/`, `task/`, and `$ASSETS/` mounted read-only (`:ro`), and `task/final-evaluation` absent until the outer harness executed `check --final`.
- No pre-existing custom MUSA C++ or TileLang attention implementations were accessible from inside the container; all custom kernels in `wan22_vsa/` were authored from scratch during the 10-hour session using only the installed MUSA 5.2.0 / PyTorch 2.9.1 / TileLang-MUSA 0.1.8 / `mcc` stack and the installed skill cards.
- Every tool call (`760` MCP sandbox calls) was logged by an outer audit wrapper outside the container.

## Risks to Watch from Formulation Onwards

- **Precision floor**: BF16 compute + FP32 master weights, FP32 gradient reduction, FP32 AdamW, and FP32 norms/softmax/modulation; no FP8/INT8 and no `allow_tf32`.
- **Host/device memory ceiling**: `80 GiB` HBM per GPU leaves limited headroom once activation checkpointing is relaxed; OOM or allocator fragmentation can destabilize multi-step runs.
- **Async kernel faults & sync deadlocks**: Custom MUSA C++ / TileLang-MUSA kernels can suffer silent asynchronous OOB writes or async-pipeline barrier hangs; correctness tests must run under `MUSA_LAUNCH_BLOCKING=1` with short per-config timeouts.
- **Read-only judge integrity**: `bench/`, `numerics/`, `task/`, `targets/reference/`, and `$ASSETS/` are verified by SHA-256 before and after every check.

## Interventions (infrastructure only; zero optimization direction was ever given)

1. **Node contention migration (elapsed ~2.5 h, 13-min pause)**: Another cluster workload began sharing the initial 8-GPU node mid-session, risking noisy timing measurements. The harness paused the session cleanly, relocated the identical sandbox container onto an idle 8× MTT S5000 node with the exact same workspace state and remaining active budget, and resumed without providing any technical hint.
2. **Orphaned process cleanup after pipeline deadlock (elapsed ~7.8 h)**: During Round 11 kernel sweeps, testing a disabled-pipelining TileLang toggle deadlocked GPU 0 on async barriers and hit the `600 s` command timeout (`rc=124`), leaving a background `docker exec` process holding GPU 0. The supervisor terminated only that timed-out orphan process as infrastructure cleanup; the agent itself detected and killed a second hang on GPU 1 and wrapped subsequent sweeps in per-config `timeout 40`.
3. **Automatic harness resumes on turn/connection limits**: Whenever the CLI client ended a turn early before the 10-hour wall-clock budget expired, the outer harness issued a neutral continuation prompt pointing only to `GOAL.md`, `WORK_LOG.md`, and remaining time, and ran `bench/bench.py check --final` automatically at budget expiry.

## What the Judge Actually Caught

- **Gate 3 caught a subtle `NaN` that passed Gate 2 bitwise (`R12a`)**: Dropping `vbs` boundary masking in `vsa_bwd_dq` measured `4.7431 s/iter` (`5.3317×`) and matched the previous `R11` checkpoint across all `71,008` Gate 2 tensors, but Gate 3 (`traj_s10`, 6-step trajectory) failed with `grad_norm = NaN` on step 1 when pretrained QK-RMSNorm heads produced valid tokens with `lse < -127` (`exp2(0 - lse) = +Inf * 0.0 = NaN` on zero-padded keys). The agent reproduced the bug in isolation and replaced it with a bitwise-exact clamp (`T.min(S - lse, 1.0)` before `exp2`) that passed Gate 3 cleanly (`R12`, `3258fd7`).
- **Gate 1 rejected noisy / regressive micro-optimizations (`R13`, FSDP2 bucket/overlap sweeps)**: CTA-uniform `need_clamp` branching in `R13` passed Gate 3 (`N1`) at `4.8049 s/iter`, `+20.3 ms` slower than branchless `R12`, and was reverted; FSDP2 overlap and non-default `bpb=6/10` regressed under `bench.py time` and were discarded.
- **Runtime / compiler failures**: Disabling default TileLang pipeline scheduling deadlocked async barriers; `BN=192` failed TileLang tile-layout divisibility checks; `T.GemmWarpPolicy.Square`/`FullCol` broke `T.reduce_max(..., dim=1)` layout (`err = 1.00`).

## Known Limits of the Judge

- Gate 3 runs 1 paired deterministic seed (`seed 10`, 6 steps) against a `±3σ` bound calibrated from 6 reference seeds (`seeds 10..15`), following the current RBA judge rules. While it caught the `R12a` `NaN` immediately on step 1, very small multi-hundred-step convergence drift below `3σ` is not tested within 6 steps.
- `bench/recipe.py` carries a default fallback asset path that was sanitized to `./assets`, and `bench/` had hardware-specific bandwidth defaults redacted when archiving this task directory (runtime always expands `$ASSETS` from `bench.json` and `task/formulation.json`); see [`../README.md`](../README.md#archived-records) regarding `evaluator_sha256` after archival sanitization.

## Lessons

- **Memory headroom unlocks compute**: Writing a zero-materialization TileLang-MUSA VSA kernel in Round 1 freed `33.5 GiB` HBM immediately, which later enabled async pinned-CPU activation offload and full elimination of forward GEMM/VSA/Cross-Attention recomputation (`3.92×` by Round 3).
- **Measure library and communication behavior empirically**: Benchmarking native vendor Cross-Attention on short `S_k=512`, FSDP2 bucketing (`bpb=5` vs `6/10`), and overlap settings early avoided investing effort in unprofitable rewrites.
- **Bitwise-like 2-step checks are not enough without multi-step trajectory gates**: `R12a` proved that a real floating-point edge case (`lse < -127`) can stay invisible on a short 2-step fixture yet trigger `NaN` on the 6-step pretrained trajectory; keeping Gate 3 mandatory prevented shipping a broken `5.33×` kernel.

## Handoff

- **Certified results**: [`task/final_check.json`](task/final_check.json) (verbatim `check --final` record, paths sanitized to `$WORK`), [`TRACE.md`](TRACE.md), [`trace.json`](trace.json), and [`WORK_LOG.md`](WORK_LOG.md).
- **Re-running the judge**: Set `ASSETS` to a directory containing `models/Wan2.2-TI2V-5B` and `data/wan22_vsa_data_704x1280_121f`, re-freeze `task/records/` and `task/thresholds.json` on the target 8× MTT S5000 host, and invoke `python3 bench/bench.py check --tree <candidate_tree> --out runs/<run_name>`.
