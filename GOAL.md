# Goal

*Archived note: hardware microarchitecture identifiers and peak hardware specifications were redacted after the run; task rules and timing/numerics gates are unchanged.*

Speed up **Wan2.2-TI2V-5B + Video Sparse Attention (VSA) full-parameter finetuning** on **8x Moore Threads MTT S5000** to at least **4.0x speedup** over the frozen reference (`25.2887 s/iter` -> `<= 6.32 s/iter`), with a minimum acceptable speedup of **2.5x** (`<= 10.12 s/iter`) if profiling demonstrates that the remaining hotspots cannot be reduced further within the rules.

## Single Acceptance Criterion
- From `ws/`, run:
  ```bash
  python3 bench/bench.py check --tree wan22_vsa_s5000 --config <switch.json> --out runs/<run_name>
  ```
- Success requires **`status == "PASS"`** and **`performance.speedup >= 4.0`** (or `>= 2.5` for wrap-up, with a logged explanation covering every remaining large `short`, `under-filled`, `bw-poor` kernel class and repeated chain in the latest `kernels.md`).
- Keep `ws/FINAL.json` pointing at the best verified candidate at all times:
  ```json
  {"tree": "wan22_vsa_s5000", "config": "<path or null>", "check": "<path to latest PASS check.json>"}
  ```

## Workspace & Read-Only Rules
- **Candidate tree (writable)**: `wan22_vsa_s5000/` (branch `opt`, starting from commit `6e832aaf0dd2ef8edc5400d6448529f166575f98`). Commit each accepted change separately on branch `opt`.
- **Read-only paths (never modify)**: `targets/reference/`, `bench/`, `numerics/`, `task/`, `$ASSETS/`. The judge checks `evaluator_sha256` and input hashes before and after every run.
- **No new external packages**: use only what is installed in the container image (MUSA 5.2.0 runtime container: PyTorch/Torch-MUSA 2.9.1+musa5.2.0, TileLang-MUSA 0.1.8+musa.3, MATE 0.2.5, `mcc` compiler).

## Workload & Precision Floor (Must Not Change)
- Entry point `train.py`: `parse_args()`, `main()`, and `Trainer.train()` loop header must remain AST-identical to `targets/reference/train.py`.
- Data & batching: exact same pre-encoded 704x1280x121f dataset (`S=27,280`, 480 `(4,4,4)` tiles, `vsa_sparsity=0.80`, `topk=96`), same order, `batch_size=1` per GPU on 8 GPUs (global batch 8), `flow_shift=5.0`, `lr=1e-5`, `AdamW`, `max_grad_norm=1.0`, same Rectified Flow velocity MSE loss.
- **Precision floor**: keep BF16 forward/backward compute, FP32 master weights, FP32 gradient reduction, FP32 AdamW optimizer states, FP32 norms/softmax/modulation. Do **not** drop to FP8/INT8 training, do **not** remove FP32 master weights or optimizer states, and do **not** enable `allow_tf32`. Measured tier must be `<= N2`.
- **Same code path**: never inspect judge/bench environment variables or behave differently during timing vs judging.
- **Deterministic mode note**: in deterministic judge mode (`torch.are_deterministic_algorithms_enabled()`), `bench/wan22_vsa_adapter.py` routes MUSA's nondeterministic PyTorch `F.scaled_dot_product_attention` and `_block_sparse_attention_pytorch` to same-precision deterministic implementations for both reference and candidates using those paths. Any custom VSA or attention kernel added by a candidate is **not** bypassed and must compute at the same precision with a deterministic mode when `torch.are_deterministic_algorithms_enabled()` is True.

## How the Judge Decides & Command Costs
- **Gate 0 (Structure + Loader)**: AST match of `parse_args`/`main`/loop + first 4 batches bitwise identical (~5 s).
- **Gate 1 (Benefit)**: `bench.py check` runs one uninstrumented steady-state timing run (warmup=2, window=4, repeats=2, cold JIT cache) vs frozen reference (`25.2887 s/iter`, spread `0.002%`, bar `1.02x`). `bench.py check --final` (harness only, gated by `task/final-evaluation`) runs `ref_a -> cand -> ref_b`.
- **Gate 2 (Bitwise)**: 2-step deterministic recording across all 8 GPUs (`71,008` tensors). All exact items must pass; bitwise identical -> **PASS (N0)**.
- **Gate 3 (Statistical)**: when Gate 2 is not bitwise identical, runs 1 paired deterministic 6-step trajectory on seed 10; every endpoint must satisfy `|cand - ref| <= 3 * sigma_seed` (`loss_tail`: `3*sigma = 0.0199`; `gate_top10_step1`: `3*sigma = 1.018e-7 = 0.517%`) -> **PASS (N1/N2)**; otherwise **INCORRECT**.
- Commands:
  - `python3 bench/bench.py profile --tree wan22_vsa_s5000 --config <cfg> --out runs/<dir> --iters 1` (~1–3 min)
  - `python3 bench/bench.py time --tree wan22_vsa_s5000 --config <cfg> --out runs/<dir>` (~1–5 min)
  - `python3 bench/bench.py check --quick --tree wan22_vsa_s5000 --config <cfg> --out runs/<dir>` (~2–4 min)
  - `python3 bench/bench.py check --n1 --tree wan22_vsa_s5000 --config <cfg> --out runs/<dir>` (~1–5 min)
  - `python3 bench/bench.py check --tree wan22_vsa_s5000 --config <cfg> --out runs/<dir>` (~3–10 min)

## Workflow & Logging Rules
1. **Profile first every round** using `bench/bench.py profile` before editing code. Maintain a bytes model and log the top kernel classes and repeated chains from `kernels.md` in `WORK_LOG.md`.
2. **Time only through `bench/bench.py`** (never write ad-hoc timing scripts for reported speedups).
3. **Pace**: keep each optimization round focused (target <= 30 min per round: profile -> one hotspot class -> `time` -> `check` -> log).
4. **Update `WORK_LOG.md` every round** (including failed or no-gain rounds) and keep a running summary at the bottom with current best config, latest `check.json`, gain breakdown, and dead ends.
5. **Skills**: treat `train-opt-tiered-profiling`, `train-opt-action-ladder`, `train-opt-fusion-patterns`, `train-opt-measurement`, `train-opt-numerics-validation`, and `kernel-utilization-analysis` as priors to be verified experimentally on this task.

## Time Budget & Sandbox Rules
- **Time budget**: **10 hours (600 minutes)** wall clock. The harness stops the session automatically at 600 minutes and immediately runs `bench/bench.py check --final` on whatever candidate is recorded in `ws/FINAL.json`.
- **Keep `ws/FINAL.json` and `ws/WORK_LOG.md` updated after every accepted round** so a timeout never loses verified progress.
- **Strict sandbox**: you operate inside an isolated MTT S5000 container where `targets/reference/`, `bench/`, `numerics/`, `task/`, and `$ASSETS/` are mounted read-only. Use only the code inside this workspace, packages already installed in the container, and the installed skill cards. Write all custom kernels from scratch.
