"""
Reward system — Normalized variant, v2 (for the GPU training box).

Changes vs v1 (reward_system_normalized.py):

  1. OZ BASELINE FROM THE DATASET.
     `performance_reward_func` now takes `oz_count` as a kwarg. TRL's
     GRPOTrainer passes every non-prompt dataset column to reward
     functions as a list aligned with `completions`, so the `oz_count`
     column written by prepare_dataset_cpu_v2.py arrives here for free.
     _OZ_CACHE and _get_oz_count are DELETED: zero baseline compiles at
     train time, and the baseline is guaranteed to be the one computed
     with the same `opt` binary + same counter as the dataset build.

  2. OPTIONAL PARALLEL EVALUATION.
     Candidate sequences are deduplicated per batch, then evaluated on a
     persistent multiprocessing pool (spawn context, one CompilerGym env
     per worker, lazily created). Control with the REWARD_WORKERS env
     var:  REWARD_WORKERS=0 or 1  -> serial (best for debugging),
           REWARD_WORKERS=8       -> 8 parallel opt+count workers.

  3. TIMING TELEMETRY. Every tally print now includes wall time per
     reward call, so the step-5 smoke run tells you whether reward
     evaluation is a bottleneck worth more workers.

  4. GRANULAR HALLUCINATION PENALTY (this revision).
     validate_passes() no longer stops at the first unrecognized pass
     name. It walks the whole list and returns (valid_canonical_passes,
     num_hallucinated). A sequence with any hallucinated pass is
     rejected outright (never sent to opt) with:
         penalty = max(-1.5, -0.5 + num_hallucinated * -0.125)
     e.g. 3 hallucinated passes -> -0.5 + (3 * -0.125) = -0.875.
     Enough hallucinations floor out at -1.5. This replaces the old
     "reject whole sequence at -0.5, no matter how many bad names"
     policy described in earlier revisions of this docstring.

Reward for a valid, compiling, non-hallucinated sequence:
    gap      = Count(original) - Count(Oz)
    progress = (Count(original) - Count(optimized)) / gap
    < 100% of Oz   -> -0.5   (worse than the standard heuristic)
    == 100% of Oz  ->  0.0   (matched -Oz, no discovery)
    > 100% of Oz   ->  float(progress), capped at 1.5 for GRPO stability

NOTE ON WORKING DIRECTORY: this module reads grand_spellbook.json with a
relative path, and (under spawn) each pool worker re-imports this module
and reads it again. Launch training from the directory that contains
grand_spellbook.json and ir_cache/.
"""

import re
import os
import json
import time
import atexit
import subprocess
import tempfile
import multiprocessing as mp

import compiler_gym

# =====================================================================
# CONFIG
# =====================================================================
LLVM_10_OPT_BIN = os.path.expanduser("~/rlc-project/llvm-10-toolchain/bin/opt")
OPT_TIMEOUT_SECONDS = 30.0   # throughput guard only; timeouts are NOT penalized
MAX_PASSES = 55             # length-spam guard
TALLY_FILE = "tally_state.json" # <--- ADD THIS

# Float tolerance around progress == 1.0 (exact Oz match), to avoid
# false "beat/worse than Oz" classifications from float rounding noise
# in the instruction-count arithmetic.
_OZ_EPS = 1e-6

# 0 or 1 = serial evaluation in the trainer process (clean tracebacks).
# >1     = persistent worker pool, one CompilerGym env per worker.
N_REWARD_WORKERS = int(os.environ.get("REWARD_WORKERS", "0"))

with open("grand_spellbook.json", "r") as f:
    _SPELLBOOK = set(json.load(f))
# case-insensitive, dash-insensitive lookup -> canonical name
_SPELLBOOK_LOOKUP = {p.lower().lstrip("-"): p for p in _SPELLBOOK}

# =====================================================================
# PER-PROCESS COMPILERGYM ENV (counting only)
# In serial mode this lives in the trainer process; in parallel mode
# each pool worker gets its own copy (module globals are per-process).
# =====================================================================
_ENV = None

def _get_env():
    global _ENV
    if _ENV is None:
        _ENV = compiler_gym.make("llvm-v0")
    return _ENV

def _restart_env():
    global _ENV
    try:
        if _ENV is not None:
            _ENV.close()
    except Exception:
        pass
    _ENV = None
    return _get_env()

def _count_instructions(ll_path: str) -> int:
    """Ground-truth IrInstructionCount via CompilerGym, with one retry
    if the long-lived backend service has died."""
    try:
        env = _get_env()
        env.reset(benchmark=env.make_benchmark(ll_path))
        return int(env.observation["IrInstructionCount"])
    except Exception:
        env = _restart_env()
        env.reset(benchmark=env.make_benchmark(ll_path))
        return int(env.observation["IrInstructionCount"])

# =====================================================================
# PERSISTENT TALLY SYSTEM
# =====================================================================
_TALLY = {
    "no_answer": 0, "hallucinated": 0, "too_many_passes": 0,
    "opt_crash": 0, "opt_timeout": 0, "infra": 0,
    "no_headroom": 0, "no_baseline": 0, "scored": 0,
    "beat_oz": 0, "worse_than_oz": 0, "matched_oz": 0,
    "trivial_oz_only": 0, "unique_oz_beating": 0  # <--- Added unique_oz_beating
}
_BEATEN_IR_PATHS = set()

def load_tally():
    """Loads the global tally from disk to survive crashes."""
    global _TALLY, _BEATEN_IR_PATHS
    if os.path.exists(TALLY_FILE):
        try:
            with open(TALLY_FILE, "r") as f:
                data = json.load(f)
                _TALLY.update(data.get("tally", {}))
                _BEATEN_IR_PATHS = set(data.get("beaten_ir_paths", []))
                _TALLY["unique_oz_beating"] = len(_BEATEN_IR_PATHS)
        except Exception:
            pass

def save_tally():
    """Saves the global tally and unique beaten IRs to disk."""
    try:
        with open(TALLY_FILE, "w") as f:
            json.dump({"tally": _TALLY, "beaten_ir_paths": list(_BEATEN_IR_PATHS)}, f)
    except Exception:
        pass

# Load tally state immediately on startup
load_tally()
# =====================================================================
# EXTRACTION (unchanged)
# =====================================================================
THINK_CLOSE_TAG = "</think>"
ANSWER_PATTERN = re.compile(r"<answer\s*>\s*(.*?)\s*</answer\s*>",
                            re.DOTALL | re.IGNORECASE)
_STRAY_WRAPPER_CHARS = "[]'\"`() "

def extract_passes(text: str):
    """Look ONLY after the LAST </think>; take the LAST <answer> block
    there. Anything the model says mid-reasoning can never affect scoring.
    Returns a list of raw names, or None if no usable answer exists."""
    idx = text.rfind(THINK_CLOSE_TAG)
    if idx == -1:
        return None
    post_think = text[idx + len(THINK_CLOSE_TAG):]

    matches = ANSWER_PATTERN.findall(post_think)
    if not matches:
        return None
    raw = matches[-1]
    if "<" in raw or ">" in raw:      # stray tag chars inside the block
        return None

    tokens = re.split(r"[,\n]", raw)  # commas AND newlines
    names = [t.strip().strip(_STRAY_WRAPPER_CHARS).strip() for t in tokens]
    names = [n for n in names if n]
    return names

def validate_passes(names):
    """Resolve every name against the spellbook WITHOUT short-circuiting.

    Returns (canonical_valid_passes, hallucinated_passes):
      - canonical_valid_passes: the subset of `names` that resolved.
      - hallucinated_passes: list of strings that did NOT resolve.
    """
    canonical = []
    hallucinated = []
    for n in names:
        c = _SPELLBOOK_LOOKUP.get(n.lower().lstrip("-"))
        if c is None:
            hallucinated.append(n)
        else:
            canonical.append(c)
    return canonical, hallucinated

# =====================================================================
# APPLY PASSES (LLVM 10 opt) + COUNT (CompilerGym)  (unchanged)
# =====================================================================
def apply_and_count(ir_path: str, passes):
    """Returns (count, status): status in {ok, timeout, crash, infra}."""
    fd, out_ll = tempfile.mkstemp(suffix=".ll")
    os.close(fd)
    try:
        flags = [f"-{p.lstrip('-')}" for p in passes]
        cmd = [LLVM_10_OPT_BIN] + flags + [ir_path, "-S", "-o", out_ll]
        try:
            result = subprocess.run(cmd, capture_output=True,
                                    timeout=OPT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return None, "timeout"     # slow != wrong: neutral upstream
        if result.returncode != 0:
            return None, "crash"       # names were valid, so opt genuinely died
        try:
            return _count_instructions(out_ll), "ok"
        except Exception:
            return None, "infra"       # our counting service, not the model
    finally:
        if os.path.exists(out_ll):
            os.remove(out_ll)

# =====================================================================
# PARALLEL EVALUATION (new in v2)
# =====================================================================
def _eval_job(job):
    """Runs inside a pool worker. job = (ir_path, passes_tuple).
    Must be a top-level function so spawn can pickle it by reference.
    Never raises: any unexpected failure is reported as 'infra'."""
    ir, passes = job
    try:
        count, status = apply_and_count(ir, list(passes))
    except Exception:
        count, status = None, "infra"
    return job, count, status

_POOL = None

def _get_pool():
    global _POOL
    if _POOL is None:
        # spawn, not fork: the trainer process holds a CUDA context, and
        # forking a CUDA process is a well-known source of silent hangs.
        # Spawned workers re-import this module (so they read
        # grand_spellbook.json from the cwd) and lazily create their own
        # CompilerGym env on first job. First reward call therefore pays
        # a one-time startup cost of a few seconds per worker.
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(N_REWARD_WORKERS)
    return _POOL

def _shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL = None

atexit.register(_shutdown_pool)

# =====================================================================
# FORMAT REWARD (Granular Penalties)
# =====================================================================
def evaluate_format(text: str) -> float:
    """Returns 0.0 for correct format, or a specific penalty for errors."""
    idx = text.rfind(THINK_CLOSE_TAG)
    if idx == -1:
        return -2.0  # Heavy penalty: completely missed the CoT </think> structure

    post_think = text[idx + len(THINK_CLOSE_TAG):]
    matches = ANSWER_PATTERN.findall(post_think)
    if not matches:
        return -2.0  # Heavy penalty: forgot the <answer> tags

    raw = matches[-1].strip()
    try:
        names = json.loads(raw)
        if not isinstance(names, list):
            return -1.75  # Medium penalty: valid JSON, but not a list
        return 0.0        # Perfect format!
    except json.JSONDecodeError:
        return -1.75      # Medium penalty: tried to output JSON but had syntax errors

def format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards 0.0 for correct format, applies negative penalties for failures."""
    prompts = kwargs.get("prompts", kwargs.get("prompt", ["<PROMPT NOT FOUND>"] * len(completions)))

    rewards = []
    for i, (p, output) in enumerate(zip(prompts, completions)):
        text = output[0]["content"] if isinstance(output, list) else str(output)

        if _CALL_N.get("n", 0) == 0 and i == 0:
            print("\n" + "=" * 80)
            print("DEBUG: FULL PROMPT SEEN BY THE MODEL (INPUT):")
            print(p)
            print("-" * 80)
            print("DEBUG: EXACT TEXT GENERATED BY THE MODEL (OUTPUT):")
            print(text)
            print("=" * 80 + "\n")

        rewards.append(evaluate_format(text))
    return rewards

# =====================================================================
# PERFORMANCE REWARD (normalized progress toward -Oz)
# =====================================================================
_CALL_N = {"n": 0}
_TIME = {"total": 0.0}
_PRINT_EVERY = 1

# Telemetry accumulators used by _score(). These were referenced but
# never initialized in the previous revision -> NameError on first
# scored completion. Fixed here.
_PROGRESS_STATS = {"sum": 0.0, "n": 0, "min": None, "max": None}
_SOURCE_STATS = {}   # source label -> {"n": int, "sum_progress": float}

# def _score(count, status, orig, gap, source, is_oz_only, step_tally) -> float:
def _score(count, status, orig, gap, source, is_oz_only, ir_path, step_tally) -> float:
    """Turn one (count, status) result into a reward.
    Only clean (0 hallucination) sequences reach this function."""
    if status == "timeout":
        _TALLY["opt_timeout"] += 1
        step_tally["opt_timeout"] += 1
        return 0.0
    if status == "crash":
        _TALLY["opt_crash"] += 1
        step_tally["opt_crash"] += 1
        return -0.5
    if status == "infra":
        _TALLY["infra"] += 1
        step_tally["infra"] += 1
        return 0.0

    _TALLY["scored"] += 1
    step_tally["scored"] += 1
    progress = (orig - count) / gap

    if progress < 1.0 - _OZ_EPS:
        # Below Oz: reward scales with how close you got, floored at
        # -1.0 so a badly botched sequence can't fall off a cliff.
        final_reward = max(-1.0, float(progress) - 1.0)
    elif progress > 1.0 + _OZ_EPS:
        final_reward = min(float(progress), 1.5)
    else:
        final_reward = 0.0

    # ---- telemetry only, no effect on final_reward --------------------
    if progress > 1.0 + _OZ_EPS:
        _TALLY["beat_oz"] += 1
        step_tally["beat_oz"] += 1

        # <--- ADD THIS NEW UNIQUE TRACKING BLOCK --->
        if ir_path not in _BEATEN_IR_PATHS:
            _BEATEN_IR_PATHS.add(ir_path)
            _TALLY["unique_oz_beating"] = len(_BEATEN_IR_PATHS)
            step_tally["unique_oz_beating"] += 1

    elif progress < 1.0 - _OZ_EPS:
        _TALLY["worse_than_oz"] += 1
        step_tally["worse_than_oz"] += 1
    else:
        _TALLY["matched_oz"] += 1
        step_tally["matched_oz"] += 1
    if is_oz_only:
        _TALLY["trivial_oz_only"] += 1
        step_tally["trivial_oz_only"] += 1

    _PROGRESS_STATS["sum"] += progress
    _PROGRESS_STATS["n"] += 1
    _PROGRESS_STATS["min"] = progress if _PROGRESS_STATS["min"] is None else min(_PROGRESS_STATS["min"], progress)
    _PROGRESS_STATS["max"] = progress if _PROGRESS_STATS["max"] is None else max(_PROGRESS_STATS["max"], progress)

    src_stats = _SOURCE_STATS.setdefault(source, {"n": 0, "sum_progress": 0.0})
    src_stats["n"] += 1
    src_stats["sum_progress"] += progress

    return final_reward

def performance_reward_func(completions, ir_path, initial_count, oz_count,
                            **kwargs) -> list[float]:
    """`ir_path`, `initial_count`, `oz_count` are dataset columns that TRL
    forwards as lists aligned with `completions` (each prompt appears
    num_generations times, already expanded by the trainer).

    `source` is optional dataset-column telemetry (e.g. which generator
    produced the seed sequence). If your dataset doesn't have a `source`
    column, this falls back to "unknown" instead of crashing.
    """
    t0 = time.time()
    n = len(completions)
    rewards = [None] * n
    source = kwargs.get("source")  # optional; was referenced but never sourced before

    # Initialize the step tally to track ONLY this specific batch
    step_tally = {k: 0 for k in _TALLY.keys()}

    # ---- Phase 1: cheap, serial — extract, validate, dedupe ----------
    jobs = {}    # (ir_path, passes_tuple) -> [completion indices]
    meta = {}    # completion idx -> (orig, gap, source_label, is_oz_only)

    for idx, output in enumerate(completions):
        text = output[0]["content"] if isinstance(output, list) else str(output)

        names = extract_passes(text)
        if names is None:
            _TALLY["no_answer"] += 1
            step_tally["no_answer"] += 1
            rewards[idx] = 0.0           # format func already withheld 1.5
            continue

        if len(names) > MAX_PASSES:
            _TALLY["too_many_passes"] += 1
            step_tally["too_many_passes"] += 1
            rewards[idx] = -0.5
            continue

        # Walks the whole list; does NOT stop at the first bad name, so
        # num_hallucinated is an accurate count for the penalty formula.
        passes, hallucinated_passes = validate_passes(names)
        num_hallucinated = len(hallucinated_passes)

        # --- INSTANT REJECTION: any hallucination skips LLVM entirely ---
        if num_hallucinated > 0:
            _TALLY["hallucinated"] += 1
            step_tally["hallucinated"] += 1

            # -0.5 base penalty + -0.125 per hallucinated pass
            raw_penalty = -0.5 + (num_hallucinated * -0.125)

            # Floor the worst case at -1.5
            final_reward = max(-1.5, raw_penalty)

            # Print the exact bad passes that caused the rejection!
            print(f"[perf] {num_hallucinated} hallucinations {hallucinated_passes} -> {final_reward} (Sequence dropped)")

            rewards[idx] = final_reward
            continue  # never goes to LLVM

        # --- ONLY CLEAN SEQUENCES PROCEED PAST THIS POINT ---

        orig = initial_count[idx]
        oz = oz_count[idx]

        if oz is None or orig is None or orig <= 0:
            _TALLY["no_baseline"] += 1
            step_tally["no_baseline"] += 1
            rewards[idx] = 0.0
            continue

        gap = orig - oz
        if gap < 1:
            _TALLY["no_headroom"] += 1
            step_tally["no_headroom"] += 1
            rewards[idx] = 0.0
            continue

        is_oz_only = (len(passes) == 1 and passes[0].lower().lstrip("-") == "oz")
        src_label = source[idx] if source is not None else "unknown"

        key = (ir_path[idx], tuple(passes))
        jobs.setdefault(key, []).append(idx)

        # meta[idx] = (orig, gap, src_label, is_oz_only)
        meta[idx] = (orig, gap, src_label, is_oz_only, ir_path[idx])

    # ---- Phase 2: expensive — compile+count each UNIQUE sequence -----
    if jobs:
        job_list = list(jobs.keys())
        if N_REWARD_WORKERS > 1:
            results = _get_pool().map(_eval_job, job_list)
        else:
            results = [_eval_job(j) for j in job_list]

        for key, count, status in results:
            for idx in jobs[key]:
                rewards[idx] = _score(count, status, *meta[idx], step_tally)


    save_tally()
    # ---- Telemetry ---------------------------------------------------
    dt = time.time() - t0
    _CALL_N["n"] += 1
    _TIME["total"] += dt
    if _CALL_N["n"] % _PRINT_EVERY == 0:
        avg = _TIME["total"] / _CALL_N["n"]
        
        # Filter step_tally to only show > 0 metrics (keeps logs clean)
        active_step_tally = {k: v for k, v in step_tally.items() if v > 0}
        
        msg = (f"\n\n{'='*80}\n"
               f"===== STEP {_CALL_N['n']} RESULTS (Batch of {n}): {active_step_tally}\n"
               f"===== GLOBAL TALLY (All Time): {_TALLY}\n"
               f"===== last call {dt:.1f}s, avg {avg:.1f}s, "
               f"{len(jobs)} unique seqs this call =====\n"
               f"{'='*80}\n\n")
        
        print(msg, flush=True)

    return rewards