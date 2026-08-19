"""
CPU-only GRPO dataset builder: per-source targets, SFT exclusion, -Oz precompute.

Produces:
  ./compiler_dataset_train/   HF dataset (prompt, ir_relpath, initial_count,
                              oz_count, source, bench_key)
  ./ir_cache/*.ll             raw IR files
  ./oz_baselines.json         ir_relpath -> oz_count
  ./grand_spellbook.json      single source of truth for the reward validator

Ship: tar czf grpo_data_2.tgz ir_cache compiler_dataset_train \
        oz_baselines.json grand_spellbook.json

Requires: compiler_gym, LLVM 10 `opt`, datasets.
"""

import os
import json
import random
import itertools
import subprocess
import tempfile
import multiprocessing as mp
from datasets import Dataset

# =====================================================================
# CONFIG
# =====================================================================
LLVM_10_OPT_BIN = os.path.expanduser("~/rlc-project/llvm-10-toolchain/bin/opt")
IR_CACHE_DIR = "./ir_cache"
OUT_DATASET = "./compiler_dataset_train"
OUT_BASELINES = "./oz_baselines.json"

MIN_INSTS, MAX_INSTS = 200, 10_000
N_WORKERS = max(1, mp.cpu_count() - 2)
OPT_TIMEOUT = 60.0            # one-off offline job, be generous

# Per-source targets. `oversample` = how many candidates to scan per kept
# program; anghabench needs a huge factor because most of its functions
# are far below the 200-instruction floor.
# Per-source targets and exact scan numbers based on prior run telemetry.
# SOURCE_PLAN = {
#     "benchmark://github-v0":     {"target": 2000, "scan": 19866},
#     "benchmark://linux-v0":      {"target": 2000, "scan": 6552},
#     "benchmark://poj104-v1":     {"target": 2000, "scan": 4961},
#     "benchmark://anghabench-v1": {"target": 1000, "scan": 9006},
# }

SOURCE_PLAN = {
    "benchmark://github-v0":     {"target": 5429, "scan": 51207},
    "benchmark://linux-v0":      {"target": 5429, "scan": 16971},
    "benchmark://poj104-v1":     {"target": 5428, "scan": 12921},
    "benchmark://anghabench-v1": {"target": 2714, "scan": 23900},
}

# Smoke mode: SMOKE=1 python prepare_dataset_cpu_v2.py
if os.environ.get("SMOKE") == "1":
    for _cfg in SOURCE_PLAN.values():
        _cfg["target"] = 20
    print("*** SMOKE MODE: all per-source targets forced to 20 ***")

random.seed(3407)

# Held-out evaluation suites. Never sampled, no matter what.
HELD_OUT = {"cbench-v1", "chstone-v0", "mibench-v1", "npb-v0"}

# JSON files containing benchmark keys already consumed by SFT, in the
# format you already have: ["linux-v0_10440", "github-v0_37025", ...]
EXCLUDE_FILES = [
    "sft_trained_names.json",
]

# ==========================================
# 2. SPELLBOOK & COMPILER DICTIONARY
# ==========================================
PASS_DESCRIPTIONS = {
    "--add-discriminators": "Add discriminators for better debug info.",
    "--adce": "Aggressively eliminate dead code.",
    "--aggressive-instcombine": "Aggressive instruction combining.",
    "--alignment-from-assumptions": "Optimize memory alignment based on assumptions.",
    "--always-inline": "Inline all always_inline functions.",
    "--argpromotion": "Promote arguments from byref to byval.",
    "--attributor": "Propagate attributes across modules.",
    "--barrier": "Place barriers before code generation.",
    "--bdce": "Bit-level dead code elimination.",
    "--break-crit-edges": "Break critical edges to simplify CFG.",
    "--simplifycfg": "Simplify the control flow graph.",
    "--callsite-splitting": "Split indirect call sites based on constants.",
    "--called-value-propagation": "Propagate called values at indirect call sites.",
    "--canonicalize-aliases": "Canonicalize aliases for better analysis.",
    "--consthoist": "Hoist constants to higher scopes.",
    "--constmerge": "Merge duplicate constants.",
    "--constprop": "Simple constant propagation.",
    "--coro-cleanup": "Remove coroutine scheduling remnants.",
    "--coro-early": "Early coroutine transformation.",
    "--coro-elide": "Remove unnecessary coroutine constructs.",
    "--coro-split": "Split coroutines into multiple functions.",
    "--correlated-propagation": "Propagate correlated value info.",
    "--cross-dso-cfi": "Cross-DSO control flow integrity.",
    "--deadargelim": "Remove unused function arguments.",
    "--dce": "Dead code elimination.",
    "--die": "Dead instruction elimination.",
    "--dse": "Dead store elimination.",
    "--reg2mem": "Convert registers to stack memory references.",
    "--div-rem-pairs": "Optimize division and remainder pairs.",
    "--early-cse-memssa": "Early CSE based on memory SSA.",
    "--early-cse": "Early common subexpression elimination.",
    "--elim-avail-extern": "Convert available external globals to definitions.",
    "--ee-instrument": "Instrument exception handling for stack space.",
    "--flattencfg": "Flatten the control flow graph.",
    "--float2int": "Optimize floating-point to integer computations.",
    "--forceattrs": "Force setting function attributes.",
    "--inline": "Inline function code at call sites.",
    "--insert-gcov-profiling": "Insert GCOV-compatible instrumentation.",
    "--gvn-hoist": "Hoist redundant expressions.",
    "--gvn": "Global value numbering.",
    "--globaldce": "Global dead code elimination.",
    "--globalopt": "Global variable optimization.",
    "--globalsplit": "Split global variables into fragments.",
    "--guard-widening": "Widen guard conditions.",
    "--hotcoldsplit": "Split hot and cold paths.",
    "--ipconstprop": "Interprocedural constant propagation.",
    "--ipsccp": "Interprocedural sparse conditional constant propagation.",
    "--indvars": "Canonicalize loop induction variables.",
    "--irce": "Inductive range check elimination.",
    "--infer-address-spaces": "Infer address spaces.",
    "--inferattrs": "Infer attributes for unknown functions.",
    "--inject-tli-mappings": "Inject target library info mappings.",
    "--instsimplify": "Remove redundant instructions.",
    "--instcombine": "Combine instructions into simpler forms.",
    "--instnamer": "Assign names to unnamed instructions.",
    "--jump-threading": "Thread conditional jumps.",
    "--lcssa": "Convert loops to loop-closed SSA form.",
    "--licm": "Move loop-invariant code out of loops.",
    "--libcalls-shrinkwrap": "Optimize library call wrappers.",
    "--load-store-vectorizer": "Vectorize adjacent loads and stores.",
    "--loop-data-prefetch": "Prefetch data in loops.",
    "--loop-deletion": "Delete useless loops.",
    "--loop-distribute": "Distribute loops for parallelism.",
    "--loop-fusion": "Fuse loops to reduce overhead.",
    "--loop-guard-widening": "Widen loop guard conditions.",
    "--loop-idiom": "Recognize and replace common idioms in loops.",
    "--loop-instsimplify": "Simplify instructions in loops.",
    "--loop-interchange": "Interchange nested loops.",
    "--loop-load-elim": "Eliminate redundant loads in loops.",
    "--loop-predication": "Convert branches in loops to selects.",
    "--loop-reroll": "Reroll unrolled loops.",
    "--loop-rotate": "Rotate loops for better execution.",
    "--loop-simplifycfg": "Simplify loop control flow graph.",
    "--loop-simplify": "Canonicalize loop form.",
    "--loop-sink": "Sink instructions in loops.",
    "--loop-reduce": "Loop strength reduction.",
    "--loop-unroll-and-jam": "Unroll and jam nested loops.",
    "--loop-unroll": "Unroll loops.",
    "--loop-unswitch": "Extract conditions from loops.",
    "--loop-vectorize": "Vectorize loops.",
    "--loop-versioning-licm": "Create loop versions for LICM.",
    "--loop-versioning": "Create multiple loop versions.",
    "--loweratomic": "Lower atomic instructions.",
    "--lower-constant-intrinsics": "Lower constant intrinsics.",
    "--lower-expect": "Lower llvm.expect intrinsics.",
    "--lower-guard-intrinsic": "Lower guard intrinsics.",
    "--lowerinvoke": "Lower invoke and unwind instructions.",
    "--lower-matrix-intrinsics": "Lower matrix operation intrinsics.",
    "--lowerswitch": "Lower switch instructions.",
    "--lower-widenable-condition": "Lower widenable conditions.",
    "--memcpyopt": "Optimize memory copy operations.",
    "--mergefunc": "Merge duplicate functions.",
    "--mergeicmps": "Merge consecutive compare instructions.",
    "--mldst-motion": "Move memory load/store operations.",
    "--sancov": "Instrument sanitizer coverage.",
    "--name-anon-globals": "Name anonymous global variables.",
    "--nary-reassociate": "Reassociate n-ary expressions.",
    "--newgvn": "New global value numbering.",
    "--pgo-memop-opt": "Profile-guided memory operation optimization.",
    "--partial-inliner": "Partially inline hot paths.",
    "--partially-inline-libcalls": "Partially inline library calls.",
    "--post-inline-ee-instrument": "Instrument exception handling after inlining.",
    "--functionattrs": "Infer function attributes.",
    "--mem2reg": "Convert memory references to registers.",
    "--prune-eh": "Remove unreachable exception handling code.",
    "--reassociate": "Reassociate expressions.",
    "--redundant-dbg-inst-elim": "Remove redundant debug instructions.",
    "--rpo-functionattrs": "Infer function attributes in reverse postorder.",
    "--rewrite-statepoints-for-gc": "Rewrite statepoints for garbage collection.",
    "--sccp": "Sparse conditional constant propagation.",
    "--slp-vectorizer": "Superword-level parallelism vectorization.",
    "--sroa": "Scalar replacement of aggregates.",
    "--scalarizer": "Convert vector operations to scalar operations.",
    "--separate-const-offset-from-gep": "Separate constant offsets from GEP instructions.",
    "--simple-loop-unswitch": "Simplified loop unswitching.",
    "--sink": "Sink instructions to their use points.",
    "--speculative-execution": "Speculatively execute instructions.",
    "--slsr": "Straight-line strength reduction.",
    "--strip-dead-prototypes": "Remove unused function prototypes.",
    "--strip-debug-declare": "Remove debug declarations.",
    "--strip-nondebug": "Remove all non-debug information.",
    "--strip": "Remove all symbol information.",
    "--tailcallelim": "Eliminate tail calls.",
    "--mergereturn": "Merge multiple return points into one.",
    "-Oz": "Optimize aggressively for size."
}

def format_spellbook_by_category() -> str:
    buckets = {}
    for p, desc in PASS_DESCRIPTIONS.items():
        clean_name = p
        # Strip the trailing period from the description
        clean_desc = desc.rstrip('.')
        buckets.setdefault(clean_desc, []).append(clean_name)
        
    return "\n".join(
        f"  - {desc}: {', '.join(sorted(names))}"
        for desc, names in sorted(buckets.items())
    )

GRAND_SPELLBOOK_BLOCK = format_spellbook_by_category()

SYSTEM_PROMPT = (
    "You are an LLVM Compiler RL Agent. Your objective is to discover a custom sequence of optimization passes that will shrink the instruction count of the given IR even further than the standard '-Oz' optimization level.\n\n"
    "The only passes you can use along with their descriptions are:\n"
    f"{GRAND_SPELLBOOK_BLOCK}\n\n"
    "Rules:\n"
    "- Pick however many passes are actually useful — no bonus for more, no penalty "
    "for fewer.\n"
    "- End with your chosen passes as a JSON array of strings inside <answer>...</answer>, nothing else inside "
    "the tags.\n"
)

def make_prompt(autophase: dict) -> list:
    feat_str = ", ".join(f"'{k}': {v}" for k, v in autophase.items())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"\nAutoPhase: {{ {feat_str} }}\n\n"
                "Remember: conclude your response with exactly one <answer>...</answer> "
                "block containing a JSON array of pass names as strings."   # was "separated by commas"
            ),
        },
    ]

# =====================================================================
# BENCHMARK KEY  <->  URI
# =====================================================================
def bench_key(uri: str) -> str:
    """benchmark://linux-v0/10440    -> 'linux-v0_10440'
       benchmark://poj104-v1/60/674  -> 'poj104-v1_60_674'
    Matches the naming already in your SFT exclusion JSONs."""
    return (str(uri).replace("benchmark://", "")
                    .replace("generator://", "")
                    .replace("/", "_")
                    .replace(":", "_"))

def load_exclusions() -> set:
    excl = set()
    for path in EXCLUDE_FILES:
        if not os.path.exists(path):
            print(f"[exclude] {path} not found, skipping")
            continue
        with open(path) as f:
            names = json.load(f)
        cleaned = {str(n).replace(".ll", "").strip() for n in names}
        excl |= cleaned
        print(f"[exclude] loaded {len(cleaned)} keys from {path}")
    print(f"[exclude] {len(excl)} total programs will be skipped (SFT-used)")
    return excl

# =====================================================================
# PER-WORKER COMPILERGYM ENV
# =====================================================================
_ENV = None

def _get_env():
    global _ENV
    if _ENV is None:
        import compiler_gym
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

def _count_ll(path: str):
    for attempt in (0, 1):
        try:
            env = _get_env()
            env.reset(benchmark=env.make_benchmark(path))
            return int(env.observation["IrInstructionCount"])
        except Exception:
            if attempt == 0:
                _restart_env()
            else:
                return None

def _oz_count(ir_path: str):
    """opt -Oz with the SAME binary the reward system uses, counted with the
    SAME counter. Consistency here is what makes `gap` trustworthy."""
    fd, out_ll = tempfile.mkstemp(suffix=".ll")
    os.close(fd)
    try:
        try:
            r = subprocess.run([LLVM_10_OPT_BIN, "-Oz", ir_path, "-S", "-o", out_ll],
                               capture_output=True, timeout=OPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0:
            return None
        return _count_ll(out_ll)
    finally:
        if os.path.exists(out_ll):
            os.remove(out_ll)

# =====================================================================
# WORKER
# =====================================================================
def process_uri(args):
    uri, source = args
    try:
        env = _get_env()
        env.reset(benchmark=uri)
        init_count = int(env.observation["IrInstructionCount"])
        if not (MIN_INSTS <= init_count <= MAX_INSTS):
            return None

        feat = env.observation["AutophaseDict"]
        ir_text = env.observation["Ir"]

        key = bench_key(uri)
        rel = os.path.join("ir_cache", f"{key}.ll")
        abspath = os.path.abspath(rel)
        with open(abspath, "w") as f:
            f.write(ir_text)

        oz = _oz_count(abspath)
        if oz is None:
            os.remove(abspath)          # -Oz failed: unscoreable, drop it
            return None
        if init_count - oz < 1:
            os.remove(abspath)          # no headroom: nothing to learn
            return None

        return {
            "prompt": make_prompt(feat),
            "ir_relpath": rel,
            "initial_count": init_count,
            "oz_count": oz,
            "source": source,
            "bench_key": key,
        }
    except Exception:
        return None

# =====================================================================
# MAIN
# =====================================================================
def build_candidates(exclusions):
    """Per source: pull a pool, drop held-out + SFT-used, sample
    exact pre-calculated scan candidates."""
    import compiler_gym
    env = compiler_gym.make("llvm-v0")
    plan = {}
    for name, cfg in SOURCE_PLAN.items():
        short = name.replace("benchmark://", "")
        if short in HELD_OUT:
            print(f"REFUSING to sample held-out eval set {short}")
            continue

        want = cfg["scan"]
        it = env.datasets[name].benchmark_uris()
        # Pull 50k to ensure we have enough raw candidates to survive exclusions
        pool = list(itertools.islice(it, 50_000))

        kept = [u for u in pool if bench_key(u) not in exclusions]
        dropped = len(pool) - len(kept)

        take = min(want, len(kept))
        # Randomly sample the exact known 'scan' amount from the surviving pool
        plan[short] = random.sample(kept, take)
        print(f"{short}: pool={len(pool)} excluded={dropped} "
              f"candidates={take} (target {cfg['target']})")
    env.close()
    return plan
if __name__ == "__main__":
    # Spellbook: read-only, single source of truth for the reward validator.
    # Pass names in "--pass" / "-Oz" form — same shape the model is asked
    # to emit inside <answer>...</answer>.
    with open("grand_spellbook.json") as f:
        spellbook = json.load(f)
    print(f"Loaded grand_spellbook.json with {len(spellbook)} passes")

    import compiler_gym
    with compiler_gym.make("llvm-v0") as _e:
        real = {fl.lstrip("-") for fl in _e.action_space.flags}
    bogus = [p for p in spellbook if p.lstrip("-") not in real and p != "-Oz"]
    if bogus:
        print(f"\n*** WARNING: {len(bogus)} advertised passes are NOT valid "
              f"LLVM 10 actions. With strict validation the model eats -0.5 "
              f"for using them: {bogus}\n")

    os.makedirs(IR_CACHE_DIR, exist_ok=True)
    exclusions = load_exclusions()
    plan = build_candidates(exclusions)

    rows = []
    with mp.Pool(N_WORKERS) as pool:
        for short, uris in plan.items():
            target = SOURCE_PLAN[f"benchmark://{short}"]["target"]
            kept_here = 0
            work = [(u, short) for u in uris]
            print(f"\n--- {short}: scanning {len(work)} candidates "
                  f"for {target} keepers ---")
            for i, row in enumerate(pool.imap_unordered(process_uri, work, chunksize=8)):
                if row is not None:
                    rows.append(row)
                    kept_here += 1
                    if kept_here >= target:
                        print(f"  {short}: target {target} reached "
                              f"after {i+1} scanned")
                        break
                if (i + 1) % 1000 == 0:
                    print(f"  {short}: {i+1} scanned, {kept_here} kept "
                          f"({kept_here/(i+1):.1%} survival)")
            if kept_here < target:
                print(f"  {short}: WARNING only {kept_here}/{target} kept — "
                      f"raise 'oversample' for this source")

    by_src = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("\nFinal composition:", by_src, "TOTAL:", len(rows))

    # Sanity: no SFT program leaked in.
    leaked = [r["bench_key"] for r in rows if r["bench_key"] in exclusions]
    print(f"SFT leakage check: {len(leaked)} leaked (must be 0)")

    with open(OUT_BASELINES, "w") as f:
        json.dump({r["ir_relpath"]: r["oz_count"] for r in rows}, f, indent=2)

    random.shuffle(rows)
    Dataset.from_list(rows).save_to_disk(OUT_DATASET)
    print(f"\nSaved {len(rows)} examples to {OUT_DATASET}")
    print("Now: tar czf grpo_data_2.tgz ir_cache compiler_dataset_train "
          "oz_baselines.json grand_spellbook.json")
