import os
import ast
import json
import random
import pandas as pd
from tqdm import tqdm
import boto3
import compiler_gym
from dotenv import load_dotenv
import re
from collections import Counter

# ==========================================
# 1. CORE SETUP & CONFIGURATION
# ==========================================

load_dotenv()

CSV_INPUT_FILE = "Experiment_1_2.csv"
OUTPUT_WITH_SPELLBOOK = "dataset_4000_balanced_with_spellbook_2.jsonl"
OUTPUT_WITHOUT_SPELLBOOK = "dataset_4000_balanced_without_spellbook_2.jsonl"

# Initialize AWS Bedrock Client
# Boto3 will automatically look for AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION in your environment
bedrock_client = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))

# Bedrock Model IDs are different from Anthropic's direct API.
# Note: Extended thinking is supported on Claude 3.7 Sonnet. Change this if needed.
MODEL_ID = "us.anthropic.claude-4-5-sonnet-20250219-v1:0" 

MIN_PASS_MENTION_RATIO = 1

VALID_DATASETS = [
    'poj104-v1', 'github-v0', 'linux-v0', 
    'tensorflow-v0', 'blas-v0', 'opencv-v0'
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

_CANONICAL_BY_STRIPPED = {p.lstrip('-').lower(): p for p in PASS_DESCRIPTIONS}

def canonicalize_pass(name: str) -> str:
    key = name.strip().lstrip('-').lower()
    return _CANONICAL_BY_STRIPPED.get(key, name) 

def format_spellbook_by_category() -> str:
    buckets = {}
    for p, desc in PASS_DESCRIPTIONS.items():
        clean_desc = desc.rstrip('.')
        buckets.setdefault(clean_desc, []).append(p)
        
    return "\n".join(
        f"  - {desc}: {', '.join(sorted(names))}"
        for desc, names in sorted(buckets.items())
    )

GRAND_SPELLBOOK_BLOCK = format_spellbook_by_category()

def passes_missing_from_trace(text: str, pass_list: list) -> list:
    text_lower = text.lower()
    missing = []
    for p in set(pass_list):
        bare = p.lstrip('-').lower()
        # Lookbehind ALLOWS hyphens (no \-) so "--prune-eh" matches "prune-eh"
        # Lookahead BLOCKS hyphens (has \-) so "loop-unroll" doesn't match "loop-unroll-and-jam"
        pattern = r'(?<![a-z0-9_])' + re.escape(bare) + r'(?![a-z0-9_\-])'
        if not re.search(pattern, text_lower):
            missing.append(p)
    return missing

def extra_passes_mentioned(text: str, pass_list: list) -> list:
    text_lower = text.lower()
    target_set = {p.lstrip('-').lower() for p in pass_list}
    found = []
    for bare, canonical in _CANONICAL_BY_STRIPPED.items():
        if bare in target_set or bare == "oz":
            continue
        # Same fix applied here
        pattern = r'(?<![a-z0-9_])' + re.escape(bare) + r'(?![a-z0-9_\-])'
        if re.search(pattern, text_lower):
            found.append(canonical)
    return found

# ==========================================
# 3. DATA LOADING, FILTERING & BALANCING
# ==========================================
print(f"Loading {CSV_INPUT_FILE}...")
df = pd.read_csv(CSV_INPUT_FILE)
df = df[df['OverOz'] > 0.0]

def is_valid_dataset_format(filename: str) -> bool:
    """Strictly ensures the filename matches train/dataset_v0/dataset_v0_id.ll format"""
    if pd.isna(filename):
        return False
    for ds in VALID_DATASETS:
        # Matches patterns exactly like: train/poj104-v1/poj104-v1_50_62.ll
        if re.search(rf"(?:train/)?{ds}/{ds}_.*\.ll$", filename):
            return True
    return False

# Apply strict filtering and safely reset the index
print("Filtering for valid datasets...")
df = df[df['Filename'].apply(is_valid_dataset_format)].reset_index(drop=True)

print("Parsing pass sequences for balancing...")
df['parsed_passes'] = df['PassSequence'].apply(
    lambda x: [canonicalize_pass(p) for p in ast.literal_eval(x) if p.lstrip('-').lower() != "oz"]
)

def print_pass_distribution(df, pass_column='parsed_passes'):
    print("\n" + "="*50)
    print(" PASS DISTRIBUTION TRACKER")
    print("="*50)
    
    all_passes = []
    for passes in df[pass_column]:
        if isinstance(passes, str):
            passes = ast.literal_eval(passes)
        all_passes.extend(passes)
        
    counts = Counter(all_passes)
    
    if not counts:
        print("No passes found in dataset.")
        return
        
    max_count = max(counts.values())
    min_count = min(counts.values())
    
    print(f"Total rows in dataset: {len(df)}")
    print(f"Total unique passes used: {len(counts)}")
    print(f"Min occurrences of a pass: {min_count}")
    print(f"Max occurrences of a pass: {max_count}")
    print(f"Spread (Max - Min): {max_count - min_count}\n")
    
    print("Detailed Counts (Least to Most Frequent):")
    for p, count in sorted(counts.items(), key=lambda x: (x[1], x[0])):
        print(f"  {count:4d} | {p}")
    print("="*50 + "\n")


# --- NEW DIAGNOSTIC CHECK ---
all_source_passes = set()
for seq in df['PassSequence']:
    try:
        parsed = [canonicalize_pass(p) for p in ast.literal_eval(seq) if p.lstrip('-').lower() != "oz"]
        all_source_passes.update(parsed)
    except:
        pass
print("\n" + "="*50)
print(f"[*] RAW DATA DIAGNOSTIC:")
print(f"[*] Total unique passes available in the filtered source CSV: {len(all_source_passes)} out of {len(PASS_DESCRIPTIONS)}")
print("="*50 + "\n")
# ----------------------------

print("Parsing pass sequences for balancing...")
df['parsed_passes'] = df['PassSequence'].apply(
    lambda x: [canonicalize_pass(p) for p in ast.literal_eval(x) if p.lstrip('-').lower() != "oz"]
)


# --- GREEDY BALANCING ALGORITHM (4000 ROWS) ---
BALANCED_CSV = "balanced_4000_checkpoints.csv"

if os.path.exists(BALANCED_CSV) and os.path.getsize(BALANCED_CSV) > 0:
    print(f"Loading existing balanced dataset from {BALANCED_CSV}...")
    balanced_df = pd.read_csv(BALANCED_CSV)
    balanced_df['parsed_passes'] = balanced_df['parsed_passes'].apply(ast.literal_eval)
else:
    print("Balancing dataset distribution from scratch...")
    pass_counts = {p: 0 for p in PASS_DESCRIPTIONS}
    pass_to_indices = {p: set() for p in PASS_DESCRIPTIONS}

    for idx, row in df.iterrows():
        for p in set(row['parsed_passes']):
            if p in pass_to_indices:
                pass_to_indices[p].add(idx)

    selected_indices = set()
    selected_rows = []
    TARGET_SIZE = min(4000, len(df))
    
    # Use a deterministic random seed so reruns produce the exact same distribution,
    # while ensuring we don't just pick rows from the very top of the CSV.
    rng = random.Random(42)
    df_indices_sorted = sorted(list(df.index)) 

    while len(selected_rows) < TARGET_SIZE:
        sorted_passes = sorted(pass_counts.keys(), key=lambda k: pass_counts[k])
        row_picked = None
        
        for p in sorted_passes:
            available = pass_to_indices[p] - selected_indices
            if available:
                # Random choice breaks the clustering at the start of the dataframe
                row_picked = rng.choice(sorted(list(available))) 
                break
                
        if row_picked is None:
            remaining = set(df_indices_sorted) - selected_indices
            if remaining:
                row_picked = rng.choice(sorted(list(remaining)))
            else:
                break

        selected_indices.add(row_picked)
        selected_rows.append(row_picked)
        
        for rp in df.loc[row_picked, 'parsed_passes']:
            if rp in pass_counts:
                pass_counts[rp] += 1

    balanced_df = df.loc[selected_rows].copy().reset_index(drop=True)
    balanced_df.to_csv(BALANCED_CSV, index=False)
    print(f"Dataset successfully balanced to {len(balanced_df)} rows and saved to {BALANCED_CSV}.")

print_pass_distribution(balanced_df)

try:
    processed_count = int(input(f"Enter the number of already processed rows to skip (enter 0 to start): "))
except ValueError:
    print("Invalid input. Defaulting to 0.")
    processed_count = 0

print(f"Skipping the first {processed_count} rows. Resuming from there...")

# ==========================================
# 4. INITIALIZE COMPILER GYM & PROCESS LOOP
# ==========================================

def get_gym_uris(filename: str) -> list:
    """Returns a list of possible CompilerGym benchmark URIs to try."""
    base_name = os.path.basename(filename).replace('.ll', '')
    for ds in VALID_DATASETS:
        if base_name.startswith(ds + "_"):
            benchmark_id = base_name[len(ds)+1:]
            
            # Primary format (e.g., benchmark://poj104-v1/53_501)
            uri_primary = f"benchmark://{ds}/{benchmark_id}"
            
            # Secondary fallback format (e.g., benchmark://poj104-v1/53/501)
            uri_secondary = f"benchmark://{ds}/{benchmark_id.replace('_', '/')}"
            
            # If there were no underscores, return just the primary
            if uri_primary == uri_secondary:
                return [uri_primary]
                
            return [uri_primary, uri_secondary]
    return []

print("Initializing CompilerGym LLVM Environment...")
env = compiler_gym.make('llvm-v0')

with open(OUTPUT_WITH_SPELLBOOK, "a", encoding="utf-8") as file_with, \
     open(OUTPUT_WITHOUT_SPELLBOOK, "a", encoding="utf-8") as file_without:
     
    for loop_index, row in tqdm(balanced_df.iterrows(), total=len(balanced_df), desc="Synthesizing SFT Traces"):
        if loop_index < processed_count:
            continue
            
        raw_filename = row['Filename']
        gym_uris = get_gym_uris(raw_filename)
        
        if not gym_uris:
            print(f"\n[Warning] Could not parse gym URI for {raw_filename}")
            continue
            
        autophase_dict = None
        last_error = None
        
        # Try both URI formats. If the first fails, try the fallback.
        for uri in gym_uris:
            try:
                env.reset(benchmark=uri)
                autophase_dict = env.observation['AutophaseDict']
                break  # Success! Break out of the retry loop.
            except Exception as e:
                last_error = e
                continue
                
        # If both formats failed, we skip and move on
        if autophase_dict is None:
            print(f"\n[Warning] Skipping benchmark {gym_uris[0]}. Both formats failed: {last_error}")
            continue
            
        # Optional: Print a small success message if the fallback worked
        if uri != gym_uris[0]:
            pass # The fallback format (slash) successfully loaded
            
        clean_passes = row['parsed_passes']
        optimal_passes_json_str = json.dumps(clean_passes)
        feat_str = ", ".join(f"'{k}': {v}" for k, v in autophase_dict.items())
        
        oracle_prompt = (
            f"You are generating synthetic Chain-of-Thought (CoT) training data for an RL LLVM Agent.\n\n"
            f"Benchmark state:\n"
            f"AutoPhase metrics: {feat_str}\n\n"
            f"List of passess given to the RL LLVM Agent:\n {GRAND_SPELLBOOK_BLOCK}"
            f"The ultimate sequence of passes the agent must organically deduce is: {optimal_passes_json_str}\n\n"
            f"Write the internal thinking process of the agent as it works out this exact sequence reason to reduce code better than -Oz "
            f"the way a frontier reasoning model actually works through a hard problem, not the way a textbook explains one it already solved."
            f"Concretely:\n"
            f"- Start from the raw AutoPhase numbers and form a hypothesis about what kind of code this is "
            f"(loop-heavy, branch-heavy, lots of small functions, etc.) before naming any pass.\n"
            f"- Weigh candidate passes against each other using the specific numbers, not vague heuristics — "
            f"explain why a metric argues for one pass and against a plausible alternative.\n"
            f"- Justify ordering and any repetition (why does a pass need to run again once an earlier pass "
            f"changes the numbers it depends on?).\n"
            f"- Build the sequence forward, in the same order it appears in the final answer — don't state the "
            f"answer first and justify it backwards.\n"
            f"Depth matters here: a two-sentence justification per pass is a failure. This should read like "
            f"extended, uncertain, self-correcting reasoning, not a tidy summary.\n\n"
            f"Keep your reasoning deeply grounded."
            f"CRITICAL RULES:\n"
            f"1. NO META-AWARENESS: Never mention that you were given the optimal sequence. Roleplay perfectly as if you are deducing this from scratch.\n"
            f"2. GROUNDED REASONING: Connect the target passes directly to the numbers in the AutoPhase dict.\n"
            f"3. PHASED EXECUTION: Logically justify why passes are repeated.\n"
            f"4. STRICT OUTPUT STRUCTURE: You MUST output exactly two XML tags. DO NOT use markdown code blocks (```) or asterisks anywhere. The answer block MUST contain a valid JSON array of strings.\n\n"
            f"5. PLAIN TEXT ONLY: Use plain ASCII punctuation throughout — straight quotes (' and \"), "
            f"regular hyphens (-), no em/en dashes, no smart quotes, no bullet characters or other "
            f"typographic symbols.\n"
            f"Format your entire response EXACTLY like this (no other text before or after):\n"
            f"<think>\n"
            f"[Your detailed reasoning process here]\n"
            f"</think>\n"
            f"<answer>\n"
            f"{optimal_passes_json_str}\n"
            f"</answer>"
        )

        MAX_RETRIES = 5
        MIN_THOUGHT_LENGTH = 500
        thought_process = ""
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # AWS Bedrock payload matching the Anthropic standard
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 8000,
                    "system": "You are an expert compiler optimization engineer specializing in writing Chain of thought for LLVM optimization passess.",
                    "thinking": {"type": "enabled", "budget_tokens": 4000},
                    "messages": [
                        {
                            "role": "user",
                            "content": oracle_prompt
                        }
                    ]
                }
                
                response = bedrock_client.invoke_model(
                    modelId=MODEL_ID,
                    body=json.dumps(request_body),
                    accept="application/json",
                    contentType="application/json"
                )
                
                response_body = json.loads(response.get("body").read())

                candidate = ""
                for block in response_body.get("content", []):
                    if block.get("type") == "text":
                        candidate += block.get("text", "") + "\n"

                candidate = candidate.strip()
                candidate = re.sub(r'<tool_call>.*?</tool_call>', '', candidate, flags=re.DOTALL)
                candidate = re.sub(r'</?arg_value>', '', candidate)
                candidate = re.sub(r'</?thinking>\n?', '', candidate)
                candidate = re.sub(r'</?think>\n?', '', candidate)
                candidate = candidate.split('<answer>')[0].strip()
                candidate = candidate.replace('<tool_call>', '').replace('</tool_call>', '').strip()

                missing = passes_missing_from_trace(candidate, clean_passes)

                if len(candidate) >= MIN_THOUGHT_LENGTH and not missing:
                    thought_process = candidate
                    extras = extra_passes_mentioned(candidate, clean_passes)
                    if extras:
                        print(f"[info] index {loop_index}: trace also names {extras} (not in target, likely rejected alternatives — worth a spot check)")
                        pass
                    break
                else:
                    print(f"\n[Retry {attempt}/{MAX_RETRIES}] Trace rejected for row {loop_index} (len={len(candidate)}, missing: {missing}). Retrying...")

            except Exception as e:
                print(f"\n[Retry {attempt}/{MAX_RETRIES}] API failure at row {loop_index}: {e}")

        if not thought_process:
            print(f"\n[Warning] Gave up on row {loop_index} after {MAX_RETRIES} attempts.")
            continue
            
        # -------------------------------------------------------------
        # 5. ASSEMBLE TWO SFT VERSIONS (WITH and WITHOUT Spellbook)
        # -------------------------------------------------------------
        
        rules = (
            "Rules:\n"
            "- Pick however many passes are actually useful — no bonus for more, no penalty for fewer.\n"
            "- End with your chosen passes as a JSON array of strings inside <answer>...</answer>, nothing else inside the tags.\n"
        )
        
        sys_prompt_with = (
            "You are an LLVM Compiler RL Agent. Your objective is to discover a custom sequence of optimization passes that will shrink the instruction count of the given IR even further than the standard '-Oz' optimization level.\n\n"
            "The only passes you can use along with their descriptions are:\n"
            f"{GRAND_SPELLBOOK_BLOCK}\n\n"
            f"{rules}"
        )
        
        sys_prompt_without = (
            "You are an LLVM Compiler RL Agent. Your objective is to discover a custom sequence of optimization passes that will shrink the instruction count of the given IR even further than the standard '-Oz' optimization level.\n\n"
            f"{rules}"
        )
        
        user_prompt = f"\nAutoPhase: {{ {feat_str} }}\n\nRemember: conclude your response with exactly one <answer>...</answer> block containing a JSON array of pass names as strings."
        assistant_reply = f"<think>\n{thought_process}\n</think>\n<answer>{optimal_passes_json_str}</answer>"
        
        sft_entry_with = {
            "messages": [
                {"role": "system", "content": sys_prompt_with},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_reply}
            ]
        }
        
        sft_entry_without = {
            "messages": [
                {"role": "system", "content": sys_prompt_without},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_reply}
            ]
        }
        
        file_with.write(json.dumps(sft_entry_with, ensure_ascii=False) + "\n")
        file_without.write(json.dumps(sft_entry_without, ensure_ascii=False) + "\n")
        
        file_with.flush()
        file_without.flush()

env.close()
print(f"\nProcessing successfully completed. Check {OUTPUT_WITH_SPELLBOOK} and {OUTPUT_WITHOUT_SPELLBOOK}.")