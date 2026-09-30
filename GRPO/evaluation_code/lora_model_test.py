import os

# ==========================================
# 0. vLLM ENVIRONMENT CONFIGURATION
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
import random
import torch
import compiler_gym
from tqdm import tqdm
from unsloth import FastLanguageModel
from vllm import SamplingParams
from vllm.lora.request import LoRARequest # <--- REQUIRED

# ==========================================
# 1. CONFIGURATION
# ==========================================

# 1. Base Model
BASE_SFT_MODEL = "bcoding/deepseek-R1-0528-Qwen3-8b-base-16-llvm-sft-trained-16bit-epoch-2-merged"

# 2. LoRA Adapter to apply on top
GRPO_CHECKPOINT_DIR = "compiler-llvm-deepseek-8b-grpo-checkpoints/checkpoint-1550"

# 3. Output File
OUTPUT_FILE = "results_lora_grpo.json"  

IR_CACHE_DIR = "ir_cache"
SFT_TRAINED_NAMES_FILE = "sft_trained_names.json"
NUM_TEST_BENCHMARKS = 250
LORA_RANK = 32

# ==========================================
# 2. PROMPT CONSISTENCY
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
        buckets.setdefault(desc.rstrip('.'), []).append(p)
    return "\n".join(f"  - {desc}: {', '.join(sorted(names))}" for desc, names in sorted(buckets.items()))

SYS_PROMPT = (
    "You are an LLVM Compiler RL Agent. Your objective is to discover a custom sequence of optimization passes that will shrink the instruction count of the given IR even further than the standard '-Oz' optimization level.\n\n"
    "The only passes you can use along with their descriptions are:\n"
    f"{format_spellbook_by_category()}\n\n"
    "Rules:\n"
    "- Pick however many passes are actually useful — no bonus for more, no penalty for fewer.\n"
    "- End with your chosen passes as a JSON array of strings inside <answer>...</answer>, nothing else inside the tags.\n"
)

def get_unseen_local_files():
    excluded_set = set()
    if os.path.exists(SFT_TRAINED_NAMES_FILE):
        with open(SFT_TRAINED_NAMES_FILE, "r") as f:
            excluded_set = set(json.load(f))
            
    available_files = []
    for fname in os.listdir(IR_CACHE_DIR):
        if fname.endswith(".ll") and fname[:-3] not in excluded_set:
            available_files.append((fname[:-3], os.path.join(IR_CACHE_DIR, fname)))
            
    available_files.sort()
    random.seed(42)
    random.shuffle(available_files)
    return available_files[:NUM_TEST_BENCHMARKS]

def main():
    dataset = []
    print(f"\n[+] Extracting Autophase Features for {NUM_TEST_BENCHMARKS} files...")
    
    with compiler_gym.make("llvm-v0") as env:
        test_files = get_unseen_local_files()
        for base_name, filepath in tqdm(test_files, desc="Gym Extraction"):
            try:
                benchmark = env.make_benchmark(filepath)
                env.reset(benchmark=benchmark)
                input_count = env.observation["IrInstructionCount"]
                oz_count = env.observation["IrInstructionCountOz"]
                autophase_dict = env.observation["AutophaseDict"]
                
                feat_str = ", ".join(f"'{k}': {v}" for k, v in autophase_dict.items())
                user_prompt = f"\nAutoPhase: {{ {feat_str} }}\n\nRemember: conclude your response with exactly one <answer>...</answer> block containing a JSON array of pass names as strings."
                
                dataset.append({
                    "filename": base_name,
                    "filepath": filepath,
                    "input_count": input_count,
                    "oz_count": oz_count,
                    "messages": [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt}]
                })
            except Exception:
                continue

    if not dataset:
        print("No valid benchmarks. Exiting.")
        return

    print(f"\n[+] Booting vLLM Engine for ({BASE_SFT_MODEL}) with LoRA Enabled...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_SFT_MODEL,
        max_seq_length=6000,
        dtype=torch.bfloat16,
        load_in_4bit=False, 
        fast_inference=True,         
        enforce_eager=True, 
        enable_lora=True,             # <--- REQUIRES THIS
        max_lora_rank=LORA_RANK,      # <--- REQUIRES THIS
        gpu_memory_utilization=0.9,  
    )
    
    texts = [tokenizer.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False) for d in dataset]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=50, max_tokens=3048)

    print(f"\n[+] Generating Responses using LoRA: {GRPO_CHECKPOINT_DIR}...")
    
    # <--- THE MISSING PIECE --->
    lora_req = LoRARequest("grpo_adapter", 1, GRPO_CHECKPOINT_DIR) 
    outputs = model.fast_generate(texts, sampling_params=sampling_params, lora_request=lora_req)
    
    for idx, out in enumerate(outputs):
        dataset[idx]["model_response"] = out.outputs[0].text if hasattr(out, "outputs") else str(out)
        del dataset[idx]["messages"] 
        
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n✅ Generation complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()