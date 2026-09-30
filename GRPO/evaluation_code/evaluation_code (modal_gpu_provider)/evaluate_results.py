import os
import re
import json
import warnings
import subprocess
import tempfile
import compiler_gym
from tqdm import tqdm

# Suppress the noisy NetworkX FutureWarning from CompilerGym internals
warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")

# ==========================================
# CONFIGURATION (CHANGE THESE TO MATCH YOUR FILES)
# ==========================================
FILE_A = "results_grpo_lora_model.json"
FILE_B = "results_grpo_base_model.json"

MODEL_A_NAME = "grpo_trained_model"
MODEL_B_NAME = "sft_trained_model"

LLVM_10_OPT_BIN = os.path.expanduser("~/rlc-project/llvm-10-toolchain/bin/opt")
OPT_TIMEOUT_SECONDS = 30.0

def _count_instructions(ll_path: str, env) -> int:
    env.reset(benchmark=env.make_benchmark(ll_path))
    return int(env.observation["IrInstructionCount"])

def apply_and_count(ir_path: str, passes, env):
    fd, out_ll = tempfile.mkstemp(suffix=".ll")
    os.close(fd)
    try:
        flags = [f"-{p.lstrip('-')}" for p in passes]
        cmd = [LLVM_10_OPT_BIN] + flags + [ir_path, "-S", "-o", out_ll]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=OPT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if result.returncode != 0:
            return None, "crash"
        try:
            return _count_instructions(out_ll, env), "ok"
        except Exception:
            return None, "infra"
    finally:
        if os.path.exists(out_ll):
            os.remove(out_ll)

def extract_passes(text: str):
    THINK_CLOSE_TAG = "</think>"
    ANSWER_PATTERN = re.compile(r"<answer\s*>\s*(.*?)\s*</answer\s*>", re.DOTALL | re.IGNORECASE)
    _STRAY_WRAPPER_CHARS = "[]'\"`() "
    idx = text.rfind(THINK_CLOSE_TAG)
    post_think = text[idx + len(THINK_CLOSE_TAG):] if idx != -1 else text
    matches = ANSWER_PATTERN.findall(post_think)
    if not matches:
        return []
    raw = matches[-1].strip()
    try:
        names = json.loads(raw)
        if isinstance(names, list):
            return names
    except json.JSONDecodeError:
        pass
    tokens = re.split(r"[,\n]", raw)
    names = [t.strip().strip(_STRAY_WRAPPER_CHARS).strip() for t in tokens]
    return [n for n in names if n]

def main():
    if not os.path.exists(FILE_A):
        print(f"Error: {FILE_A} not found.")
        return
    if not os.path.exists(FILE_B):
        print(f"Error: {FILE_B} not found.")
        return

    with open(FILE_A, "r") as f:
        data_a = {d["filename"]: d for d in json.load(f)}
    with open(FILE_B, "r") as f:
        data_b = {d["filename"]: d for d in json.load(f)}

    # Ensure we only compare files that exist in both outputs
    common_files = list(set(data_a.keys()).intersection(set(data_b.keys())))
    if not common_files:
        print("Error: No common benchmarks found between the two files.")
        return

    m1_valid, m2_valid = 0, 0
    m1_beat_oz, m2_beat_oz = 0, 0
    m1_wins, m2_wins, ties = 0, 0, 0

    print(f"\n[+] Evaluating {len(common_files)} benchmarks in CompilerGym...")
    
    with compiler_gym.make("llvm-v0") as env:
        for filename in tqdm(common_files, desc="Validating Output Passes"):
            item_a = data_a[filename]
            item_b = data_b[filename]
            
            # filename now holds the URI (e.g., benchmark://npb-v0/1)
            uri = item_a["filename"]
            oz_count = item_a["oz_count"]

            # Load the benchmark from CompilerGym
            try:
                env.reset(benchmark=uri)
                ir_code = env.observation["Ir"]
            except Exception as e:
                print(f"\n[!] Skipping {uri} due to load error: {e}")
                continue

            # Dump the IR code to a temporary file for your opt subprocess
            fd, base_ir_path = tempfile.mkstemp(suffix=".ll")
            with os.fdopen(fd, 'w') as f:
                f.write(ir_code)

            try:
                # Evaluate Model A
                passes_a = extract_passes(item_a["model_response"])
                count_a, _ = apply_and_count(base_ir_path, passes_a, env)
                count_a = count_a if count_a is not None else float('inf')

                # Evaluate Model B
                passes_b = extract_passes(item_b["model_response"])
                count_b, _ = apply_and_count(base_ir_path, passes_b, env)
                count_b = count_b if count_b is not None else float('inf')
            finally:
                # Clean up the temporary base IR file
                if os.path.exists(base_ir_path):
                    os.remove(base_ir_path)

            # Accumulate Stats
            if count_a != float('inf'): m1_valid += 1
            if count_b != float('inf'): m2_valid += 1
                
            if count_a < oz_count: m1_beat_oz += 1
            if count_b < oz_count: m2_beat_oz += 1
                
            if count_a < count_b: m1_wins += 1
            elif count_b < count_a: m2_wins += 1
            else: ties += 1

    # Print Report
    print("\n\n" + "="*70)
    print("                🚀 FINAL EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Benchmarks Processed:  {len(common_files)}\n")

    print("[ ROBUSTNESS & FORMATTING ]")
    print("  (Failures indicate broken JSON, hallucinated passes, or opt crashes)")
    print(f"  {MODEL_A_NAME} Valid Compilations: {m1_valid} / {len(common_files)}  ({len(common_files)-m1_valid} failures)")
    print(f"  {MODEL_B_NAME} Valid Compilations: {m2_valid} / {len(common_files)}  ({len(common_files)-m2_valid} failures)\n")

    print("[ PERFORMANCE vs STANDARD -Oz ]")
    print(f"  {MODEL_A_NAME} Beat -Oz:           {m1_beat_oz} / {len(common_files)} files")
    print(f"  {MODEL_B_NAME} Beat -Oz:           {m2_beat_oz} / {len(common_files)} files\n")

    print(f"[ HEAD-TO-HEAD COMPRESSION ({MODEL_A_NAME} vs {MODEL_B_NAME}) ]")
    print(f"  {MODEL_A_NAME} strictly better on: {m1_wins} files")
    print(f"  {MODEL_B_NAME} strictly better on: {m2_wins} files")
    print(f"  Tie (same instruction count):       {ties} files\n")
    
    if m1_wins > m2_wins and m1_valid >= m2_valid:
        print(f"🏆 CONCLUSION: {MODEL_A_NAME} is the winner!")
    elif m2_wins > m1_wins:
        print(f"⚠️ CONCLUSION: {MODEL_B_NAME} is the winner!")
    else:
        print("⚖️ CONCLUSION: Both models are performing equally or trading blows.")
        
    print("="*70 + "\n")

if __name__ == "__main__":
    main()



# ======================================================================
#                 🚀 FINAL EVALUATION SUMMARY
# ======================================================================
# Total Benchmarks Processed:  190

# [ ROBUSTNESS & FORMATTING ]
#   (Failures indicate broken JSON, hallucinated passes, or opt crashes)
#   SFT BASE MODEL EPOCH Valid Compilations: 190 / 190  (0 failures)
#   SFT BASE MODEL Valid Compilations: 190 / 190  (0 failures)

# [ PERFORMANCE vs STANDARD -Oz ]
#   grpo_trained_model Beat -Oz:           64 / 190 files
#   sft_trained_model Beat -Oz:           43 / 190 files

# [ HEAD-TO-HEAD COMPRESSION (SFT BASE MODEL EPOCH vs SFT BASE MODEL) ]
#   SFT BASE MODEL EPOCH strictly better on: 80 files
#   SFT BASE MODEL strictly better on: 17 files
#   Tie (same instruction count):       93 files

# 🏆 CONCLUSION: grpo_trained_model is the winner!
# ======================================================================









# ======================================================================
#                 FINAL EVALUATION SUMMARY
# ======================================================================
# Total Benchmarks Processed:  1661

# [ ROBUSTNESS & FORMATTING ]
#   (Failures indicate broken JSON, hallucinated passes, or opt crashes)
#   sft_trained_model Valid Compilations: 1642 / 1661  (19 failures)
#   grpo_trained_model Valid Compilations: 1653 / 1661  (8 failures)

# [ PERFORMANCE vs STANDARD -Oz ]
#   sft_trained_model Beat -Oz:           317 / 1661 files
#   grpo_trained_model Beat -Oz:           599 / 1661 files

# [ HEAD-TO-HEAD COMPRESSION (sft_trained_model vs grpo_trained_model) ]
#   sft_trained_model strictly better on: 125 files
#   grpo_trained_model strictly better on: 782 files
#   Tie (same instruction count):       754 files

# ⚠️ CONCLUSION: grpo_trained_model is the winner!
# ======================================================================