import os
import re
import json
import subprocess
import tempfile
import compiler_gym
from tqdm import tqdm

# ==========================================
# CONFIGURATION (CHANGE THESE TO MATCH YOUR FILES)
# ==========================================
FILE_A = "results_sft_lora_model.json"
FILE_B = "results_sft_base_model.json"

MODEL_A_NAME = "SFT BASE MODEL EPOCH"
MODEL_B_NAME = "SFT BASE MODEL"

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
            filepath = item_a["filepath"]
            oz_count = item_a["oz_count"]

            # Evaluate Model A
            passes_a = extract_passes(item_a["model_response"])
            count_a, _ = apply_and_count(filepath, passes_a, env)
            count_a = count_a if count_a is not None else float('inf')

            # Evaluate Model B
            passes_b = extract_passes(item_b["model_response"])
            count_b, _ = apply_and_count(filepath, passes_b, env)
            count_b = count_b if count_b is not None else float('inf')

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
    print("                 🚀 FINAL EVALUATION SUMMARY")
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


# Example output
# ======================================================================
#                  🚀 FINAL EVALUATION SUMMARY
# ======================================================================
# Total Benchmarks Processed:  250

# [ ROBUSTNESS & FORMATTING ]
#   (Failures indicate broken JSON, hallucinated passes, or opt crashes)
#   LORA MODEL Valid Compilations: 246 / 250  (4 failures)
#   SFT BASE MODEL Valid Compilations: 249 / 250  (1 failures)

# [ PERFORMANCE vs STANDARD -Oz ]
#   LORA MODEL Beat -Oz:           69 / 250 files
#   SFT BASE MODEL Beat -Oz:           56 / 250 files

# [ HEAD-TO-HEAD COMPRESSION (LORA MODEL vs SFT BASE MODEL) ]
#   LORA MODEL strictly better on: 139 files
#   SFT BASE MODEL strictly better on: 69 files
#   Tie (same instruction count):       42 files

# ⚖️ CONCLUSION: Both models are performing equally or trading blows.
# ======================================================================




# ======================================================================
#                  🚀 FINAL EVALUATION SUMMARY
# ======================================================================
# Total Benchmarks Processed:  250

# [ ROBUSTNESS & FORMATTING ]
#   (Failures indicate broken JSON, hallucinated passes, or opt crashes)
#   SFT BASE MODEL EPOCH 2 Valid Compilations: 249 / 250  (1 failures)
#   SFT BASE MODEL EPOCH 1 Valid Compilations: 245 / 250  (5 failures)

# [ PERFORMANCE vs STANDARD -Oz ]
#   SFT BASE MODEL EPOCH 2 Beat -Oz:           56 / 250 files
#   SFT BASE MODEL EPOCH 1 Beat -Oz:           29 / 250 files

# [ HEAD-TO-HEAD COMPRESSION (SFT BASE MODEL EPOCH 2 vs SFT BASE MODEL EPOCH 1) ]
#   SFT BASE MODEL EPOCH 2 strictly better on: 129 files
#   SFT BASE MODEL EPOCH 1 strictly better on: 71 files
#   Tie (same instruction count):       50 files

# 🏆 CONCLUSION: SFT BASE MODEL EPOCH 2 is the winner!
# ======================================================================


# ======================================================================
#                  🚀 FINAL EVALUATION SUMMARY
# ======================================================================
# Total Benchmarks Processed:  250

# [ ROBUSTNESS & FORMATTING ]
#   (Failures indicate broken JSON, hallucinated passes, or opt crashes)
#   SFT BASE MODEL EPOCH 2 4 BIT Valid Compilations: 247 / 250  (3 failures)
#   SFT BASE MODEL EPOCH 2 BASE 16 Valid Compilations: 249 / 250  (1 failures)

# [ PERFORMANCE vs STANDARD -Oz ]
#   SFT BASE MODEL EPOCH 2 4 BIT Beat -Oz:           45 / 250 files
#   SFT BASE MODEL EPOCH 2 BASE 16 Beat -Oz:           56 / 250 files

# [ HEAD-TO-HEAD COMPRESSION (SFT BASE MODEL EPOCH 2 4 BIT vs SFT BASE MODEL EPOCH 2 BASE 16) ]
#   SFT BASE MODEL EPOCH 2 4 BIT strictly better on: 98 files
#   SFT BASE MODEL EPOCH 2 BASE 16 strictly better on: 100 files
#   Tie (same instruction count):       52 files

# ⚠️ CONCLUSION: SFT BASE MODEL EPOCH 2 BASE 16 is the winner!
# ======================================================================


