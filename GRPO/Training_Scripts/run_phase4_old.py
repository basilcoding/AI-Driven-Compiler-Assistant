import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import shutil
import tempfile
import torch
import multiprocessing as mp # If you need it here
from datasets import load_from_disk
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel
from reward_system import format_reward_func, performance_reward_func
from transformers.trainer_utils import get_last_checkpoint

from transformers import TrainerCallback
import json
from huggingface_hub import HfApi

# ---------------------------------------------------------
# Define all constants, classes, and functions up here
# ---------------------------------------------------------
LORA_RANK = 32
# SFT_MODEL_PATH = "bcoding/deepseek-llvm-sft-lora-7B"
SFT_MODEL_PATH = "bcoding/deepseek-R1-0528-Qwen3-8b-base-16-llvm-sft-trained-16bit-epoch-2-merged"


# ---------------------------------------------------------
# WRAP YOUR EXECUTION LOGIC IN THE MAIN GUARD
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_PATH,
        max_seq_length=6000, 
        load_in_4bit=False,
        dtype=torch.bfloat16,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,   # <-- pushed up, see below
    )

    # 2. Attaching adapters since its a merged model
    if not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    # SFT_MODEL_PATH is an unmerged LoRA checkpoint — attaching adapters
    # automatically is the ONLY correct outcome here. If this assert fires,
    # stop: do not let get_peft_model() paper over it. That would silently
    # start GRPO from a random, untrained adapter on the raw base model,
    # discarding every epoch of SFT.
    # assert isinstance(model, PeftModel), (
    #     f"{SFT_MODEL_PATH} did not load as a PeftModel — the SFT LoRA "
    #     "adapter failed to attach. Check that this folder actually contains "
    #     "adapter_config.json + adapter weights before proceeding."
    # )

    # 3. Load Dataset
    dataset = load_from_disk("./compiler_dataset_train")
    dataset = dataset.map(
        lambda r: {"ir_path": os.path.abspath(r["ir_relpath"])},
        desc="resolving absolute IR paths",
    )

    # Define a custom callback to save logs safely to a file
    class LogToFileCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                with open("training_metrics.txt", "a") as f:
                    f.write(f"Step {state.global_step}: {json.dumps(logs)}\n")


    class UploadToHFCallback(TrainerCallback):

        def __init__(self):
            self.futures = []
            self.api = HfApi()  # Created once, keeps uploads on one ordered queue
            self._tmp_files = []  # track snapshot copies so we can clean them up later

        def on_save(self, args, state, control, **kwargs):
            try:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

                # Define Hugging Face repositories
                model_repo_id = "bcoding/compiler-llvm-deepseek-8b-grpo-checkpoints"
                dataset_repo_id = "bcoding/compiler-llvm-deepseek-8b-grpo-logs"

                # ==========================================
                # 1. Queue Model Checkpoint Upload
                # ==========================================
                if not self.api.repo_exists(repo_id=model_repo_id, repo_type="model"):
                    self.api.create_repo(repo_id=model_repo_id, repo_type="model", private=True)

                if os.path.exists(ckpt_dir):
                    print(f"\n[Callback] Queuing model step {state.global_step} for background upload...")
                    future_model = self.api.upload_folder(
                        folder_path=ckpt_dir,
                        repo_id=model_repo_id,
                        repo_type="model",
                        path_in_repo=f"checkpoint-{state.global_step}",  # Isolates each checkpoint's folder
                        run_as_future=True,
                    )
                    self.futures.append((f"Model step {state.global_step}", future_model))

                # ==========================================
                # 2. Queue Tally and Log Uploads (as Dataset)
                # ==========================================
                if not self.api.repo_exists(repo_id=dataset_repo_id, repo_type="dataset"):
                    self.api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=True)

                tally_file = "tally_state.json"
                log_file = os.path.expanduser("~/logs/train_capture.log")  # '~' expanded safely

                # --- Tally state: snapshot NOW, before queuing, since the file keeps changing ---
                if os.path.exists(tally_file):
                    tally_snapshot = os.path.join(
                        tempfile.gettempdir(), f"tally_state_{state.global_step}.json"
                    )
                    shutil.copy2(tally_file, tally_snapshot)  # synchronous copy at snapshot time
                    self._tmp_files.append(tally_snapshot)

                    future_tally = self.api.upload_file(
                        path_or_fileobj=tally_snapshot,
                        path_in_repo=f"checkpoint-{state.global_step}/tally_state.json",
                        repo_id=dataset_repo_id,
                        repo_type="dataset",
                        run_as_future=True,
                    )
                    self.futures.append((f"Tally step {state.global_step}", future_tally))

                # --- Log file: uploaded directly, no snapshot copy (saves time; accepts
                # that a lagging upload queue could grab slightly newer log content than
                # existed at this exact step) ---
                if os.path.exists(log_file):
                    future_log = self.api.upload_file(
                        path_or_fileobj=log_file,
                        path_in_repo=f"checkpoint-{state.global_step}/train_capture.log",
                        repo_id=dataset_repo_id,
                        repo_type="dataset",
                        run_as_future=True,
                    )
                    self.futures.append((f"Log step {state.global_step}", future_log))

            except Exception as e:
                print(f"❌ [Callback] Failed to queue upload: {e}")

        def on_train_end(self, args, state, control, **kwargs):
            for name, future in self.futures:
                try:
                    future.result()
                    print(f"✅ {name} upload confirmed.")
                except Exception as e:
                    print(f"❌ {name} upload failed: {e}")

            # Clean up local snapshot copies now that all uploads are confirmed
            for f in self._tmp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as e:
                    print(f"⚠️ Could not remove temp file {f}: {e}")
    
    # from vllm import SamplingParams
    # vllm_sampling_params = SamplingParams(
    #     temperature=0.8,
    #     seed=3407,
    #     stop=[tokenizer.eos_token],
    # )

    # 4. Configure Trainer
    training_args = GRPOConfig(
        # vllm_sampling_params = vllm_sampling_params,
        output_dir="compiler-llvm-deepseek-8b-grpo-checkpoints",
        learning_rate=1e-5,
        optim = "adamw_8bit",
        # max_steps=2300,
        save_steps=10,         # Triggers a save every 10 steps
        save_total_limit=3,    # Keeps only the 3 most recent checkpoints
        temperature=0.8,
        top_p=0.95,  
        logging_steps = 1,
        num_train_epochs=1,               # revisit — see timing section below
        num_generations=8,            # 32 completions per prompt
        generation_batch_size=8,      # = num_generations -> exactly 1 unique prompt per rollout round
        per_device_train_batch_size=1, # small micro-batch for the backward pass
        gradient_accumulation_steps=8, 
        max_prompt_length=2600,
        max_completion_length=3400,    # candidate to lower if you hit KV-cache OOM (see below)
        report_to="none",
        use_vllm=True,
        vllm_gpu_memory_utilization=0.25,  # bumped up from 0.5 — 32-wide KV cache needs more room
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            performance_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        callbacks=[LogToFileCallback(), UploadToHFCallback()]
    )

    # 5. Train
    print("Igniting Compiler-R1 GRPO Phase...")

    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            print(f"Resuming from {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print("Output directory exists but no checkpoints found. Starting fresh.")
            trainer.train() # <--- Added missing fallback
    else:
        print("Output directory not found. Starting fresh.")
        trainer.train()

    model.save_lora("compiler_r1_final_grpo_lora")
