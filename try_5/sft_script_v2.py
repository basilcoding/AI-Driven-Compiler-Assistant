import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. HARDWARE & CONTEXT CONFIGURATION
# ==========================================
MAX_SEQ_LENGTH = 10000
# MODEL_NAME = "unsloth/DeepSeek-R1-0528-Qwen3-8B"
MODEL_NAME = "unsloth/DeepSeek-R1-0528-Qwen3-8B"
DATASET_FILE = "sft_combined_balanced_dataset_4000.jsonl"

# ==========================================
# 2. LOAD MODEL & APPLY LoRA
# =======================================76 ``===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 3. LOAD & FORMAT DATASET
# ==========================================
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

USER_TAG = tokenizer.apply_chat_template(
    [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
    tokenize=False, add_generation_prompt=False,
).split("x", 1)[1].split("y", 1)[0]
# -> "<｜User｜>"

ASSISTANT_TAG = tokenizer.apply_chat_template(
    [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
).split("x", 1)[1]
# -> "<｜Assistant｜><think>\n"   (confirmed by your last line of output)

EOS = tokenizer.eos_token
# -> "<｜end▁of▁sentence｜>"   — note this is U+2581 "▁" between the words, not
# a regular underscore. Your corrupted version had "_"; your very first upload
# actually had this right (\u2581). This is now sourced correctly either way.

def flatten_chat(row):
    system_txt, user_txt, assistant_txt = "", "", ""
    for m in row["messages"]:
        if m["role"] == "system": system_txt = m["content"]
        elif m["role"] == "user": user_txt = m["content"]
        elif m["role"] == "assistant": assistant_txt = m["content"]

    prompt_prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_txt},
         {"role": "user", "content": user_txt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    row["text"] = prompt_prefix + assistant_txt + EOS

    ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
    return row

dataset = dataset.map(flatten_chat, num_proc=4)

# ==========================================
# 4. BUILD TRAINER
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4,
        packing = False,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 2,
        warmup_steps = 5,
        save_strategy = "epoch",
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_sft",
    ),
)

# ==========================================
# 5. MASK PROMPT TOKENS WITH -100
# ==========================================

trainer = train_on_responses_only(
    trainer,
    instruction_part = USER_TAG,
    response_part    = ASSISTANT_TAG,
)

print("========This is whats around the input_ids========\nj")
ids = trainer.train_dataset[0]["input_ids"][:5]
print(ids)
print([tokenizer.decode([i]) for i in ids])
print("BOS token id:", tokenizer.bos_token_id)

row = dict(dataset[0])
out = flatten_chat(row)
idx = out["text"].find("<｜Assistant｜>")
print("======= This is whats around the <｜Assistant｜> tag ======")
print(repr(out["text"][idx:idx+80]))


sample = trainer.train_dataset[0]
labels = sample["labels"]
kept = [t for t in labels if t != -100]
decoded_output = tokenizer.decode(kept)

print("=== Tokens the model is graded on ===")
print(decoded_output[:]) 

# Flip the assertion: we WANT it to start with <think> now!
assert decoded_output.lstrip().startswith("<think>"), \
    "Graded region is missing <think> — it should be right at the start!"
assert "</think>" in decoded_output, "Reasoning close tag missing from labels!"
assert "<answer>" in decoded_output, "Answer tag missing from labels!"

# ==========================================
# 6. START SFT TRAINING
# ==========================================
trainer.train()

model.save_pretrained("compiler_r1_sft_lora")
tokenizer.save_pretrained("compiler_r1_sft_lora")