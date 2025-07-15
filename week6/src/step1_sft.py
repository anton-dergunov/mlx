import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from tqdm import tqdm

# --------------------------
# CONFIG
# --------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "CarperAI/openai_summarize_tldr"
OUTPUT_DIR = "models/qwen3_sft_lora"
BATCH_SIZE = 2
LR = 2e-5
NUM_EPOCHS = 0  # For testing
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 64
EVAL_INTERVAL = 2

# --------------------------
# LOAD MODEL & TOKENIZER
# --------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
model = model.to(DEVICE)

# --------------------------
# PREPARE LoRA
# --------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# --------------------------
# LOAD DATA
# --------------------------

dataset = load_dataset(DATASET_NAME)

# Use train and valid splits
train_data = dataset["train"]
valid_data = dataset["valid"]

# --------------------------
# TOKENIZE
# --------------------------

def preprocess(examples):
    inputs = []
    targets = []
    prompt_lens = []

    for prompt, label in zip(examples["prompt"], examples["label"]):
        if not prompt.strip().endswith("\nTL;DR:"):
            print("WARNING: prompt does not end with TL;DR:, skipping.")
            continue
        if label.strip().lower().startswith("tl;dr:"):
            print("WARNING: label has TL;DR:, skipping.")
            continue

        full_prompt = "Summarize:\n" + prompt.strip() + "\nTL;DR:"
        inputs.append(full_prompt)
        targets.append(label.strip())

    input_encodings = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True
    )
    target_encodings = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        add_special_tokens=False
    )

    input_ids = []
    labels = []
    prompt_lens = []

    for inp_ids, tgt_ids in zip(input_encodings["input_ids"], target_encodings["input_ids"]):
        ids = inp_ids + tgt_ids + [tokenizer.eos_token_id]
        label = [-100] * len(inp_ids) + tgt_ids + [tokenizer.eos_token_id]
        input_ids.append(ids)
        labels.append(label)
        prompt_lens.append(len(inp_ids))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_lens
    }

def preprocess(examples):
    inputs = ["summarize: " + doc for doc in examples["prompt"]]
    targets = [label for label in examples["label"]]

    input_encodings = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True
    )

    target_encodings = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        add_special_tokens=False  # <- don't double add EOS if your tokenizer does it.
    )

    # Concatenate and mask
    input_ids = []
    labels = []

    for inp_ids, tgt_ids in zip(input_encodings["input_ids"], target_encodings["input_ids"]):
        ids = inp_ids + tgt_ids + [tokenizer.eos_token_id]
        label = [-100] * len(inp_ids) + tgt_ids + [tokenizer.eos_token_id]
        input_ids.append(ids)
        labels.append(label)

    return {"input_ids": input_ids, "labels": labels}

train_tokenized = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
valid_tokenized = valid_data.map(preprocess, batched=True, remove_columns=valid_data.column_names)

def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    prompt_lens = [x["prompt_len"] for x in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "prompt_len": torch.tensor(prompt_lens, dtype=torch.long)
    }

train_loader = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_tokenized, batch_size=1, collate_fn=collate_fn)

# --------------------------
# OPTIMIZER & SCHEDULER
# --------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_training_steps)

# --------------------------
# TRAIN LOOP
# --------------------------

model.train()

global_step = 0
running_loss = 0.0
for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        global_step += 1

        if global_step % EVAL_INTERVAL == 0:
            avg_loss = running_loss / EVAL_INTERVAL
            print(f"\n[global_step {global_step}] Running train loss (avg over last {EVAL_INTERVAL} steps): {avg_loss:.4f}")
            running_loss = 0.0

            model.eval()

            val_sample = next(iter(valid_loader))

            val_input_ids = val_sample["input_ids"].to(DEVICE)
            val_attention_mask = val_sample["attention_mask"].to(DEVICE)
            val_prompt_len = val_sample["prompt_len"].to(DEVICE)
            val_labels = val_sample["labels"].to(DEVICE)

            # Only take the prompt portion
            prompt_input_ids = val_input_ids[:, :val_prompt_len]
            prompt_attention_mask = val_attention_mask[:, :val_prompt_len]

            generated_ids = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=MAX_TARGET_LEN,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=4
            )

            prompt_text = tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            orig_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print("=== EVAL SAMPLE ===")
            print(f"**Prompt**: {prompt_text[:200]}...")
            print(f"**Generated**: {generated_text[-200:]}\n")
            print(f"**Original**: {orig_text[-200:]}\n")

            model.train()

# --------------------------
# SAVE LoRA ONLY
# --------------------------

model.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter weights saved to {OUTPUT_DIR}")
