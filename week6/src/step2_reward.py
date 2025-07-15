import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
EVAL_SIZE = 64

SFT_ADAPTER_PATH = "models/qwen3_sft_lora"  # output from Step 1
REWARD_OUTPUT_PATH = "models/qwen3_reward_lora"

BATCH_SIZE = 8
LR = 1e-5
EPOCHS = 1
MAX_LEN = 512

EVAL_INTERVAL = 1000
SAVE_INTERVAL = 1000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load base model + SFT adapter
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
model = prepare_model_for_kbit_training(model)

# Add reward head
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.base_lm = base_lm
        self.reward_head = nn.Linear(base_lm.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = output.hidden_states[-1]  # (batch, seq_len, hidden)
        # Take hidden state at last token
        rewards = self.reward_head(last_hidden[:, -1, :]).squeeze(-1)  # (batch,)
        return rewards

reward_model = RewardModel(model).to(DEVICE)

# Load preference dataset
dataset = load_dataset(DATASET_NAME)

# Use train and valid splits
train_data = dataset["train"]
valid_data = dataset["valid1"].select(EVAL_SIZE)

def preprocess(ex):
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]

    if prompt.strip().endswith("TL;DR:"):
        print("WARNING: Reward prompt ends with TL;DR:, skipping.")
        return None
    if not chosen.strip().startswith("TL;DR: "):
        print("WARNING: Chosen missing TL;DR:, skipping.")
        return None
    if not rejected.strip().startswith("TL;DR: "):
        print("WARNING: Rejected missing TL;DR:, skipping.")
        return None

    chosen_enc = tokenizer("Summarize:\n" + prompt + "\n" + chosen, truncation=True, max_length=MAX_LEN)
    rejected_enc = tokenizer("Summarize:\n" + prompt + "\n" + rejected, truncation=True, max_length=MAX_LEN)

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }

train_tokenized = train_data.map(preprocess).filter(lambda x: x is not None)
valid_tokenized = valid_data.map(preprocess).filter(lambda x: x is not None)

def collate_fn(batch):
    chosen_ids = [torch.tensor(x["chosen_input_ids"]) for x in batch]
    chosen_mask = [torch.tensor(x["chosen_attention_mask"]) for x in batch]
    rejected_ids = [torch.tensor(x["rejected_input_ids"]) for x in batch]
    rejected_mask = [torch.tensor(x["rejected_attention_mask"]) for x in batch]

    chosen_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    chosen_mask = pad_sequence(chosen_mask, batch_first=True, padding_value=0)
    rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    rejected_mask = pad_sequence(rejected_mask, batch_first=True, padding_value=0)

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
    }

train_loader = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_tokenized, batch_size=BATCH_SIZE, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(reward_model.parameters(), lr=LR)

global_step = 0
running_loss = 0.0

# Train loop
reward_model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()

        try:
            chosen_rewards = reward_model(
                input_ids=batch["chosen_input_ids"].to(DEVICE),
                attention_mask=batch["chosen_attention_mask"].to(DEVICE)
            )
            rejected_rewards = reward_model(
                input_ids=batch["rejected_input_ids"].to(DEVICE),
                attention_mask=batch["rejected_attention_mask"].to(DEVICE)
            )

            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
            loss.backward()
            optimizer.step()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM! input_ids shape: {batch['chosen_input_ids'].shape}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise

        running_loss += loss.item()
        global_step += 1

        loop.set_postfix(loss=loss.item(), running_avg=running_loss / global_step)

        # Validate
        if global_step % EVAL_INTERVAL == 0:
            reward_model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in valid_loader:
                    val_chosen = reward_model(
                        input_ids=val_batch["chosen_input_ids"].to(DEVICE),
                        attention_mask=val_batch["chosen_attention_mask"].to(DEVICE)
                    )
                    val_rejected = reward_model(
                        input_ids=val_batch["rejected_input_ids"].to(DEVICE),
                        attention_mask=val_batch["rejected_attention_mask"].to(DEVICE)
                    )
                    val_loss = -torch.nn.functional.logsigmoid(val_chosen - val_rejected).mean()
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"\n[global_step {global_step}] Val loss: {avg_val_loss:.4f}")
            reward_model.train()

        # Save LoRA checkpoint
        if global_step % SAVE_INTERVAL == 0:
            ckpt_path = f"{REWARD_OUTPUT_PATH}_ckpt_step_{global_step}"
            model.save_pretrained(ckpt_path)
            print(f"Saved interim LoRA to {ckpt_path}")

# Save only LoRA
model.save_pretrained(REWARD_OUTPUT_PATH)
print(f"Reward LoRA adapter saved to {REWARD_OUTPUT_PATH}")
