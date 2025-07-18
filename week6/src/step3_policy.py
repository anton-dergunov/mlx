import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer warnings

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "CarperAI/openai_summarize_tldr"

SFT_ADAPTER_PATH = "models/qwen3_sft_lora"
REWARD_ADAPTER_PATH = "models/qwen3_reward_lora"
POLICY_OUTPUT_PATH = "models/qwen3_policy_lora"

MAX_PROMPT_LEN = 256
MAX_NEW_TOKENS = 64
EPOCHS = 1
BATCH_SIZE = 6

LR = 1e-5
PPO_EPOCHS = 1
CLIP_EPS = 0.2
KL_BETA = 0.1

EVAL_INTERVAL = 100
SAVE_INTERVAL = 200


# --------------------------
# LOAD MODELS & TOKENIZER
# --------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # do the padding on the left side for generate such as [PAD PAD prompt prompt prompt]

# --- 1. Load the base model only ONCE ---
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16
)
base_model = prepare_model_for_kbit_training(base_model)

# --- 2. Create the PEFT model and load all adapters ---
# Load the first adapter which becomes the default
policy_lm = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, adapter_name="policy")

# Load the other adapters without merging
policy_lm.load_adapter(SFT_ADAPTER_PATH, adapter_name="sft")
policy_lm.load_adapter(REWARD_ADAPTER_PATH, adapter_name="reward")

# --- 3. The same peft model is used for all roles ---
# We just need to ensure the correct adapter is active before a forward pass
policy_lm.to(DEVICE)

class RewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.base_lm = base_lm
        self.reward_head = nn.Linear(base_lm.config.hidden_size, 1, bias=False).to(base_lm.dtype)

    def forward(self, input_ids, attention_mask):
        # We expect the 'reward' adapter to be active on self.base_lm
        output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = output.hidden_states[-1]
        rewards = self.reward_head(last_hidden[:, -1, :]).squeeze(-1)
        return rewards

reward_model = RewardModel(policy_lm).to(DEVICE)

# Also load the reward head
head_weights_path = os.path.join(REWARD_ADAPTER_PATH, "reward_head.pt")
state_dict = torch.load(head_weights_path, map_location=DEVICE)
reward_model.reward_head.load_state_dict(state_dict)
print(f"Loaded reward head weights from: {head_weights_path}")

reward_model.eval()

# Extend policy model with value head (to provide the predicted value of the state under the policy)
class PolicyWithValue(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.base_lm = base_lm
        self.value_head = nn.Linear(base_lm.config.hidden_size, 1, bias=False).to(base_lm.dtype)

    def forward(self, input_ids, attention_mask):
        output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = output.hidden_states[-1]  # (batch, seq_len, hidden)
        values = self.value_head(last_hidden[:, -1, :]).squeeze(-1)  # (batch,)
        return output, values
    
policy_model_with_value = PolicyWithValue(policy_lm).to(DEVICE)

# At top-level
torch.backends.cuda.enable_flash_sdp(True)
torch.set_float32_matmul_precision("high")

# Compile once
policy_model_with_value = torch.compile(policy_model_with_value)
reward_model = torch.compile(reward_model)

# Update the optimizer to target only the 'policy' adapter's parameters and value_head params
optimizer = torch.optim.AdamW(
    list(filter(lambda p: p.requires_grad, policy_lm.parameters())) +
    list(policy_model_with_value.value_head.parameters()),
    lr=LR
)

# Load dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Filter + preprocess in one go
def preprocess(ex):
    if not ex["prompt"].strip().endswith("\nTL;DR:"):
        return {"keep": False}

    prompt = "Summarize:\n" + ex["prompt"].strip() + " "
    encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "keep": True
    }

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda ex: ex["keep"])
dataset = dataset.with_format("torch")

def collate_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]

    # Left pad: pad_sequence defaults to right, so reverse → pad → reverse again
    input_ids = [torch.flip(x, dims=[0]) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids = torch.flip(input_ids, dims=[1])

    attention_masks = [torch.flip(x, dims=[0]) for x in attention_masks]
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    attention_masks = torch.flip(attention_masks, dims=[1])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks
    }

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True  # this helps CPU→GPU transfer
)

global_step = 0
running_loss = 0.0

# Training loop
for epoch in range(EPOCHS):
    for batch in tqdm(loader):
        try:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # === 1) Rollout ===
            # Set the active adapter to 'policy' for generation
            policy_lm.set_adapter("policy")
            policy_lm.train() # Use train mode for generation to ensure dropout is active if it was during SFT

            with autocast(device_type=DEVICE, dtype=torch.bfloat16):
                output = policy_lm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens = MAX_NEW_TOKENS,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            completions = output[:, input_ids.shape[1]:]
            full_ids = torch.cat([input_ids, completions], dim=1)
            full_attention_mask = torch.ones_like(full_ids).to(DEVICE)

            # === 2) Reward ===
            with torch.no_grad():
                policy_lm.set_adapter("reward") # Activate reward adapter
                reward = reward_model(full_ids, full_attention_mask)

            # === 3) Old log probs ===
            with torch.no_grad():
                policy_lm.set_adapter("sft") # Activate reference SFT adapter
                old_outputs = policy_lm(input_ids=full_ids, attention_mask=full_attention_mask)
                logits = old_outputs.logits[:, :-1, :]
                labels = full_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                prompt_len = input_ids.shape[1]
                # The logprobs are for tokens from index 1 to end, so the completion starts at index (prompt_len - 1)
                completion_log_probs = token_log_probs[:, prompt_len - 1:]
                old_logprobs = completion_log_probs.sum(dim=1)

            # === 4) PPO Update ===
            # We do one forward pass with the policy model to get new logprobs and values
            with autocast(device_type=DEVICE, dtype=torch.bfloat16):
                policy_lm.set_adapter("policy") # Activate trainable policy adapter
                outputs, values = policy_model_with_value(full_ids, full_attention_mask)
                
                logits = outputs.logits[:, :-1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

                prompt_len = input_ids.shape[1]
                completion_log_probs = token_log_probs[:, prompt_len - 1:]
                new_logprobs = completion_log_probs.sum(dim=1)

                # Detach values here to stop gradients from flowing through the value loss to the policy loss
                values_detached = values.detach()

                # --- Calculate KL divergence and build the final reward signal ---
                kl_div = new_logprobs - old_logprobs # old_logprobs is already detached
                
                # Clip the raw reward from the reward model
                clipped_reward = torch.clamp(reward, -5.0, 5.0)
                
                # The final reward signal is the clipped reward minus the KL penalty
                final_reward = clipped_reward - KL_BETA * kl_div
                
                # --- Calculate advantage ---
                advantage = (final_reward - values_detached).detach()

                # --- Calculate PPO policy loss ---
                ratio = torch.exp(new_logprobs - old_logprobs) # new_logprobs has gradients, old_logprobs does not
                
                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
                
                # Minus sign because we perform gradient ascent (maximize objective)
                ppo_loss = -torch.min(surrogate1, surrogate2).mean()

                # --- Calculate value loss ---
                # The value function is trained to predict the RAW, unpenalized reward
                # This is important as the value function's goal is to predict the environment's reward
                value_loss = nn.functional.mse_loss(values, clipped_reward)

                # --- Combine losses ---
                total_loss = ppo_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ OOM at global_step={global_step} batch={input_ids.shape}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise

        global_step += 1

        if global_step % EVAL_INTERVAL == 0:
            avg_loss = running_loss / max(1, EVAL_INTERVAL)
            running_loss = 0.0

            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = tokenizer.decode(completions[0], skip_special_tokens=True)

            print(f"=== Step {global_step} ===")
            print(f"**Prompt**: {prompt_text[:200]}...")
            print(f"**Generated**: {generated_text[:200]}...")
            print(f"**Reward**: {reward[0].item():.4f}")
            print(f"**Avg PPO loss**: {avg_loss:.4f}")

        # Save LoRA checkpoint
        if global_step % SAVE_INTERVAL == 0:
            ckpt_path = f"{POLICY_OUTPUT_PATH}_ckpt_step_{global_step}"
            policy_lm.save_pretrained(ckpt_path)
            torch.save(policy_model_with_value.value_head.state_dict(), os.path.join(ckpt_path, "value_head.pt"))
            print(f"Saved interim LoRA and value head to {ckpt_path}")

# Save only LoRA weights
policy_lm.save_pretrained(POLICY_OUTPUT_PATH)
torch.save(policy_model_with_value.value_head.state_dict(), os.path.join(POLICY_OUTPUT_PATH, "value_head.pt"))
print(f"Saved policy LoRA adapter and value head to {POLICY_OUTPUT_PATH}")
