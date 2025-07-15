import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "CarperAI/openai_summarize_tldr"

SFT_ADAPTER_PATH = "models/qwen3_sft_lora"
REWARD_ADAPTER_PATH = "models/qwen3_reward_lora"
POLICY_OUTPUT_PATH = "models/qwen3_policy_lora"

MAX_LEN = 256
BATCH_SIZE = 1
EPOCHS = 1
PRINT_EVERY = 10  # Print every N rollouts

PPO_EPOCHS = 4
CLIP_EPS = 0.2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load SFT model as policy
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
policy_model = prepare_model_for_kbit_training(policy_model)
policy_model.train()

# Load reward model
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.base_lm = base_lm
        self.reward_head = nn.Linear(base_lm.config.hidden_size, 1, bias=False).to(base_lm.dtype)

    def forward(self, input_ids, attention_mask):
        output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = output.hidden_states[-1]  # (batch, seq_len, hidden)
        rewards = self.reward_head(last_hidden[:, -1, :]).squeeze(-1)  # (batch,)
        return rewards

# Load base model for reward
base_model_for_reward = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
reward_base = PeftModel.from_pretrained(base_model_for_reward, REWARD_ADAPTER_PATH)  # Load LoRA into base LM
reward_model = RewardModel(reward_base).to(DEVICE)  # Then wrap it
reward_model.eval()

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

# Load prompts
dataset = load_dataset(DATASET_NAME, split="train")

running_loss = 0.0

for epoch in range(EPOCHS):
    loop = tqdm(range(0, len(dataset), BATCH_SIZE))
    for i in loop:
        ex = dataset[i]
        prompt = ex["prompt"]
        if not prompt.strip().endswith("\nTL;DR:"):
            print("WARNING: prompt does not end with TL;DR:, skipping.")
            continue
        prompt = "Summarize:\n" + prompt.strip() + " "

        # Encode prompt
        prompt_encoding = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = prompt_encoding.input_ids.to(DEVICE)
        attention_mask = prompt_encoding.attention_mask.to(DEVICE)

        # === 1) Rollout ===
        with torch.no_grad():
            output = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_LEN,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = output[0][input_ids.shape[1]:]
        generated = tokenizer.decode(completion, skip_special_tokens=True)

        full_ids = torch.cat([input_ids, completion.unsqueeze(0)], dim=1)
        full_attention_mask = torch.ones_like(full_ids).to(DEVICE)

        # === 2) Reward ===
        with torch.no_grad():
            reward = reward_model(full_ids, full_attention_mask).detach()

        # === 3) Compute old log probs ===
        with torch.no_grad():
            old_outputs = policy_model(
                input_ids=full_ids,
                attention_mask=full_attention_mask
            )
            old_logits = old_outputs.logits[:, :-1, :]
            old_labels = full_ids[:, 1:]

            old_logprobs_per_token = -nn.functional.cross_entropy(
                old_logits.reshape(-1, old_logits.size(-1)),
                old_labels.reshape(-1),
                reduction='none'
            )
            old_logprobs = old_logprobs_per_token.view(old_labels.shape).sum(dim=1)

        advantage = reward  # simple version, no baseline

        # === 4) PPO update ===
        for ppo_epoch in range(PPO_EPOCHS):
            outputs = policy_model(
                input_ids=full_ids,
                attention_mask=full_attention_mask
            )
            logits = outputs.logits[:, :-1, :]
            labels = full_ids[:, 1:]

            logprobs_per_token = -nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='none'
            )
            logprobs = logprobs_per_token.view(labels.shape).sum(dim=1)

            ratio = torch.exp(logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            surrogate1 = ratio * advantage
            surrogate2 = clipped_ratio * advantage
            ppo_loss = -torch.min(surrogate1, surrogate2).mean()

            optimizer.zero_grad()
            ppo_loss.backward()
            optimizer.step()

            running_loss += ppo_loss.item()

        if i % PRINT_EVERY == 0:
            avg_loss = running_loss / max(1, PRINT_EVERY)
            running_loss = 0.0
            print(f"=== Step {i} ===")
            print(f"**Prompt**: {prompt[:200]}...")
            print(f"**Generated**: {generated[:200]}...")
            print(f"**Reward**: {reward.item():.4f}")
            print(f"**Avg PPO loss**: {avg_loss:.4f}")

# Save only LoRA weights
policy_model.save_pretrained(POLICY_OUTPUT_PATH)
print(f"Saved policy LoRA adapter to {POLICY_OUTPUT_PATH}")
