import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

SFT_ADAPTER_PATH = "models/qwen3_sft_lora"
REWARD_ADAPTER_PATH = "models/qwen3_reward_lora"
POLICY_OUTPUT_PATH = "models/qwen3_policy_lora"

MAX_LEN = 256
BATCH_SIZE = 1
EPOCHS = 1
PRINT_EVERY = 10  # Print every N rollouts

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
        self.reward_head = nn.Linear(base_lm.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = output.hidden_states[-1]  # (batch, seq_len, hidden)
        rewards = self.reward_head(last_hidden[:, -1, :]).squeeze(-1)  # (batch,)
        return rewards

reward_model = RewardModel(base_model)
reward_model = PeftModel.from_pretrained(reward_model, REWARD_ADAPTER_PATH)
reward_model = reward_model.to(DEVICE)
reward_model.eval()

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

# Load prompts
dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")

running_loss = 0.0

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        ex = dataset[i]
        prompt = ex["prompt"]
        if not prompt.strip().endswith("\nTL;DR:"):
            print("WARNING: prompt does not end with TL;DR:, skipping.")
            continue
        prompt = "Summarize:\n" + prompt.strip() + " "

        # Encode prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Generate summary
        output = policy_model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_LEN,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = output[0][input_ids.shape[1]:]
        generated = tokenizer.decode(completion, skip_special_tokens=True)

        # Compute reward
        full_ids = torch.cat([input_ids, completion.unsqueeze(0)], dim=1)
        attention_mask = torch.ones_like(full_ids).to(DEVICE)
        reward = reward_model(full_ids, attention_mask).detach()

        # Compute log probs under policy
        outputs = policy_model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels=full_ids
        )
        logprobs = -outputs.loss  # log likelihood is -loss for LM

        # REINFORCE loss: negative reward * log prob
        rl_loss = -reward * logprobs

        rl_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += rl_loss.item()

        # Print example every N steps
        if i % PRINT_EVERY == 0:
            avg_loss = running_loss / PRINT_EVERY
            running_loss = 0.0
            print(f"=== Step {i} ===")
            print(f"**Prompt**: {prompt[:200]}...")
            print(f"**Generated**: {generated[:200]}...")
            print(f"**Reward**: {reward.item():.4f}")
            print(f"**Running RL loss**: {avg_loss:.4f}")

# Save only LoRA weights
policy_model.save_pretrained(POLICY_OUTPUT_PATH)
print(f"Saved policy LoRA adapter to {POLICY_OUTPUT_PATH}")
