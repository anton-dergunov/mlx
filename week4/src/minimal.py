import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from utils import get_device
from tqdm import tqdm

# -----------------------------------
# Config
# -----------------------------------
DEVICE = get_device()
print(f"Device: {DEVICE}")

CLIP_NAME = "openai/clip-vit-base-patch32"
GPT2_NAME = "distilgpt2"  # Smaller than GPT2

PREFIX_LEN = 5  # number of prefix tokens
EMBED_DIM = 768  # GPT-2 hidden size for distilgpt2

# -----------------------------------
# Model: CLIP + MLP + GPT2 Decoder
# -----------------------------------

class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CLIP for image
        self.clip_model = CLIPModel.from_pretrained(CLIP_NAME)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        for p in self.clip_model.parameters():
            p.requires_grad = False  # freeze CLIP

        # MLP to map image embedding -> prefix tokens
        self.prefix_len = PREFIX_LEN
        clip_dim = self.clip_model.config.projection_dim  # 512 for CLIP-ViT-B/32
        self.mapping = nn.Sequential(
            nn.Linear(clip_dim, EMBED_DIM * PREFIX_LEN),
            nn.Tanh()
        )

        # GPT-2 decoder
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_NAME)

    def forward(self, images, captions_input_ids, captions_attention_mask):
        # 1) Get image embedding from CLIP
        clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(DEVICE)
        image_embeds = self.clip_model.get_image_features(**clip_inputs)

        # 2) Map to prefix tokens
        prefix = self.mapping(image_embeds)  # (B, prefix_len * embed_dim)
        prefix = prefix.view(-1, self.prefix_len, EMBED_DIM)  # (B, prefix_len, embed_dim)

        # 3) Get caption input embeddings
        caption_embeds = self.gpt2_model.transformer.wte(captions_input_ids)

        # 4) Concatenate prefix + caption
        inputs_embeds = torch.cat((prefix, caption_embeds), dim=1)
        B, prefix_seq, _ = prefix.shape

        # 5) Adjust attention mask
        B = captions_attention_mask.size(0)
        prefix_attention_mask = torch.ones(B, self.prefix_len).to(DEVICE)
        attention_mask = torch.cat((prefix_attention_mask, captions_attention_mask), dim=1)

        # FIXME Do this in collate
        # Fix: labels must be same length as inputs_embeds
        # So we pad labels with -100 for prefix part
        pad_labels = torch.full((B, prefix_seq), -100).to(DEVICE)  # -100 means ignore in loss
        labels = torch.cat((pad_labels, captions_input_ids), dim=1)

        # 6) Pass through GPT-2
        outputs = self.gpt2_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# -----------------------------------
# Load dataset
# -----------------------------------

dataset = load_dataset("nlphuji/flickr30k")['test']  # small for test

# -----------------------------------
# Collate function
# -----------------------------------

def collate_fn(batch, tokenizer):
    images = [b["image"] for b in batch]
    captions = [b["caption"][0] for b in batch]  # take first caption
    encodings = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=40
    )
    return images, encodings["input_ids"], encodings["attention_mask"]

# -----------------------------------
# DataLoader
# -----------------------------------

model = ImageCaptioningModel().to(DEVICE)
tokenizer = model.gpt2_tokenizer

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer)
)

# -----------------------------------
# Training loop
# -----------------------------------

optimizer = torch.optim.AdamW(model.mapping.parameters(), lr=1e-4)
model.gpt2_model.train()

EPOCHS = 5

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    total_loss = 0

    for images, input_ids, attention_mask in tqdm(loader, desc="Training"):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(images, input_ids, attention_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train loss: {total_loss / len(loader):.4f}")

print("Done!")

# -----------------------------------
# Test generation
# -----------------------------------

model.eval()
with torch.no_grad():
    for i in range(50):
        example = dataset[i]
        image = example["image"]
        print("Actual captions:")
        for caption in example["caption"]:
            print(caption)

        clip_inputs = model.clip_processor(images=image, return_tensors="pt").to(DEVICE)
        image_embed = model.clip_model.get_image_features(**clip_inputs)
        prefix = model.mapping(image_embed).view(-1, PREFIX_LEN, EMBED_DIM)

        generated = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)
        generated_embeds = model.gpt2_model.transformer.wte(generated)

        for _ in range(20):
            inputs_embeds = torch.cat((prefix, generated_embeds), dim=1)
            outputs = model.gpt2_model(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)
            generated_embeds = model.gpt2_model.transformer.wte(generated)

        print("Generated caption:")
        print(tokenizer.decode(generated[0]))
        print()
