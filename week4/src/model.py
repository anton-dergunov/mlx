import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from utils import get_device
from tqdm import tqdm


PREFIX_LEN = 5  # number of prefix tokens
EMBED_DIM = 768  # GPT-2 hidden size for distilgpt2

GPT2_NAME = "distilgpt2"  # Smaller than GPT2


# -----------------------------------
# Model: CLIP + MLP + GPT2 Decoder
# -----------------------------------

class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()

        # MLP to map image embedding -> prefix tokens
        self.prefix_len = PREFIX_LEN
        # FIXME Don't hardcode
        clip_dim = 512  # self.clip_model.config.projection_dim  # 512 for CLIP-ViT-B/32
        self.mapping = nn.Sequential(
            nn.Linear(clip_dim, EMBED_DIM * PREFIX_LEN),
            nn.Tanh()
        )

        # GPT-2 decoder
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_NAME)

    def forward(self, image_embed, caption_ids, captions_attention_mask):
        B = image_embed.size(0)

        # Map CLIP image embed -> prefix
        prefix = self.mapping(image_embed).view(B, self.prefix_len, EMBED_DIM)

        # Add BOS
        bos = torch.tensor([[self.gpt2_tokenizer.bos_token_id]] * B).to(image_embed.device)
        caption_input = torch.cat([bos, caption_ids], dim=1)  # shift right

        # Embed text
        caption_embeds = self.gpt2_model.transformer.wte(caption_input)

        # Concatenate prefix + caption
        inputs_embeds = torch.cat([prefix, caption_embeds], dim=1)

        # Labels: prefix is ignored, BOS to EOS are real targets
        ignore = torch.full((B, self.prefix_len), -100).to(image_embed.device)
        labels = torch.cat([ignore, caption_ids, bos], dim=1)

        prefix_attention_mask = torch.ones(B, self.prefix_len + 1).to(image_embed.device)
        attention_mask = torch.cat((prefix_attention_mask, captions_attention_mask), dim=1)

        outputs = self.gpt2_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, image_embed):
        B = image_embed.size(0)

        # Map CLIP image embed -> prefix
        prefix = self.mapping(image_embed).view(B, self.prefix_len, EMBED_DIM)

        # Add BOS
        bos = torch.tensor([[self.gpt2_tokenizer.bos_token_id]] * B).to(image_embed.device)

        # Embed text
        bos_embeds = self.gpt2_model.transformer.wte(bos)

        # Concatenate prefix + caption
        inputs_embeds = torch.cat([prefix, bos_embeds], dim=1)

        generated = self.gpt2_model.generate(
            inputs_embeds=inputs_embeds,
            max_length=50,  # FIXME don't hardcode it
            eos_token_id=self.gpt2_tokenizer.eos_token_id
        )

        return generated
