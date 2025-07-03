import torch
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


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
        self.gpt2_model.loss_type = "ForCausalLM"

    def forward(self, image_embed, caption_ids, captions_attention_mask):
        B = image_embed.size(0)
        device = image_embed.device

        # Map CLIP image embed -> prefix
        prefix = self.mapping(image_embed).view(B, self.prefix_len, EMBED_DIM)

        bos = torch.full((B, 1), self.gpt2_tokenizer.bos_token_id).to(device)
        eos = torch.full((B, 1), self.gpt2_tokenizer.eos_token_id).to(device)

        # INPUT: prefix + BOS + text + EOS
        caption_input = torch.cat([bos, caption_ids, eos], dim=1)
        caption_embeds = self.gpt2_model.transformer.wte(caption_input)
        inputs_embeds = torch.cat([prefix, caption_embeds], dim=1)

        # LABELS: ignore prefix and BOS + text + EOS (labels are shifted inside the model)
        # (see https://github.com/huggingface/transformers/blob/b31e9d19a6607aafdd921bc592897900712ba61d/src/transformers/models/gpt2/modeling_gpt2.py#L1339)
        ignore = torch.full((B, self.prefix_len + 1), -100).to(device)
        labels = torch.cat([ignore, caption_ids, eos], dim=1)

        # ATTENTION MASK: ones for prefix + BOS + text + EOS
        prefix_attention_mask = torch.ones(B, self.prefix_len + 1).to(device)
        attention_mask = torch.cat([prefix_attention_mask, captions_attention_mask], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(B, 1).to(device)], dim=1)

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

        attention_mask = torch.ones(inputs_embeds.shape[:2]).to(image_embed.device)

        generated = self.gpt2_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=50,  # FIXME don't hardcode it
            eos_token_id=self.gpt2_tokenizer.eos_token_id,
            pad_token_id=self.gpt2_tokenizer.eos_token_id
        )

        return generated
