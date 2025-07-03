import torch
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


PREFIX_LEN = 5  # number of prefix tokens


# -----------------------------------
# Model: CLIP + MLP + GPT2 Decoder
# -----------------------------------

class ImageCaptioningModel(nn.Module):
    def __init__(self, image_dim=512, decoder_type="gpt2"):
        super().__init__()

        self.decoder_type = decoder_type
        if decoder_type == "gpt2":
            self.embed_dim = 768  # GPT-2 hidden size for distilgpt2
            gpt_name = "distilgpt2"

            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.lm = GPT2LMHeadModel.from_pretrained(gpt_name)
            self.lm.loss_type = "ForCausalLM"
    
        elif decoder_type == "qwen":
            self.embed_dim = 4096  # Qwen hidden dim
            gpt_name = "Qwen/Qwen3-0.6B-Base"

            self.tokenizer = AutoTokenizer.from_pretrained(gpt_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(gpt_name, torch_dtype=torch.float16)
            base_model.resize_token_embeddings(len(self.tokenizer))  # required after adding PAD token

            # TODO Expose the params in the config
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            self.lm = get_peft_model(base_model, lora_config)
            self.lm.print_trainable_parameters()  # FIXME print for other models as well

        elif decoder_type == "custom":
            self.embed_dim = None  # FIXME
            pass

        else:
            raise NotImplementedError(f"Unsupported decoder: {decoder_type}")

        # MLP to map image embedding -> prefix tokens
        self.prefix_len = PREFIX_LEN

        self.mapping = nn.Sequential(
            nn.Linear(image_dim, self.embed_dim * PREFIX_LEN),
            nn.Tanh()
        )

    def forward(self, image_embed, caption_ids, captions_attention_mask):
        B = image_embed.size(0)
        device = image_embed.device

        # Map CLIP image embed -> prefix
        prefix = self.mapping(image_embed).view(B, self.prefix_len, self.embed_dim)

        bos = torch.full((B, 1), self.tokenizer.bos_token_id).to(device)
        eos = torch.full((B, 1), self.tokenizer.eos_token_id).to(device)

        # INPUT: prefix + BOS + text + EOS
        caption_input = torch.cat([bos, caption_ids, eos], dim=1)
        caption_embeds = self.lm.transformer.wte(caption_input) if self.decoder_type != "qwen" else self.lm.model.embed_tokens(caption_input)
        inputs_embeds = torch.cat([prefix, caption_embeds], dim=1)

        # LABELS: ignore prefix and BOS + text + EOS (labels are shifted inside the model)
        # (see https://github.com/huggingface/transformers/blob/b31e9d19a6607aafdd921bc592897900712ba61d/src/transformers/models/gpt2/modeling_gpt2.py#L1339)
        ignore = torch.full((B, self.prefix_len + 1), -100).to(device)
        labels = torch.cat([ignore, caption_ids, eos], dim=1)

        # ATTENTION MASK: ones for prefix + BOS + text + EOS
        prefix_attention_mask = torch.ones(B, self.prefix_len + 1).to(device)
        attention_mask = torch.cat([prefix_attention_mask, captions_attention_mask], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(B, 1).to(device)], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, image_embed):
        B = image_embed.size(0)
        device = image_embed.device

        # Map CLIP image embed -> prefix
        prefix = self.mapping(image_embed).view(B, self.prefix_len, self.embed_dim)

        # Add BOS and embed
        bos = torch.tensor([[self.tokenizer.bos_token_id]] * B).to(device)
        bos_embeds = self.lm.transformer.wte(bos) if self.decoder_type != "qwen" else self.lm.model.embed_tokens(bos)

        # Concatenate prefix + caption
        inputs_embeds = torch.cat([prefix, bos_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2]).to(device)

        generated = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=50,  # FIXME don't hardcode it
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return generated
