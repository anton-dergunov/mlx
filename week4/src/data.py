import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer
from datasets import load_dataset
import pickle
from tqdm import tqdm
from functools import partial
import os


CLIP_NAME = "openai/clip-vit-base-patch32"
GPT2_NAME = "distilgpt2"  # Smaller than GPT2


class FlickrPrecomputedDataset(Dataset):
    def __init__(self, cache_dir, flickr_dataset, clip_model, clip_processor, tokenizer, device):
        self.images = []
        self.captions = []

        os.makedirs(cache_dir, exist_ok=True)        
        cache_file = os.path.join(cache_dir, "flickr_50k.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached dataset: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.images, self.captions = pickle.load(f)
        else:
            print(f"Generating composite dataset: {cache_file}")
            self._precompute(flickr_dataset, clip_model, clip_processor, tokenizer, device)
            with open(cache_file, 'wb') as f:
                pickle.dump((self.images, self.captions), f)


    def _precompute(self, flickr_dataset, clip_model, clip_processor, tokenizer, device):
        clip_model = clip_model.to(device)
        clip_model.eval()

        with torch.no_grad():
            for row in tqdm(flickr_dataset, desc="Precomputing"):
                processed = clip_processor(images=row['image'], return_tensors="pt")
                processed = {k: v.to(device) for k, v in processed.items()}
                image_embed = clip_model.get_image_features(**processed)
                self.images.append(image_embed.cpu())

                for caption in row['caption']:
                    encoding = tokenizer(caption)
                    self.captions.append((encoding["input_ids"], len(self.images) - 1))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption, image_idx = self.captions[idx]
        return self.images[image_idx], caption


def collate_fn(pad_token, batch):
    images, captions = zip(*batch)
    captions = [torch.tensor(caption) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_token)
    attention_mask = (captions != pad_token).long()
    images = torch.cat(images, dim=0)
    return images, captions, attention_mask


def create_flickr_dataloaders(device, cache_dir, valid_fraction=0.2, batch_size=4, num_workers=8):
    flickr_dataset = load_dataset("nlphuji/flickr30k")['test']  # only the test dataset is available

    clip_model = CLIPModel.from_pretrained(CLIP_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME, use_fast=False)  # slow processor was saved with this model

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    dataset = FlickrPrecomputedDataset(
        cache_dir,
        flickr_dataset,
        clip_model,
        clip_processor,
        gpt2_tokenizer,
        device)
    
    total_size = len(dataset)
    valid_size = int(total_size * valid_fraction)
    train_size = total_size - valid_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    collate = partial(collate_fn, gpt2_tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    return train_loader, valid_loader
