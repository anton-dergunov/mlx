import torch
import torch.nn.functional as F
from torch.nn.functional import pad
import torchaudio
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from functools import partial


DEFAULT_DATASET_NAME = "danavery/urbansound8K"
DEFAULT_BATCH_SIZE = 32
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_NUM_MELS = 64


def preprocess_batch(mel_transform, sample_rate, batch):
    waveform = torch.tensor(batch['audio']['array'], dtype=torch.float32)
    sr = batch['audio']['sampling_rate']

    # Convert to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    mel = mel_transform(waveform)
    mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    mel_db = mel_db.unsqueeze(0).numpy()

    batch['mel'] = mel_db
    batch['label'] = batch['classID']
    return batch


class UrbanSoundDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        mel = torch.tensor(item['mel'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)

        # TODO Add augmentation

        return mel, label


def pad_collate(batch):
    """
    Pads spectrograms to the longest in the batch.
    """
    specs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    # Find max time dim
    max_len = max([spec.shape[-1] for spec in specs])

    padded_specs = []
    for spec in specs:
        pad_amt = max_len - spec.shape[-1]
        # Pad on time dim (last)
        spec_padded = pad(spec, (0, pad_amt))
        padded_specs.append(spec_padded)

    specs_batch = torch.stack(padded_specs)
    return specs_batch, labels


class DataLoadersFactory:
    def __init__(self, hf_dataset, batch_size=DEFAULT_BATCH_SIZE):
        self.ds = hf_dataset
        self.batch_size = batch_size

    def get_dataloaders(self, fold):
        # TODO Find out why filter is so slow
        train_ds = UrbanSoundDataset(self.ds.filter(lambda x: x['fold'] != fold))
        test_ds = UrbanSoundDataset(self.ds.filter(lambda x: x['fold'] == fold))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)

        return train_loader, test_loader


def create_dataloaders_factory(
    dataset_name=DEFAULT_DATASET_NAME,
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
    num_mels=DEFAULT_NUM_MELS,
    batch_size=DEFAULT_BATCH_SIZE,
):

    dataset = load_dataset(dataset_name, split="train")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=num_mels
    )
    preprocess_fn = partial(preprocess_batch, mel_transform, sample_rate)

    # https://huggingface.co/docs/datasets/en/cache#enable-or-disable-caching
    dataset = dataset.map(
        preprocess_fn,
        remove_columns=['audio'],
        num_proc=1,  # does not work in multiple processes due to torchaudio limitations
        desc="Precomputing spectrograms"
    )

    return DataLoadersFactory(dataset, batch_size)
