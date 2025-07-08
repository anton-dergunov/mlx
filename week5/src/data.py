import torch
import torch.nn.functional as F
from torch.nn.functional import pad
import torchaudio
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


BATCH_SIZE = 32
NUM_WORKERS = 4

# TODO Expose these parameters in config; make sure the work well with the cache
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
NUM_MELS = 64
DURATION = 4  # seconds
FIXED_LENGTH = SAMPLE_RATE * DURATION


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=NUM_MELS
)

def preprocess_batch(batch):
    waveform = torch.tensor(batch['audio']['array'], dtype=torch.float32)
    sr = batch['audio']['sampling_rate']

    # Convert to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

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
    def __init__(self, hf_dataset, batch_size=BATCH_SIZE):
        self.ds = hf_dataset
        self.batch_size = batch_size

    def get_dataloaders(self, fold):
        # TODO Find out why filter is so slow
        train_ds = UrbanSoundDataset(self.ds.filter(lambda x: x['fold'] != fold))
        test_ds = UrbanSoundDataset(self.ds.filter(lambda x: x['fold'] == fold))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)

        return train_loader, test_loader


def create_dataloaders_factory(batch_size=BATCH_SIZE):
    dataset = load_dataset("danavery/urbansound8K", split="train")

    dataset = dataset.map(
        preprocess_batch,
        remove_columns=['audio'],
        num_proc=1,  # does not work in multiple processes due to torchaudio limitations
        desc="Precomputing spectrograms"
    )

    return DataLoadersFactory(dataset, batch_size)
