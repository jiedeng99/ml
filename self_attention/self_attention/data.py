import json
import os
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


class SpeakerDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = Path(data_dir)
        self.segment_len = segment_len

        mapping_path = self.data_dir / "mapping.json"
        with mapping_path.open() as f:
            mapping = json.load(f)
        self.speaker2id = mapping["speaker2id"]

        metadata_path = self.data_dir / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)["speakers"]

        self.speaker_num = len(metadata)
        self.data = []
        for speaker, utterances in metadata.items():
            for utterance in utterances:
                self.data.append((utterance["feature_path"], self.speaker2id[speaker]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.as_tensor(mel[start : start + self.segment_len], dtype=torch.float32)
        else:
            mel = torch.as_tensor(mel, dtype=torch.float32)

        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


def collate_batch(batch):
    mels, speakers = zip(*batch)
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)
    speakers = torch.tensor(speakers, dtype=torch.long)
    return mels, speakers


def get_dataloader(data_dir, batch_size, n_workers, segment_len=128):
    dataset = SpeakerDataset(data_dir, segment_len=segment_len)
    speaker_num = dataset.get_speaker_number()
    train_len = int(0.9 * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    train_set, valid_set = random_split(dataset, lengths)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        testdata_path = self.data_dir / "testdata.json"
        with testdata_path.open() as f:
            metadata = json.load(f)
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, torch.as_tensor(mel, dtype=torch.float32)


def inference_collate_batch(batch):
    feat_paths, mels = zip(*batch)
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)
    return feat_paths, mels


def get_inference_dataloader(data_dir, batch_size=1, n_workers=8):
    dataset = InferenceDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        collate_fn=inference_collate_batch,
    )
