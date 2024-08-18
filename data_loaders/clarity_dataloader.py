import os
from os.path import *
import random
from pathlib import Path
from typing import *

import numpy as np
import json
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.io import wavfile

from scipy.signal import resample_poly

import torch.nn.functional as F

def pad_collate_fn(batch):
    # Unpack the batch into individual components
    signals, references, paras = zip(*batch)

    # Determine the maximum length in the batch
    max_len = max(signal.shape[1] for signal in signals)

    # Pad signals and references to the maximum length
    padded_signals = [F.pad(signal, (0, max_len - signal.shape[1])) for signal in signals]
    padded_references = [F.pad(reference, (0, max_len - reference.shape[1])) for reference in references]

    # Stack the padded signals and references
    signals = torch.stack(padded_signals)
    references = torch.stack(padded_references)

    return signals, references, paras

class ClarityDataset(Dataset):
    def __init__(self, data, data_path):
        self.data = data
        self.data_path = data_path
        self.scene_listener_pairs = [(item['scene'], item['listener']['ID']) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        scene, listener_id = self.scene_listener_pairs[index]

        # Load the audio files for the selected scene
        sample_rate, signal_ch1 = wavfile.read(
            Path(self.data_path) / f"{scene}_mix_CH1.wav"
        )
        _, signal_ch2 = wavfile.read(
            Path(self.data_path) / f"{scene}_mix_CH2.wav"
        )
        _, signal_ch3 = wavfile.read(
            Path(self.data_path) / f"{scene}_mix_CH3.wav"
        )
        _, reference = wavfile.read(
            Path(self.data_path) / f"{scene}_reference.wav"
        )

        # Normalize signals to float32 range [-1, 1]
        signal_ch1 = (signal_ch1 / 32768.0).astype(np.float32)
        signal_ch2 = (signal_ch2 / 32768.0).astype(np.float32)
        signal_ch3 = (signal_ch3 / 32768.0).astype(np.float32)
        reference = (reference / 32768.0).astype(np.float32)

        # Stack signals
        try:
            rank_zero_info(f"Shape of signal_ch1: {signal_ch1.shape}")
            rank_zero_info(f"Shape of signal_ch2: {signal_ch2.shape}")
            rank_zero_info(f"Shape of signal_ch3: {signal_ch3.shape}")
            rank_zero_info(f"Shape of reference: {reference.shape}")
            signals = np.stack([signal_ch1, signal_ch2, signal_ch3], axis=0)
        except:
            import pdb;pdb.post.mortem()
        reference = reference[np.newaxis, :]

        paras = {
            'index': index,
            'target': self.data[index]['target'],
            'sample_rate': sample_rate,
            'dataset': 'train',
            'snr': self.data[index].get('SNR', None)
        }

        return (
            torch.as_tensor(signals, dtype=torch.float32),  # shape [chn, time]
            torch.as_tensor(reference, dtype=torch.float32),  # shape [1, chn, time]
            paras,
        )

class ClarityDataLoader(LightningDataModule):
    def __init__(self, train_json_file, test_json_file, batch_size=32, seed=42, train_data_path="", test_data_path="", num_workers=0):
        super().__init__()
        self.train_json_file = train_json_file
        self.test_json_file = test_json_file
        self.batch_size = batch_size
        self.seed = seed
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.num_workers=num_workers

    def setup(self, stage=None):
        # Load train and validation data
        with open(self.train_json_file, 'r') as f:
            train_data = json.load(f)
        
        # Load test data
        with open(self.test_json_file, 'r') as f:
            self.test_data = json.load(f)
        
        torch.manual_seed(self.seed)
        
        val_size = int(0.2 * len(train_data))
        train_size = len(train_data) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            ClarityDataset(train_data, self.train_data_path), [train_size, val_size]
        )
        self.test_dataset = ClarityDataset(self.test_data, self.test_data_path)

    def train_dataloader(self):
        rank_zero_info("Train DataLoader created.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        rank_zero_info("Validation DataLoader created.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=pad_collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        rank_zero_info("Test DataLoader created.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=pad_collate_fn, num_workers=self.num_workers)


def main():
    train_json_file = '/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/metadata/scenes.train.json'
    test_json_file = '/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/metadata/scenes.dev.json'  # Update with the actual test JSON file path
    train_data_path = '/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes'    # Update with the actual path to audio files
    test_data_path = '/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/dev/scenes'    # Update with the actual path to audio files
    batch_size = 1
    
    data_module = ClarityDataLoader(train_json_file, test_json_file, batch_size, train_data_path=train_data_path, test_data_path=test_data_path, num_workers=1)
    data_module.setup()

    print("Train Loader:")
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        rank_zero_info("Batch:", batch)
        break

    print("Validation Loader:")
    val_loader = data_module.val_dataloader()
    for batch in val_loader:
        rank_zero_info("Batch:", batch)
        break

    print("Test Loader:")
    test_loader = data_module.test_dataloader()
    for batch in test_loader:
        rank_zero_info("Batch:", batch)
        break

if __name__ == "__main__":
    main()
