import os
from os.path import *
import random
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from scipy.signal import resample_poly


class ClarityDataset(Dataset):

    def __init__(
        self,
        dataset: str = "/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes",
        target: str = "/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes",
        clarity_dir: str = '/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data',
        sample_rate: int = 8000,
        return_noise: bool = False,
        return_rvbt: bool = False,
    ) -> None:


        with open(clarity_dir+"/metadata/scenes.train.json", encoding="utf-8") as fp:
            scenes_listeners = json.load(fp)

            # Make list of all scene listener pairs that will be run
        self.scene_listener_pairs = make_scene_listener_list(
            scenes_listeners, cfg.evaluate.small_test
        )

        # Set the length of the dataset
        self.length = len(self.scene_listener_pairs)

        # Store the configuration
        self.cfg = cfg

    @staticmethod
    def make_scene_listener_list(scenes_listeners, small_test=False):
        """Make the list of scene-listener pairing to process"""
        scene_listener_pairs = [
            (scene, listener)
            for scene in scenes_listeners
            for listener in scenes_listeners[scene]
        ]

        # Can define a standard 'small_test' with just 1/15 of the data
        if small_test:
            scene_listener_pairs = scene_listener_pairs[::15]

        return scene_listener_pairs    

    def __getitem__(self, index_seed: tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))

        # Assuming scene_listener_pairs is accessible in this context
        scene, listener_id = self.scene_listener_pairs[index]

        # Read the audio files for the selected scene
        sample_rate, signal_ch1 = wavfile.read(
            pathlib.Path(self.cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )

        _, signal_ch2 = wavfile.read(
            pathlib.Path(self.cfg.path.scenes_folder) / f"{scene}_mix_CH2.wav"
        )

        _, signal_ch3 = wavfile.read(
            pathlib.Path(self.cfg.path.scenes_folder) / f"{scene}_mix_CH3.wav"
        )

        _, reference = wavfile.read(pathlib.Path(self.cfg.path.scenes_folder) / f"{scene}_reference.wav")

        # Convert to 32-bit floating point scaled between -1 and 1
        signal_ch1 = (signal_ch1 / 32768.0).astype(np.float32)
        signal_ch2 = (signal_ch2 / 32768.0).astype(np.float32)
        signal_ch3 = (signal_ch3 / 32768.0).astype(np.float32)

        reference = (reference / 32768.0).astype(np.float32)

        # Concatenate all channels along a new first axis [channels, time]
        signals = np.stack([signal_ch1, signal_ch2, signal_ch3], axis=0)

        reference = reference[np.newaxis, :, :]

        paras = {
            'index': index,
            'seed': seed,
            #'saveto': [str(p).removeprefix(str(self.wsj0_dir))[1:] for p in uttr_paths],
            'target': self.target,
            'sample_rate': self.sample_rate,
            'dataset': f'CHiME3_moving/{self.dataset0}',
            'snr': float(snr_real),
            'audio_time_len': self.audio_time_len,
            'num_spk': num_spk,
            'rir': {
                'RT60': rir_dict['RT60'],
                'pos_src': rir_dict['pos_src'],
                'pos_rcv': rir_dict['pos_rcv'],
            },
            'data': {
                'rir': rir,
                'noise': noise,
                'rvbt': rvbts,
            }
        }

        return (
            torch.as_tensor(signals, dtype=torch.float32),  # shape [chn, time]
            torch.as_tensor(reference, dtype=torch.float32),  # shape [spk, chn, time]
            paras,
        )

    def __len__(self):
        return self.length


class ClarityDataModule(LightningDataModule):

    def __init__(
        self,
        clarity_dir: str = '/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes',
        target: str = "/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes",
        datasets: Tuple[str, str, str, str] = ['train', 'val', 'test', 'test'],  # datasets for train/val/test/predict
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = False,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.clarity_dir = clarity_dir
        self.target = target
        self.datasets = datasets
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset:Clarity CEC3 task2")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'target: {self.target}')

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        # this is called once in single processor in case of multi node
        # if you need to use multi processing , you can use prepare
        self.current_stage = stage

    def construct_dataloader(self, dataset, seed, shuffle, batch_size, collate_fn):
        ds = ClarityDataset(
            dataset=dataset,
            target=self.target,
            clarity_dir=self.clarity_dir,
            sample_rate=8000
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  # what is the use of this sampler
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    """python -m data_loaders.chime3_moving"""
    dset = ClarityDataset(
        target='train/scenes',
        dataset='train/scenes'
    )
    for i in range(100):
        dset.__getitem__((i, i))

    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(ClarityDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='train')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 1#10  # 1 for debuging
    args_dict['datasets'] = ['train_clarity', 'val_clarity', 'test_clarity', 'test_clarity2']
    datamodule = ClarityDataModule(**args_dict)
    datamodule.setup()

    if args.dataset.startswith('train'):
        dataloader = datamodule.train_dataloader()
    elif args.dataset.startswith('val'):
        dataloader = datamodule.val_dataloader()
    elif args.dataset.startswith('test'):
        dataloader = datamodule.test_dataloader()
    else:
        assert args.dataset.startswith('predict'), args.dataset
        dataloader = datamodule.predict_dataloader()

    if type(dataloader) != dict:
        dataloaders = {args.dataset: dataloader}
    else:
        dataloaders = dataloader

    for ds, dataloader in dataloaders.items():

        for idx, (noisy, tar, paras) in enumerate(dataloader):
            print(f'{idx}/{len(dataloader)}', end=' ')
            if idx > 20:
                continue
            # write target to dir
            if args.gen_target and not args.dataset.startswith('predict'):
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(tar[0, :, 0, :].numpy())) <= 1
                for spk in range(tar.shape[1]):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, tar[0, spk, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write unprocessed's 0-th channel
            if args.gen_unprocessed:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(noisy[0, 0, :].numpy())) <= 1
                for spk in range(len(paras[0]['saveto'])):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, noisy[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write noise
            if paras[0]['data']['noise'] is not None:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noise").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                sp = tar_path / basename(paras[0]['saveto'][0])
                sf.write(sp, paras[0]['data']['noise'][0], samplerate=paras[0]['sample_rate'])

            print(noisy.shape, None if args.dataset.startswith('predict') else tar.shape, paras)
