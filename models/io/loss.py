from torch import Tensor
import torch
from torch import nn
from torch_stoi import NegSTOILoss  # Import the STOI loss function

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio as sa_sdr
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate as permutate
from torchmetrics.functional.audio import perceptual_evaluation_speech_quality as pesq_torch
from typing import *
from models.io.cirm import build_complex_ideal_ratio_mask, decompress_cIRM
from models.io.stft import STFT
import torchaudio.transforms as T


def neg_sa_sdr(preds: Tensor, target: Tensor, scale_invariant: bool = False) -> Tensor:
    batch_size = target.shape[0]
    sa_sdr_val = sa_sdr(preds=preds, target=target, scale_invariant=scale_invariant)
    return -torch.mean(sa_sdr_val.view(batch_size, -1), dim=1)


def neg_si_sdr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_si_sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)


def neg_snr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_snr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    snr_val = snr(preds=preds, target=target)
    return -torch.mean(snr_val.view(batch_size, -1), dim=1)


def _mse(preds: Tensor, target: Tensor) -> Tensor:
    """calculate mse loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    diff = preds - target
    diff = diff.view(batch_size, -1)
    mse_val = torch.mean(diff**2, dim=1)
    return mse_val


def cirm_mse(preds: Tensor, target: Tensor) -> Tensor:
    """calculate mse loss for a batch of cirms

    Returns:
        loss: shape [batch], real
    """
    return _mse(preds=preds, target=target)


def cc_mse(preds: Tensor, target: Tensor) -> Tensor:
    """calculate mse loss for a batch of STFT coefficients

    Returns:
        loss: shape [batch], real
    """
    return _mse(preds=preds, target=target)


def neg_nb_pesq_torch(preds: Tensor, target: Tensor, fs: int = 8000) -> Tensor:
    """calculate negative NB PESQ loss for a batch using torchmetrics

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    pesq_vals = pesq_torch(preds=preds, target=target, fs=fs, mode='nb')
    return -torch.mean(pesq_vals.view(batch_size, -1), dim=1)

# # Define the resamplers for NB (8kHz) and WB (16kHz)
resample_to_nb = T.Resample(orig_freq=48000, new_freq=8000)
resample_to_wb = T.Resample(orig_freq=48000, new_freq=16000)

def neg_nb_pesq(preds: Tensor, target: Tensor) -> Tensor:
    """calculate negative NB PESQ loss for a batch using torchmetrics after resampling"""
    preds_resampled = resample_to_nb(preds)
    target_resampled = resample_to_nb(target)
    batch_size = target.shape[0]
    pesq_vals = pesq_torch(preds=preds_resampled, target=target_resampled, fs=8000, mode='nb')
    return -torch.mean(pesq_vals.view(batch_size, -1), dim=1)

def neg_wb_pesq(preds: Tensor, target: Tensor) -> Tensor:
    """calculate negative WB PESQ loss for a batch using torchmetrics after resampling"""
    preds_resampled = resample_to_wb(preds)
    target_resampled = resample_to_wb(target)
    batch_size = target.shape[0]
    pesq_vals = pesq_torch(preds=preds_resampled, target=target_resampled, fs=16000, mode='wb')
    return -torch.mean(pesq_vals.view(batch_size, -1), dim=1)


def neg_stoi_loss(preds: Tensor, target: Tensor, sample_rate: int = 48000, use_vad: bool = True, extended: bool = True, do_resample:bool = False) -> Tensor:
    """Calculate negative STOI loss for a batch using torch_stoi's NegSTOILoss

    Args:
        preds (Tensor): Predicted audio batch
        target (Tensor): Ground truth audio batch
        sample_rate (int): Sample rate of the audio
        vad (bool): Voice activity detection flag
        extend (bool): Extend the loss function to non-standard cases

    Returns:
        Tensor: Computed negative STOI loss
    """
    loss_func = NegSTOILoss(sample_rate=48000, use_vad =True, extended=True, do_resample=False).to('cuda')
    return loss_func(preds, target)

class Loss(nn.Module):
    is_scale_invariant_loss: bool
    name: str
    mask: str

    def __init__(self, loss_func: Callable, pit: bool, loss_func_kwargs: Dict[str, Any] = dict()):
        super().__init__()

        self.loss_func = loss_func
        self.pit = pit
        self.loss_func_kwargs = loss_func_kwargs
        self.is_scale_invariant_loss = {
            neg_sa_sdr: True if 'scale_invariant' in loss_func_kwargs and loss_func_kwargs['scale_invariant'] == True else False,
            neg_si_sdr: True,
            neg_snr: False,
            cirm_mse: False,
            cc_mse: False,
            neg_nb_pesq: False,
            neg_wb_pesq: False,
            neg_stoi_loss: False,  # Updated with the correct reference
            # Add any other new loss functions here
        }[loss_func]
        self.name = loss_func.__name__
        self.mask = 'cirm' if self.loss_func == cirm_mse else None

    def forward(self, yr_hat: Tensor, yr: Tensor, reorder: bool = None, reduce_batch: bool = True, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        if self.mask is not None:
            out, Xr, stft = kwargs['out'], kwargs['Xr'], kwargs['stft']
            Yr, _ = stft.stft(yr)
            preds, target = out, self.to_mask(Yr=Yr, Xr=Xr)
            preds, target = torch.view_as_real(preds), torch.view_as_real(target)
        elif self.loss_func == cc_mse:
            out, XrMM, stft = kwargs['out'], kwargs['XrMM'], kwargs['stft']
            Yr, _ = stft.stft(yr)
            Yr = Yr / XrMM
            preds, target = torch.view_as_real(out), torch.view_as_real(Yr)
        else:
            preds, target = yr_hat, yr
            
        #print(f"Calling {self.loss_func.__name__} with kwargs: {self.loss_func_kwargs}")
        perms = None
        if self.pit:
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode="permutation-wise", **self.loss_func_kwargs)
        else:
            losses = self.loss_func(preds=preds, target=target, **self.loss_func_kwargs)

        if reorder and perms is not None:
            yr_hat = permutate(yr_hat, perm=perms)

        return losses.mean() if reduce_batch else losses, perms, yr_hat

    def to_CC(self, out: Tensor, Xr: Tensor, stft: STFT, XrMM: Tensor) -> Tensor:
        if self.loss_func == cirm_mse:
            cIRM = decompress_cIRM(mask=out)
            Yr = Xr * cIRM
            return Yr, {'out': out, 'Xr': Xr, 'stft': stft, 'XrMM': XrMM}
        else:
            return out, {'out': out, 'Xr': Xr, 'stft': stft, 'XrMM': XrMM}

    def to_mask(self, Yr: Tensor, Xr: Tensor):
        if self.mask == 'cirm':
            return build_complex_ideal_ratio_mask(noisy=Xr, clean=Yr)
        else:
            raise Exception(f'not implemented for mask type {self.mask}')

    def extra_repr(self) -> str:
        kwargs = ""
        for k, v in self.loss_func_kwargs.items():
            kwargs += f'{k}={v},'

        return f"loss_func={self.loss_func.__name__}({kwargs}), pit={self.pit}, mask={self.mask}"
