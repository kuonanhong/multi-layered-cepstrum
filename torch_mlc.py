import numpy as np
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as F


def _peak_picking(x):
    mask = (x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])
    return F.pad(mask, (1, 1))


def _pitch_profile(x, *args):
    filterbank = torch.sparse_coo_tensor(*args)
    return filterbank @ x


def _pitch_fusion(ppt, ppf, harms, sp_ratio):
    harms_range = harms[-1] + 1
    stacked_ppt = ppt.unfold(0, harms_range, 1).flip(2)
    stacked_ppf = ppf.unfold(0, harms_range, 1)

    harms_mask = stacked_ppf[:, :, harms].all(2)
    subharms_mask = stacked_ppt[:, :, harms].all(2)
    fusion_mask = harms_mask & subharms_mask

    sparsity_ppt = stacked_ppt.float().mean(2).lt(sp_ratio)
    sparsity_ppf = stacked_ppf.float().mean(2).lt(sp_ratio)

    return fusion_mask & (sparsity_ppf | sparsity_ppt)


def _medfilt(maks, kernel_size):
    batch_pad = (kernel_size // 2, kernel_size // 2)
    maks = F.pad(maks, batch_pad)
    # maks = F.pad(maks.t().view(1, 88, -1).float(), batch_pad, "reflect").view(88, -1).t().byte()
    maks = maks.unfold(1, kernel_size, 1)
    med_maks, idx = maks.median(2)
    return med_maks


class MLC_CFP(nn.Module):
    def __init__(self, gamma, sr=44100, hop_size=None, window_size=7938, med_num=25, sparse_ratio=0.7):
        super().__init__()
        self.gamma = gamma
        self.sr = sr
        if hop_size:
            self.hop_size = hop_size
        else:
            self.hop_size = sr // 100
        self.win_size = window_size
        self.med_num = med_num
        self.hpi = int(27.5 * window_size / sr) + 1
        self.lpi = int(0.00024 / (1 / sr)) + 1  # 0.24 ms
        self.sp_ratio = sparse_ratio

        self.window = nn.Parameter(torch.Tensor(get_window('blackmanharris', window_size)), requires_grad=False)
        self.harms = [0, 12, 19, 24]
        midi_num = np.arange(-3, 134)
        fd = 440 * np.power(2, (midi_num - 69.5) / 12)

        x = np.arange(window_size)
        freq_f = x * sr / window_size
        freq_t = sr / x[1:]

        idxs = np.digitize(freq_f, fd)
        in_piano_range = np.where((idxs > 24) & (idxs < 137))
        self.filter_f_idx = (idxs[in_piano_range] - 25, x[in_piano_range])
        self.filter_f_value = nn.Parameter(torch.ones(len(in_piano_range[0])), requires_grad=False)

        idxs = np.digitize(freq_t, fd)
        in_piano_range = np.where((idxs > 0) & (idxs < 113))
        self.filter_t_idx = (idxs[in_piano_range] - 1, x[in_piano_range] + 1)
        self.filter_t_value = nn.Parameter(torch.ones(len(in_piano_range[0])), requires_grad=False)
        self.filter_size = torch.Size([112, window_size])

    @torch.no_grad()
    def forward(self, x=None, spec=None):
        # channels last style spectrum
        if spec is None:
            spec = torch.stft(x, self.win_size, self.hop_size, window=self.window, onesided=False).pow_(2).sum(2)
        spec = spec.t()
        ceps = None
        for num, g in enumerate(self.gamma):
            if num % 2:
                spec = torch.rfft(ceps.pow_(g), 1, onesided=False)[..., 0]
                spec = F.relu(spec, True)
                spec[..., :self.hpi] = spec[..., -self.hpi:] = 0
            else:
                ceps = torch.rfft(spec.pow_(g), 1, onesided=False)[..., 0] / self.win_size
                ceps = F.relu(ceps, True)
                ceps[..., :self.lpi] = ceps[..., -self.lpi:] = 0

        ceps, spec = _peak_picking(ceps).float(), _peak_picking(spec).float()

        # back to channel first style
        ppt = _pitch_profile(ceps.t(), self.filter_t_idx, self.filter_t_value, self.filter_size)
        ppf = _pitch_profile(spec.t(), self.filter_f_idx, self.filter_f_value, self.filter_size)

        final = _pitch_fusion(ppt > 0, ppf > 0, self.harms, self.sp_ratio)
        final = _medfilt(final, self.med_num)
        return final, torch.arange(final.size(0)).float() * self.hop_size / self.sr
