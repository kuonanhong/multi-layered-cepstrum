import numpy as np
from scipy import sparse
from scipy.signal import stft, argrelmax, medfilt
from scipy.fftpack import fft, ifft

# defining pitch range (piano as standard)
midi_num = np.arange(-3, 134)  # Array length = 88 + 24 *2 = 136
# maxout range
fd = 440. * np.power(2, (midi_num - 69.5) / 12)

# harmonics index
har = np.array([0, 12, 19, 24])


def _pitch_profile_period(acr, sampling_freq):
    acr_copy = acr.tocsr()[1:]
    freq_scale = sampling_freq / np.arange(1, acr.shape[0])
    idxs = np.digitize(freq_scale, fd)

    filter_bank = sparse.coo_matrix((np.broadcast_to(1, idxs.shape), (idxs, np.arange(len(freq_scale)))))
    filter_bank = filter_bank.tocsr()[1:113]

    upcpt = filter_bank @ acr_copy
    return upcpt.toarray()


def _pitch_profile(spec, freq_scale):
    idxs = np.digitize(freq_scale, fd)

    filter_bank = sparse.coo_matrix((np.broadcast_to(1, idxs.shape), (idxs, np.arange(len(freq_scale)))))
    filter_bank = filter_bank.tocsr()[25:137]

    upcp = filter_bank @ spec
    return upcp.toarray()


def _pitch_fusion(upcp, upcpa, ratio):
    nonzero_upcp = upcp > 0
    nonzero_upcpa = upcpa > 0
    upcp_stacked = np.stack((nonzero_upcp[i:i + 88] for i in range(25)), 0)
    upcpa_stacked = np.stack((nonzero_upcpa[i:i + 88] for i in range(24, -1, -1)), 0)

    upcp_f0 = upcp_stacked[har].all(0)
    upcpa_f0 = upcpa_stacked[har].all(0)

    sparsity_upcp = upcp_stacked.mean(0)
    sparsity_upcpa = upcpa_stacked.mean(0)

    upcp_final = upcp_f0 & upcpa_f0 & ((sparsity_upcp < ratio) | (sparsity_upcpa < ratio))

    return upcp_final


def MLC_CFP(x, gamma, sr=44100, hop_size=None, window_size=7938, med_num=25, sparse_ratio=0.7):
    if not hop_size:
        hop_size = sr // 100

    f, t, raw_stft = stft(x, sr, window='blackmanharris', nperseg=window_size, noverlap=window_size - hop_size,
                          return_onesided=False)
    # power-spectrogram
    spec = np.abs(raw_stft) ** 2

    # get filter index
    hpi = np.where(f > 27.5)[0][0]  # 27.5 hz
    lpi = int(0.00024 / (1 / sr)) + 1  # 0.24 ms

    # perform MLC
    for num, g in enumerate(gamma):
        if num % 2:
            spec = fft(ceps ** g, axis=0).real
            spec = np.maximum(spec, 0)  # relu
            spec[:hpi] = spec[-hpi:] = 0  # highpass
        else:
            ceps = ifft(spec ** g, axis=0).real
            ceps = np.maximum(ceps, 0)  # relu
            ceps[:lpi] = ceps[-lpi:] = 0  # lowpass

    # peak picking
    id1, id2 = argrelmax(spec), argrelmax(ceps)
    sparse_spec = sparse.coo_matrix((spec[id1], id1), shape=spec.shape)
    sparse_ceps = sparse.coo_matrix((ceps[id2], id2), shape=ceps.shape)

    # pitch profile
    upcpt = _pitch_profile_period(sparse_ceps, sr)
    upcp = _pitch_profile(sparse_spec, f)

    ##### fusion Section #####
    upcp_final = _pitch_fusion(upcp, upcpt, sparse_ratio)

    # post-processing: median filter
    upcp_final_do_med = np.apply_along_axis(medfilt, 1, upcp_final, kernel_size=med_num)

    return upcp_final_do_med, t
