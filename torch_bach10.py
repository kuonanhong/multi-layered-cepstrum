import argparse
import os
import torch
import numpy as np
from librosa import load
from time import time
from matplotlib import pyplot as plt
from pypianoroll import plot_pianoroll

from utils import multipitch_evaluation, read_bach10_F0s
from torch_mlc import MLC_CFP

parser = argparse.ArgumentParser(description='Multi-layered Cepstrum on bach10 dataset with torch.')
parser.add_argument('infile', type=str, help='input wav file')
parser.add_argument('-g', nargs='+', type=float, default=[0.24, 0.6], help='gamma values')
parser.add_argument('--window_size', type=int, default=7938, help='window size')
parser.add_argument('--f0_file', type=str, help='the ground truth used for evaluation')
parser.add_argument('--med_num', type=int, default=25, help='median filter size')
parser.add_argument('--sparse_ratio', type=float, default=0.7, help='the ratio of sparsity in harmonics selection')
parser.add_argument('--cuda', action='store_true', help='utilize gpu power')

if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    y, sr = load(args.infile, sr=None)
    y = torch.Tensor(y).to(device)
    basename = os.path.basename(args.infile)

    model = MLC_CFP(args.g, sr, window_size=args.window_size, med_num=args.med_num, sparse_ratio=args.sparse_ratio).to(
        device)

    a = time()
    result, t = model(y)
    cost = time() - a
    print("Time cost: %.4f seconds" % cost)

    result = result.cpu().numpy().astype(np.float)

    if args.f0_file:
        truth = read_bach10_F0s(args.f0_file)
        if result.shape[1] > truth.shape[1]:
            result = result[:, :truth.shape[1]]
        elif result.shape[1] < truth.shape[1]:
            result = np.pad(result, ((0, 0), (0, truth.shape[1] - result.shape[1])), 'constant', constant_values=0)
        p, r, f1 = multipitch_evaluation(result, truth, raw_value=False)
        print("Precision: %.4f, Recall: %.4f, F-score:, %.4f" % (p, r, f1))

        result_proll = np.pad(result.T, ((0, 0), (21, 19)), 'constant', constant_values=0)
        truth_proll = np.pad(truth.T, ((0, 0), (21, 19)), 'constant', constant_values=0)
        f, axes = plt.subplots(2, 1)
        plot_pianoroll(axes[0], truth_proll)
        axes[0].set_title('ground truth')
        plot_pianoroll(axes[1], result_proll)
        axes[1].set_title('predict')
        f.suptitle(basename)
    else:
        pianoroll = np.pad(result.T, ((0, 0), (21, 19)), 'constant', constant_values=0)
        ax = plt.gca()
        plot_pianoroll(ax, pianoroll)
        plt.title('predict')
    plt.show()
