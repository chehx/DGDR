import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import PIL

from utils.train import *
from utils.validate import *
from dataset.data_manager import get_dataset, get_normalize, get_post_aug, get_dataset_euqal
from modeling.model_manager import get_model, get_classifier
from utils.args import *
from utils.misc import *

import torch.nn.functional as F
import torch
import os, shutil
import copy

import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.stats import gaussian_kde

class Namespace_(argparse.Namespace):
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

class gaussian_kde_(gaussian_kde):
    """ A gaussian kde that is friendly to small samples. """
    def _compute_covariance(self) -> None:
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(
                np.cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
            w, v = np.linalg.eigh(self._data_covariance)
            # Set near-zero eigenvalues to a small number, avoiding singular covariance
            # matrices when the sample do not span the whole feature space
            w[np.where(abs(w) < 1e-9)[0]] = 0.01
            self._data_inv_cov = np.linalg.inv(v @ np.diag(w) @ v.T)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        

def compute_div(p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                eps_div: float) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    div = 0
    for i in range(len(probs)):
        if p[i] < eps_div or q[i] < eps_div:
            div += abs(p[i] - q[i]) / probs[i]
    div /= len(probs) * 2
    return div


def compute_cor(y_p: np.ndarray, z_p: np.ndarray, y_q: np.ndarray, z_q: np.ndarray,
                p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                points: np.ndarray, eps_cor: float, strict: bool = False) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    y_p_unique, y_q_unique = map(np.unique, (y_p, y_q))
    if not np.all(y_p_unique == y_q_unique):
        raise ValueError
    classes = sorted(y_p_unique)
    n_classes = len(classes)
    sample_sizes = np.zeros(n_classes, dtype=int)
    cors = np.zeros(n_classes, dtype=float)
    
    for i in range(n_classes):
        y = classes[i]
        indices_p = np.where(y_p == y)[0]
        indices_q = np.where(y_q == y)[0]
        if indices_p.shape != indices_q.shape:
            raise ValueError(f'Number of datapoints mismatch (y={y}): '
                             f'{indices_p.shape} != {indices_q.shape}')
        try:
            kde_p = gaussian_kde_(z_p[indices_p].T)
            kde_q = gaussian_kde_(z_q[indices_q].T)
            p_given_y = kde_p(points)
            q_given_y = kde_q(points)
        except (np.linalg.LinAlgError, ValueError) as exception:
            if strict:
                raise exception
            print(f'WARNING: skipping y={y} because scipy.stats.gaussian_kde '
                  f'failed. This usually happens when there is too few datapoints.')
            print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
                  f'skipped')
            continue
        sample_sizes[i] = len(indices_p)
        
        for j in range(len(probs)):
            if p[j] > eps_cor and q[j] > eps_cor:
                integrand = abs(p_given_y[j] * np.sqrt(q[j] / p[j])
                              - q_given_y[j] * np.sqrt(p[j] / q[j]))
                cor_j = integrand / probs[j]
            else:
                integrand = cor_j = 0
            cors[i] += cor_j
        cors[i] /= len(probs) * 2
        # print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
            #   f'value={cors[i]:.4f}')
    cor = np.sum(sample_sizes * cors) / np.sum(sample_sizes)
    return cor

def evaluate_cor_div(data, mode):
    y_p, z_p, y_q, z_q = data['y_p'], data['z_p'], data['y_q'], data['z_q']
    # print(f'features loaded: (p) {z_p.shape}, (q) {z_q.shape}')
    # print(f'labels   loaded: (p) {y_p.shape}, (q) {y_q.shape}')
    if len(z_p) != len(y_p) or len(z_q) != len(y_q):
        raise RuntimeError

    z_all = np.append(z_p, z_q, 0)
    scaler = StandardScaler().fit(z_all)
    z_all, z_p, z_q = map(scaler.transform, (z_all, z_p, z_q))

    # print('computing KDE for importance sampling')
    sampling_pdf = gaussian_kde(z_all.T)
    points = sampling_pdf.resample(10000, seed=cfg.SEED)
    probs = sampling_pdf(points)

    # print('computing KDE for p and q')
    p = gaussian_kde(z_p.T)(points)
    q = gaussian_kde(z_q.T)(points)

    if mode == 'div':
        # print('computing diversity shift')
        div = compute_div(p, q, probs, 1e-12)
        if np.isnan(div) or np.isinf(div):
            raise RuntimeError
        # print(f'div: {div}')
        return div

    if mode == 'cor':
        # print('computing correlation shift')
        cor = compute_cor(y_p, z_p, y_q, z_q, p, q, probs, points, 5e-4,
                            strict=False)
        if np.isnan(cor) or np.isinf(cor):
            raise RuntimeError
        # print(f'cor: {cor}')
        return cor
    
if __name__ == '__main__':

    args = get_args()
    cfg = setup_cfg(args)
    if not args.random:
        setup_seed(cfg.SEED)

    Domains = ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]

    for domain in Domains:
        test_domains = [domain]
        sub_domains = copy.deepcopy(Domains)
        sub_domains.remove(domain)

        val_loader, test_loader = get_dataset_euqal(cfg, 128, sub_domains, test_domains)
        model = get_model(args, cfg)
        classifier_div = torch.nn.Sequential(
                torch.nn.Linear(model.out_features(), model.out_features() // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(model.out_features() // 2, model.out_features() // 4))
                
        classifier_cor = torch.nn.Sequential(
                torch.nn.Linear(model.out_features(), model.out_features() // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(model.out_features() // 2, model.out_features() // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(model.out_features() // 4, 5))
        
        log_path = f'./result/Div_Cor/Test_on_{test_domains[0]}_ours'
            
        checkpoint = torch.load(os.path.join(log_path, f'model_and_metrics_{100}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier_div.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
        classifier_cor.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
        
        # Load Models via log_path.    
        model = model.cuda()
        classifier_div = classifier_div.cuda()
        classifier_cor = classifier_cor.cuda()
        model.eval()
        
        save_dict = {}
        save_dict_cor = {}
        for k, dataloader in enumerate([val_loader, test_loader]):
            envs_code = ('p', 'q')[k]
            # print(f'extracting features from envs {envs_code}')
            y_minibatches = []
            z_minibatches = []
            z_cor_minibatches = []
            for i, (x, y) in enumerate(dataloader):
                x = x.cuda()
                with torch.no_grad():
                    z = classifier_div(model(x))
                    z_cor = classifier_cor(model(x))
                y_minibatches.append(y)
                z_minibatches.append(z.cpu())
                z_cor_minibatches.append(z_cor.cpu())
            y_cat = torch.cat(y_minibatches)
            z_cat = torch.cat(z_minibatches)
            z_cor_cat = torch.cat(z_cor_minibatches)
            save_dict[f'y_{envs_code}'] = y_cat.numpy()
            save_dict[f'z_{envs_code}'] = z_cat.numpy()
            save_dict_cor[f'y_{envs_code}'] = y_cat.numpy()
            save_dict_cor[f'z_{envs_code}'] = z_cor_cat.numpy()
        
        div = evaluate_cor_div(save_dict, 'div')
        cor = evaluate_cor_div(save_dict_cor, 'cor')

        print(f"Test Domain: {domain}, Div: {div}, Cor: {cor}")