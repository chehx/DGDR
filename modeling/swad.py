'''
This codes contain modules DRGen needs.
'''

import copy
from collections import deque
import numpy as np
import logging

import torch
from torch.nn import Module
from copy import deepcopy

class SWADBase:
    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        raise NotImplementedError()

    def get_final_model(self):
        raise NotImplementedError()

class LossValley(SWADBase):
    """IIDMax has a potential problem that bias to validation dataset.
    LossValley choose SWAD range by detecting loss valley.
    """

    def __init__(self, evaluator, n_converge, n_tolerance, tolerance_ratio, **kwargs):
        """
        Args:
            evaluator
            n_converge: converge detector window size.
            n_tolerance: loss min smoothing window size
            tolerance_ratio: decision ratio for dead loss valley
        """
        self.evaluator = evaluator
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio

        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)

        self.final_model = None

        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    @property
    def is_converged(self):
        return self.converge_step is not None

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        if self.dead_valley:
            return

        frozen = copy.deepcopy(segment_swa)
        frozen.end_loss = val_loss
        self.converge_Q.append(frozen)
        self.smooth_Q.append(frozen)

        if not self.is_converged:
            if len(self.converge_Q) < self.n_converge:
                return

            min_idx = np.argmin([model.end_loss for model in self.converge_Q])
            untilmin_segment_swa = self.converge_Q[min_idx]  # until-min segment swa.
            if min_idx == 0:
                self.converge_step = self.converge_Q[0].end_step
                self.final_model = AveragedModel(untilmin_segment_swa)

                th_base = np.mean([model.end_loss for model in self.converge_Q])
                self.threshold = th_base * (1.0 + self.tolerance_ratio)

                if self.n_tolerance < self.n_converge:
                    for i in range(self.n_converge - self.n_tolerance):
                        model = self.converge_Q[1 + i]
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                elif self.n_tolerance > self.n_converge:
                    converge_idx = self.n_tolerance - self.n_converge
                    Q = list(self.smooth_Q)[: converge_idx + 1]
                    start_idx = 0
                    for i in reversed(range(len(Q))):
                        model = Q[i]
                        if model.end_loss > self.threshold:
                            start_idx = i + 1
                            break
                    for model in Q[start_idx + 1 :]:
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                print(
                    f"Model converged at step {self.converge_step}, "
                    f"Start step = {self.final_model.start_step}; "
                    f"Threshold = {self.threshold:.6f}, "
                )
            return

        if self.smooth_Q[0].end_step < self.converge_step:
            return

        # converged -> loss valley
        min_vloss = self.get_smooth_loss(0)
        if min_vloss > self.threshold:
            #self.dead_valley = True
            self.dead_valley = False
            print(f"Valley is dead at step {self.final_model.end_step}")
            return

        model = self.smooth_Q[0]
        self.final_model.update_parameters(
            model, start_step=model.start_step, end_step=model.end_step
        )

    def get_final_model(self):
        if not self.is_converged:
            logging.warning(
                "Requested final model, but model is not yet converged; return last model instead"
            )
            return self.converge_Q[-1]

        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                smooth_loss = self.get_smooth_loss(0)
                if smooth_loss > self.threshold:
                    break
                segment_swa = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment_swa, step=segment_swa.end_step)

        return self.final_model

class AveragedModel(Module):
    def __init__(self, model, device=None, avg_fn=None, rm_optimizer=False):
        super(AveragedModel, self).__init__()
        self.start_step = -1
        self.end_step = -1
        if isinstance(model, AveragedModel):
            # prevent nested averagedmodel
            model = model.module
        self.module = deepcopy(model)
        if rm_optimizer:
            for k, v in vars(self.module).items():
                if isinstance(v, torch.optim.Optimizer):
                    setattr(self.module, k, None)

        self.module = self.module.cuda()

        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))

        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
                    num_averaged + 1
                )

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        #  return self.predict(*args, **kwargs)
        return self.module(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)

    @property
    def network(self):
        return self.module.network

    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        """Update averaged model parameters

        Args:
            model: current model to update params
            step: current step. step is saved for log the averaged range
            start_step: set start_step only for first update
            end_step: set end_step
        """
        if isinstance(model, AveragedModel):
            model = model.module
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device))
                )
        self.n_averaged += 1

        if step is not None:
            if start_step is None:
                start_step = step
            if end_step is None:
                end_step = step

        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step = start_step

        if end_step is not None:
            self.end_step = end_step

    def clone(self):
        clone = copy.deepcopy(self.module)
        clone.optimizer = clone.new_optimizer(clone.network.parameters())
        return clone