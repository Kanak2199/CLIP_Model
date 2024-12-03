""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self._param_groups = self.base_optimizer.param_groups
        self.state = defaultdict(dict)

        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # Perform a single optimization step with the base optimizer
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        slow_state_dict = {
            'state': state_dict.get('slow_state', {}),
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self._param_groups = self.base_optimizer.param_groups
        for name, default in self.defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    # Expose attributes for compatibility with PyTorch GradScaler
    @property
    def _optimizer_step_pre_hooks(self):
        return getattr(self.base_optimizer, '_optimizer_step_pre_hooks', {})

    @property
    def _optimizer_step_post_hooks(self):
        return getattr(self.base_optimizer, '_optimizer_step_post_hooks', {})

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    @property
    def param_groups(self):
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._param_groups = value
        self.base_optimizer.param_groups = value
