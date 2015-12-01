#!/usr/bin/env python2

from collections import namedtuple
import numpy as np

class State:
    def __init__(self):
        self.sq_updates = dict()
        self.sq_grads = dict()

def update(net, state, config):
    rho = config.rho
    epsilon = config.eps
    lr = config.lr
    clip = config.clip

    all_norm = 0.
    for param_name in net.active_param_names():
        param = net.params[param_name]
        grad = param.diff * net.param_lr_mults(param_name)
        all_norm += np.sum(np.square(grad))
    all_norm = np.sqrt(all_norm)

    for param_name in net.active_param_names():
        param = net.params[param_name]
        grad = param.diff * net.param_lr_mults(param_name)

        if all_norm > clip:
            grad = clip * grad / all_norm

        if param_name in state.sq_grads:
            state.sq_grads[param_name] = \
                (1 - rho) * np.square(grad) + rho * state.sq_grads[param_name]
            rms_update = np.sqrt(state.sq_updates[param_name] + epsilon)
            rms_grad = np.sqrt(state.sq_grads[param_name] + epsilon)
            update = -rms_update / rms_grad * grad

            state.sq_updates[param_name] = \
                (1 - rho) * np.square(update) + rho * state.sq_updates[param_name]
        else:
            state.sq_grads[param_name] = (1 - rho) * np.square(grad)
            update = np.sqrt(epsilon) / np.sqrt(epsilon +
                    state.sq_grads[param_name]) * grad
            state.sq_updates[param_name] = (1 - rho) * np.square(update)

        param.data[...] += lr * update
        param.diff[...] = 0
