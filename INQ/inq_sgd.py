# inq_sgd.py
import torch
from torch.optim.optimizer import Optimizer, required

class INQSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum)
    for incremental network quantization.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        weight_bits (int, optional): number of bits used in the quantization of the weights
                                     (for the INQ scheduler to use, not directly used by SGD)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, weight_bits=None):
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, weight_bits=weight_bits)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(INQSGD, self).__init__(params, defaults)

        # Initialize T (mask for trainable weights) for each parameter group
        for group in self.param_groups:
            group['Ts'] = []
            for p in group['params']:
                if p.requires_grad:
                    # Initially, all weights are unquantized (trainable)
                    group['Ts'].append(torch.ones_like(p.data))
                else:
                    # Non-trainable parameters (e.g., BatchNorm stats)
                    group['Ts'].append(None) # Use None to indicate not applicable

    def __setstate__(self, state):
        super(INQSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            Ts = group['Ts'] # Get the quantization mask for this group

            for idx, p in enumerate(group['params']):
                if p.grad is None or Ts[idx] is None: # Skip if no grad or not a trainable weight
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Apply the quantization mask to the gradient
                # Only update weights in the unquantized set (where T == 1)
                d_p.mul_(Ts[idx])
                p.add_(d_p, alpha=-group['lr'])

        return loss