import torch
import torch.nn as nn
import math

class ADMMQuantizer:
    def __init__(self, model, quantize_layers, rho=1e-3, target_bits=2):
        self.model = model
        self.rho = rho
        self.G = {}  # Auxiliary variables (quantized versions)
        self.lmbda = {}  # Dual variables
        self.quantize_layers = quantize_layers
        self.target_bits = target_bits # Store target bits (e.g., 2 for ternary)

        # Initialize G and lambda
        for name, param in model.named_parameters():
            if name in quantize_layers:
                # G should be initialized as the quantized version of the initial weights
                # lambda should be initialized to zeros
                self.G[name] = self.iterative_quantize(param.data.clone(), N=1)
                self.lmbda[name] = torch.zeros_like(param.data)

    def step(self):
        """
        Performs the G-update and lambda-update steps of ADMM.
        This should be called periodically after W-update (gradient descent on augmented loss).
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.quantize_layers:
                    w = param.data
                    lmbda = self.lmbda[name]

                    # g-update with iterative quantization
                    g_new = self.iterative_quantize(w + lmbda, N=1)
                    self.G[name] = g_new

                    # lambda-update: use the updated G
                    self.lmbda[name] = lmbda + (w - g_new)
    
    def iterative_quantize(self, V, N=1, max_iter=5):
        Q = torch.zeros_like(V)
        alpha = V.abs().mean()
        if alpha == 0:
            return torch.zeros_like(V)

        for _ in range(max_iter):
            # in the paper this is the other way around, but if initializing Q with zeros at the beginning, then an additional first step that does nothing is computed, thus we do the Q first, then the alpha
            # 1. Update Q: nearest integer in {-2^N, ..., 0, ..., +2^N}
            q_vals = torch.arange(-2**N, 2**N + 1, device=V.device)
            V_scaled = V / alpha
            Q = torch.stack([(V_scaled - q).abs() for q in q_vals], dim=0)
            Q_idx = torch.argmin(Q, dim=0)
            Q = q_vals[Q_idx]

            # 2. Update alpha (least squares)
            numerator = (V * Q).sum()
            denominator = (Q * Q).sum() + 1e-8  # avoid div by 0
            alpha = numerator / denominator

        return alpha * Q

    def apply_loss_penalty(self, base_loss):
        """
        Adds the ADMM penalty term to the network's base loss.
        This penalty drives the full-precision weights (param) towards the quantized Z.
        """
        admm_penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.quantize_layers:
                G = self.G[name]
                lmbda = self.lmbda[name]
                # L2 norm squared: ||param - G + u||^2
                # param.data is implicitly used when param is involved in a loss computation.
                admm_penalty += (self.rho / 2) * torch.norm(param - G + lmbda) ** 2
        return base_loss + admm_penalty


    def calculate_effective_bit_width_admm(self, full_precision_bits=32):
        total_params = 0
        quantized_params = 0

        for name, param in self.model.named_parameters():
            numel = param.numel()
            total_params += numel
            if name in self.quantize_layers:
                quantized_params += numel
        
        if total_params == 0:
            return full_precision_bits
        
        quantized_ratio = quantized_params / total_params
        effective_bits = quantized_ratio * self.target_bits + (1 - quantized_ratio) * full_precision_bits
        return effective_bits
    