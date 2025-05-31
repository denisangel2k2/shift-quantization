import torch
import torch.nn as nn
import math

class ADMMQuantizer:
    def __init__(self, model, quantize_layers, rho=1e-3, target_bits=2):
        self.model = model
        self.rho = rho
        self.Z = {}  # Auxiliary variables (quantized versions)
        self.U = {}  # Dual variables
        self.quantize_layers = quantize_layers
        self.target_bits = target_bits # Store target bits (e.g., 2 for ternary)

        # Initialize Z and U
        for name, param in model.named_parameters():
            if name in quantize_layers:
                # Z should be initialized as the quantized version of the initial weights
                # U should be initialized to zeros
                self.Z[name] = self.iterative_quantize(param.data.clone(), N=1)
                self.U[name] = torch.zeros_like(param.data)

    def step(self):
        """
        Performs the Z-update and U-update steps of ADMM.
        This should be called periodically after W-update (gradient descent on augmented loss).
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.quantize_layers:
                    w = param.data
                    u = self.U[name]

                    # Z-update with iterative quantization
                    z_new = self.iterative_quantize(w + u, N=1)
                    self.Z[name] = z_new

                    # U-update: use the updated Z
                    self.U[name] = u + (w - z_new)
    
    def iterative_quantize(self, V, N=1, max_iter=5):
        Q = torch.zeros_like(V)
        alpha = V.abs().mean()
        if alpha == 0:
            return torch.zeros_like(V)

        for _ in range(max_iter):
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
                z = self.Z[name]
                u = self.U[name]
                # L2 norm squared: ||param - z + u||^2
                # param.data is implicitly used when param is involved in a loss computation.
                admm_penalty += (self.rho / 2) * torch.norm(param - z + u) ** 2
        return base_loss + admm_penalty

    def get_effective_bits(self, full_precision_bits=32):
        """
        Calculates the effective bit-width of the model during ADMM training.
        During training, W is full-precision. So, it's 32 bits for the ADMM paper's training phase.
        The effective bit-width for *inference* is only achieved after final quantization.
        """
        # During ADMM training, the model weights `param` are still full-precision.
        # The penalty only encourages them to be close to quantized `Z`.
        # So, for the *training phase*, the "cost" is effectively full precision.
        # The cost is only reduced *after* the final quantization.
        
        # This function will be useful for the final evaluation step to show the *target* bits.
        # It's more about the *intended* or *final* bit-width than a mixed-precision average during training.
        
        # Count total trainable parameters considered for quantization
        total_quantizable_elements = 0
        for name, param in self.model.named_parameters():
            if name in self.quantize_layers and param.requires_grad:
                total_quantizable_elements += param.numel()

        if total_quantizable_elements == 0:
            return full_precision_bits # If no layers are set for quantization

        # If all specified layers are to be quantized to target_bits,
        # and non-specified layers are full_precision_bits, then the average
        # is calculated here if we consider *all* parameters.
        
        # For simplicity, if we consider only the *quantized layers* cost, it's just self.target_bits.
        # If we need the average across *all* model parameters, we need to know total non-quantizable elements.
        
        # For the plot, let's consider the *target* bit-width for the quantized layers.
        # The paper typically aims for a specific bit-width for the *final* quantized model.
        
        # For the context of "how cost evolves", if we are tracking before and after
        # final quantization, it's 32 bits during ADMM training, and `target_bits` after.
        return self.target_bits # For the final quantized model state
    
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
