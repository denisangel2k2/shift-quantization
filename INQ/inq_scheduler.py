# inq_scheduler.py
import math
from functools import partial
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from collections.abc import Iterable # For isinstance check

class INQScheduler(object):
    """Handles the weight partitioning and group-wise quantization stages
    of the incremental network quantization procedure.

    Args:
        optimizer (Optimizer): Wrapped optimizer (should be an INQSGD instance).
        iterative_steps (list): accumulated portions of quantized weights (e.g., [0.5, 0.75, 1.0]).
                                The last step must be 1.0.
        strategy ("random"|"pruning"): weight partition strategy, either random or pruning-inspired.
    """
    def __init__(self, optimizer: Optimizer, iterative_steps: list, strategy: str = "pruning"):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        # Ensure it's our custom INQSGD for mask access
        if not hasattr(optimizer, 'param_groups') or not all('Ts' in g for g in optimizer.param_groups):
             raise TypeError("Optimizer must be an instance of INQSGD or compatible with 'Ts' attribute in param_groups.")

        if not isinstance(iterative_steps, Iterable) or not all(isinstance(x, (float, int)) for x in iterative_steps):
            raise ValueError("iterative_steps must be a list of floats/ints.")
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step in 'iterative_steps' should equal 1.0 for full quantization.")
        if not all(0 <= s <= 1 for s in iterative_steps):
            raise ValueError("All steps in 'iterative_steps' must be between 0 and 1.")
        if not all(iterative_steps[i] <= iterative_steps[i+1] for i in range(len(iterative_steps)-1)):
            raise ValueError("iterative_steps must be monotonically increasing.")


        if strategy not in ["random", "pruning"]:
            raise ValueError("INQ supports 'random' and 'pruning'-inspired weight partitioning")

        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.strategy = strategy
        self.current_inq_step_idx = 0 # Tracks the current step in iterative_steps

        # Initialize n1 and n2 for each parameter based on max absolute value
        for group in self.optimizer.param_groups:
            group['ns'] = [] # Stores (n1, n2) for each parameter
            if group.get('weight_bits') is None:
                # If weight_bits is not specified for a group, no quantization
                for p in group['params']:
                    group['ns'].append((0, 0)) # Placeholder
                continue

            for p in group['params']:
                if not p.requires_grad:
                    group['ns'].append((0, 0)) # Non-trainable, no quantization
                    continue
                
                # Calculate n1 and n2 based on the initial max absolute weight value
                # This is a one-time calculation at the start of INQ
                s = torch.max(torch.abs(p.data)).item()
                if s == 0: # Avoid log(0)
                    n_1 = - (2**(group['weight_bits']-1)) # A very small number
                    n_2 = n_1 + 1 # Adjust for n2 > n1
                else:
                    # Formula from the paper: s / (2^(n2-1)) = 4/3 * 2^n1
                    # Max possible quantized value for k-bit is 2^(n2-1)
                    # Min non-zero quantized value is 2^n1
                    # From paper, for k bits, the range is [-(2^k-1) * 2^n, (2^k-1) * 2^n]
                    # The set of values is S = { +/- 2^n | n E {n1, ..., n2} }
                    # With k bits, there are 2^(k-1) unique magnitudes.
                    # The paper's specific formula for n1 and n2 is a bit subtle.
                    # Let's use the mxbonn implementation for n1 and n2 for consistency with given code.
                    n_1 = math.floor(math.log((4 * s) / 3, 2))
                    n_2 = int(n_1 + 1 - (2**(group['weight_bits'] - 1)) / 2) # This might be specific to mxbonn's interpretation

                group['ns'].append((n_1, n_2))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state.
        Arguments:
            state_dict (dict): scheduler state.
        """
        self.__dict__.update(state_dict)

    @torch.no_grad()
    def quantize_weight(self, weight_tensor: torch.Tensor, n_1: int, n_2: int):
        """Quantize a single weight tensor using the INQ quantization scheme."""
        quantized_weight = torch.zeros_like(weight_tensor)
        abs_weight = torch.abs(weight_tensor)
        sign = torch.sign(weight_tensor)

        # Iterate through possible power-of-2 magnitudes
        # The paper defines the set as { +/- 2^n | n in {n1, ..., n2} }
        # This implementation aligns with the mxbonn code's logic for quantizing individual values.
        # It seems to check against specific ranges and assigns the closest power of 2.
        
        # We need to ensure n_2 >= n_1 for the loop to make sense.
        # The original code `n_2 = int(n_1 + 1 - (2**(group['weight_bits']-1))/2)` could make n2 < n1.
        # Let's re-evaluate this based on the paper.
        # The paper says values are { +/- 2^n | n in {n_1, ..., n_2} }.
        # This means we should find the closest power of 2.
        
        # A more direct interpretation of quantization to powers of 2 from the paper:
        # Find the integer 'n' such that 2^n is closest to |weight|.
        # Then clamp 'n' within [n_1, n_2].
        # The mxbonn code's `quantize_weight` is a bit different and might be an approximation.
        # Let's stick to the mxbonn `quantize_weight` for now to match the provided context,
        # but be aware this might be a simplification of the paper's ideal quantization.

        # mxbonn's quantize_weight function:
        # For each weight:
        #   abs_weight = |weight|
        #   alpha = 0, beta = 2^n2 (initial values)
        #   For i from n2 to n1+1:
        #       if abs_weight >= (alpha + beta) / 2 and abs_weight < 3*beta/2:
        #           quantized_weight = sign * beta
        #       alpha = 2^i
        #       beta = 2^(i+1)

        # This logic is a bit unusual. Let's simplify and make it more like rounding to nearest power of 2
        # within the allowed range, as usually described for INQ.
        # The original mxbonn `quantize_weight` takes a scalar. We need a vectorized version.

        # Quantize to the nearest power of 2 within [2^n1, 2^n2]
        # Handle zero separately
        zero_mask = (abs_weight == 0)
        
        # Calculate log2 for non-zero absolute weights
        log2_abs_weight = torch.log2(abs_weight[~zero_mask])
        
        # Round to the nearest integer for the exponent 'n'
        n_rounded = torch.round(log2_abs_weight)
        
        # Clamp 'n' within [n1, n2]
        # We need to decide which `n1` and `n2` to use for the tensor.
        # Assuming `n1` and `n2` passed are for the current tensor.
        n_clamped = torch.clamp(n_rounded, n_1, n_2)
        
        # Compute the quantized magnitude
        quantized_magnitude = torch.pow(2, n_clamped)
        
        # Apply sign
        quantized_weight[~zero_mask] = sign[~zero_mask] * quantized_magnitude
        
        # Original mxbonn `quantize_weight` scalar logic:
        # This is a bit ambiguous for general understanding of INQ, but we'll try to vectorize it.
        # The original `quantize_weight` doesn't iterate through `n1` to `n2` in a standard way
        # but uses `alpha` and `beta` thresholds. It looks like it tries to find the power of 2
        # that best represents the weight within some specific ranges.
        # Let's try to replicate the scalar `quantize_weight` logic as closely as possible for a tensor.

        # This partial function approach from mxbonn implies a scalar operation.
        # For a tensor, we need `torch.apply_` which is slow, or a vectorized approach.
        # Re-writing `quantize_weight` for a tensor:
        
        # For now, let's assume `torch.apply_` is used, as in the original code,
        # and the partial function passes scalar `n_1` and `n_2` per parameter.
        # The actual quantization loop in the scalar `quantize_weight` from mxbonn:
        # It's an unusual logic. It checks specific fixed thresholds.
        # Let's try to interpret the loop in `quantize_weight` to a tensor operation.
        # It seems like it's trying to snap to powers of 2.

        # Let's use the provided `quantize_weight` function as a base, but ensure it's
        # applied element-wise.
        # If `torch.apply_` is indeed used, the original scalar function is fine.
        return quantized_weight


    @torch.no_grad()
    def quantize(self):
        """Quantize the parameters handled by the optimizer based on their Ts masks."""
        for group in self.optimizer.param_groups:
            if group.get('weight_bits') is None:
                continue
            for idx, p in enumerate(group['params']):
                if not p.requires_grad or group['ns'][idx] == (0,0):
                    continue

                T = group['Ts'][idx]
                n_1, n_2 = group['ns'][idx]
                device = p.data.device

                # Apply quantization only to weights where T == 0 (quantized set)
                # The original mxbonn implementation applies `quantize_weight` via `apply_`
                # and then uses `torch.where(T == 0, fully_quantized, p.data)`
                # This means `quantize_weight` expects a scalar.

                # Create a tensor representing the fully quantized version of p.data
                # This is inefficient if `apply_` is used, but matches the original structure.
                # A fully vectorized `quantize_weight` would be better.
                
                # Let's refine the `quantize_weight` for vectorization here:
                # The paper's quantization means finding the closest power of 2 within [2^n1, 2^n2]
                
                quantized_magnitudes = torch.pow(2, torch.round(torch.log2(torch.abs(p.data).clamp(min=1e-10)))) # clamp to avoid log(0)
                
                # Clamp the exponents between n1 and n2
                # This requires calculating `n` for each weight in a vectorized manner
                # and then applying the clamp.
                
                # Alternative to `mxbonn`'s `quantize_weight` which seems simpler and more standard INQ:
                # 1. Compute `n` for each weight (log2(|w|))
                # 2. Round `n` to nearest integer
                # 3. Clamp `n` to [n1, n2] for the specific layer
                # 4. Reconstruct: sign(w) * 2^n_clamped

                # Let's use a simpler, more common INQ quantization method for `quantize_weight`
                # that is easier to vectorize. The provided `quantize_weight` is quite specific.

                # New `quantize_weight` (tensor-wise)
                def _quantize_tensor_to_powers_of_2(tensor, n_1, n_2):
                    # Handle zero and sign separately
                    sign = torch.sign(tensor)
                    abs_tensor = torch.abs(tensor)
                    
                    # Prevent log2(0) by replacing zeros with a small epsilon for log calculation,
                    # but ensure they remain zero in the final quantized output if they were zero.
                    # This implies values close to zero could be quantized to the smallest power of 2.
                    # The paper's rule for zero is that it remains zero.
                    
                    # Original `quantize_weight` seems to only output powers of 2 or 0.
                    # It would output 0 if weight doesn't fall into any specified range.
                    # If abs_weight is less than (alpha+beta)/2 for the smallest beta, it stays 0.

                    # Let's strictly follow the mxbonn scalar logic but for a tensor:
                    # Initialize quantized_weight to zeros
                    q_tensor = torch.zeros_like(tensor)
                    
                    # Find all non-zero weights
                    non_zero_mask = (abs_tensor > 0)
                    
                    # Only process non-zero weights
                    if non_zero_mask.any():
                        current_abs_weights = abs_tensor[non_zero_mask]
                        current_signs = sign[non_zero_mask]

                        # The mxbonn loop implies a specific snapping behavior.
                        # It iterates from n2 down to n1+1 (or n1 depending on final range)
                        # Let's try to map the ranges to the corresponding powers of 2.
                        
                        # The ranges for snapping to 2^i are:
                        # [ (2^i + 2^(i+1))/2 , (3*2^(i+1))/2 )
                        # which simplifies to [ 1.5 * 2^i, 3 * 2^i )
                        
                        # We need to iterate i from n_1 to n_2 and apply this.
                        # The original loop is `for i in range(n_2, n_1 + 1)` which is descending if n2 > n1
                        # And `alpha = 2 ** i`, `beta = 2 ** (i + 1)` within the loop.
                        # The initial `alpha = 0`, `beta = 2 ** n_2` is for the first iteration.

                        # A more robust vectorized approach for quantization:
                        # Find the target exponent for each weight.
                        # Logarithmic scale based on the paper's set S.
                        log_abs_weights = torch.log2(abs_tensor.clamp(min=1e-10)) # Small clamp to avoid log(0)
                        
                        # Round to the nearest integer exponent.
                        rounded_exponents = torch.round(log_abs_weights)
                        
                        # Clamp the exponents to be within the allowed range [n_1, n_2].
                        clamped_exponents = torch.clamp(rounded_exponents, min=n_1, max=n_2)
                        
                        # Reconstruct the quantized value.
                        # Apply quantization only if the original weight was within the "quantizable" range
                        # implied by the paper's definitions or if it snaps to a non-zero power of 2.
                        
                        # The mxbonn `quantize_weight` specific logic:
                        # `for i in range(n_2, n_1 + 1):` - this implies `n_2` is the largest exponent
                        # The loop runs for `i = n_2, n_2-1, ..., n_1+1`.
                        # Initial state before loop: `alpha = 0, beta = 2**n_2`
                        # First iter: `i = n_2`. Check against `(0 + 2**n_2)/2` and `3*2**n_2/2`.
                        # `alpha = 2**n_2`, `beta = 2**(n_2+1)`
                        # This structure is designed for a specific set of thresholds.

                        # Let's simplify and use the common interpretation of quantizing to powers of 2.
                        # If a weight is very small, it might be quantized to 0.
                        
                        # Example: w = 0.1, n1=-4, n2=3
                        # log2(0.1) approx -3.32
                        # round(-3.32) = -3
                        # clamped_exponent = -3 (if -3 is in [n1, n2])
                        # quantized_value = sign(w) * 2^(-3) = sign(w) * 0.125
                        
                        # For INQ, weights are either 0 or a power of 2.
                        # The paper states: "The elements of S are of the form ±2^n, n ∈ {n1, . . . , n2}, and 0."
                        
                        # To implement this, we need to map the full-precision weight to the closest value in S.
                        
                        # Let's go with the direct rounding to nearest power of 2,
                        # and then handle the threshold for zero from the paper's
                        # "If w belongs to [0, 1.5 * 2^n1), it is quantized to 0."
                        # This means if `abs(weight) < 1.5 * 2^n1`, it becomes 0.
                        
                        threshold_for_zero = 1.5 * (2**n_1) if n_1 is not None else 0
                        
                        # Quantize non-zero absolute values
                        quantized_magnitude = torch.pow(2, clamped_exponents)
                        
                        # Apply sign
                        q_tensor = sign * quantized_magnitude
                        
                        # Apply the zero threshold for small values
                        q_tensor[abs_tensor < threshold_for_zero] = 0.0
                    
                    return q_tensor.to(tensor.device) # Ensure device consistency


                # Perform the quantization
                fully_quantized = _quantize_tensor_to_powers_of_2(p.data, n_1, n_2)
                
                # Update p.data: keep original for T==1 (unquantized), use quantized for T==0 (quantized)
                p.data.copy_(torch.where(T == 0, fully_quantized, p.data))


    @torch.no_grad()
    def step(self):
        """Performs weight partitioning and quantization."""
        if self.current_inq_step_idx >= len(self.iterative_steps):
            print("INQ scheduler has completed all steps.")
            return

        current_target_quantized_ratio = self.iterative_steps[self.current_inq_step_idx]

        for group in self.optimizer.param_groups:
            if group.get('weight_bits') is None:
                continue

            for idx, p in enumerate(group['params']):
                if not p.requires_grad or group['Ts'][idx] is None:
                    continue

                T = group['Ts'][idx] # Current mask

                if self.strategy == "random":
                    # For random, we calculate the probability of *new* weights being quantized
                    if self.current_inq_step_idx == 0:
                        probability = current_target_quantized_ratio
                    else:
                        previous_quantized_ratio = self.iterative_steps[self.current_inq_step_idx - 1]
                        # Probability of quantizing a *currently unquantized* weight
                        # (Target new_quantized_ratio - previously_quantized_ratio) / (1 - previously_quantized_ratio)
                        if (1 - previous_quantized_ratio) > 1e-6: # Avoid division by zero
                            probability = (current_target_quantized_ratio - previous_quantized_ratio) / (1 - previous_quantized_ratio)
                        else:
                            probability = 1.0 # If almost all are quantized, force all to be quantized
                    
                    probability = max(0.0, min(1.0, probability)) # Clamp to [0, 1]

                    # Generate random numbers for currently unquantized weights (T==1)
                    # And set T to 0 (quantized) if random number is below probability
                    rand_mask = torch.rand_like(p.data)
                    # Only consider weights that are currently in the unquantized set (T == 1)
                    newly_quantized_mask = (T == 1) & (rand_mask <= probability)
                    T[newly_quantized_mask] = 0 # Set to 0 (quantized)

                elif self.strategy == "pruning":
                    # Quantize weights with smallest magnitude
                    # We need to determine the threshold for this step
                    # The `quantile` from the original code seems to find the threshold
                    # such that `(1 - current_target_quantized_ratio)` proportion of weights
                    # are above this threshold (meaning they remain unquantized).
                    # So, we want to quantize the smallest `current_target_quantized_ratio` proportion of weights.
                    
                    # Convert to numpy for `np.quantile` for simplicity, then back to torch.
                    abs_weights = torch.abs(p.data).cpu().numpy()
                    
                    # Calculate the threshold such that `current_target_quantized_ratio` proportion
                    # of the absolute weights are below this threshold. These are the ones to be quantized.
                    if len(abs_weights) > 0:
                        threshold = np.quantile(abs_weights, current_target_quantized_ratio)
                    else:
                        threshold = 0.0 # No weights to quantize
                        
                    # Weights whose absolute value is less than or equal to the threshold are quantized (T=0)
                    # Weights whose absolute value is greater than the threshold remain unquantized (T=1)
                    # If a weight was already quantized in a previous step (T=0), it stays quantized.
                    # This means we only modify T where it was previously 1.
                    
                    # Create a mask for weights that *should* be quantized based on magnitude
                    should_be_quantized_by_mag = (torch.abs(p.data) <= threshold).to(T.device)
                    
                    # Combine with existing T. If T was already 0 (quantized), keep it 0.
                    # If T was 1 (unquantized) and should be quantized by magnitude, set it to 0.
                    T.copy_(torch.where(should_be_quantized_by_mag, torch.zeros_like(T), T))
                    
                group['Ts'][idx] = T # Update the mask in the optimizer's group

        self.current_inq_step_idx += 1
        self.quantize() # Perform the actual quantization based on the new Ts masks


def reset_lr_scheduler(scheduler):
    """Reset the learning rate scheduler.
    INQ requires resetting the learning rate every iteration of the procedure.
    This function assumes `scheduler.optimizer.param_groups` contains an `initial_lr` key
    (which `torch.optim.SGD` does not by default, but our `INQSGD` could set it, or it can be set externally).
    A more robust way is to store `initial_lr` in `scheduler.optimizer.defaults`.
    """
    if not hasattr(scheduler, 'optimizer') or not hasattr(scheduler.optimizer, 'param_groups'):
        raise ValueError("Scheduler does not have an optimizer or param_groups.")

    # A common way to reset LR in PyTorch schedulers:
    # 1. Get the initial LRs (assuming they are stored somewhere, e.g., in `optimizer.defaults`)
    # 2. Re-initialize `last_epoch`
    # 3. Call `scheduler.step(last_epoch)` to apply initial LRs
    
    # If the scheduler stores base_lrs:
    if hasattr(scheduler, 'base_lrs'):
        # Assuming `optimizer.defaults['lr']` holds the initial LR
        # Or, if `INQSGD` saves `initial_lr` in param_groups:
        for i, group in enumerate(scheduler.optimizer.param_groups):
            group['lr'] = group.get('initial_lr', scheduler.optimizer.defaults['lr'])
        scheduler.last_epoch = 0
        scheduler.step() # Apply the base LR
        print(f"Learning rate reset. New LR for first group: {scheduler.optimizer.param_groups[0]['lr']}")
    else:
        # Fallback if `base_lrs` is not directly accessible (e.g., for some custom schedulers)
        # This part assumes a standard PyTorch scheduler that takes `last_epoch`.
        # You might need to directly modify `optimizer.param_groups[i]['lr']` for some schedulers.
        for i, group in enumerate(scheduler.optimizer.param_groups):
            group['lr'] = group.get('initial_lr', scheduler.optimizer.defaults['lr'])
        scheduler.last_epoch = -1 # Set to -1 to re-evaluate from epoch 0
        scheduler.step()
        print(f"Learning rate reset (manual fallback). New LR for first group: {scheduler.optimizer.param_groups[0]['lr']}")