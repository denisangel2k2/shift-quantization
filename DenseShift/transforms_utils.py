import torch

class RepeatChannels:
    """
    Transform that repeats a single-channel image to create a 3-channel image.
    Used for adapting grayscale images like MNIST to models expecting RGB input.
    """
    def __call__(self, x):
        return x.repeat(3, 1, 1)