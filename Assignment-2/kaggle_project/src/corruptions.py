import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

# ==========================================
# 1. Custom Corruption Transforms
# ==========================================

class AddGaussianNoise(object):
    """
    Applies pixel-level Gaussian noise to a tensor image.
    Required sigmas: 0.05, 0.1, 0.2.
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, tensor):
        # Generate noise with the same shape as the tensor
        noise = torch.randn(tensor.size()) * self.sigma
        # Add noise and clip values to stay within valid image range [0, 1]
        noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma={self.sigma})'


class MotionBlur(object):
    """
    Applies a synthetic motion blur using a directional convolution kernel.
    """
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size
        
        # Create a horizontal motion blur kernel
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2, :] = 1.0 / kernel_size
        self.kernel = kernel

    def __call__(self, tensor):
        # tensor shape is expected to be (C, H, W)
        c, h, w = tensor.shape
        
        # Prepare kernel for depthwise convolution (applied independently to RGB channels)
        kernel = self.kernel.repeat(c, 1, 1, 1).to(tensor.device)
        tensor = tensor.unsqueeze(0) # Reshape to (1, C, H, W) for conv2d
        
        # Pad the image to maintain original dimensions after convolution
        pad = self.kernel_size // 2
        tensor = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')
        
        # Apply the blur
        blurred = F.conv2d(tensor, kernel, groups=c)
        return blurred.squeeze(0) # Return to (C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + f'(kernel_size={self.kernel_size})'


class BrightnessShift(object):
    """
    Adjusts the brightness of the image.
    factor > 1 makes the image brighter, factor < 1 makes it darker.
    """
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, tensor):
        # PyTorch's functional adjust_brightness handles both PIL Images and Tensors
        return TF.adjust_brightness(tensor, self.factor)

    def __repr__(self):
        return self.__class__.__name__ + f'(factor={self.factor})'


# ==========================================
# 2. Pipeline Integration Helper
# ==========================================

def get_corrupted_val_transform(corruption_type, severity_param=None):
    """
    Builds a validation transform pipeline with the specified corruption injected.
    The corruption is applied after resizing/cropping and converting to Tensor, 
    but before ImageNet normalization.
    """
    # Standard ImageNet Validation base transforms
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    # Inject the chosen corruption
    if corruption_type == "gaussian_noise":
        base_transforms.append(AddGaussianNoise(sigma=severity_param))
    elif corruption_type == "motion_blur":
        # severity_param can dictate kernel size, e.g., 15
        base_transforms.append(MotionBlur(kernel_size=severity_param if severity_param else 15))
    elif corruption_type == "brightness_shift":
        # severity_param dictates the brightness factor, e.g., 1.5 (brighter) or 0.5 (darker)
        base_transforms.append(BrightnessShift(factor=severity_param if severity_param else 1.5))
    elif corruption_type == "clean":
        pass # No corruption
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    # Apply standard ImageNet normalization at the very end
    base_transforms.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    return transforms.Compose(base_transforms)