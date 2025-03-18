#!/usr/bin/env python
"""
Image Transforms Utility
------------------------
Provides configurable transforms to manipulate images for model robustness testing.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

class ImageTransform:
    """Base class for image transformations"""
    
    def __init__(self, strength=0.0, enabled=False):
        """
        Initialize the transform
        
        Args:
            strength (float): Transform strength from 0.0 to 1.0
            enabled (bool): Whether the transform is enabled
        """
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.enabled = enabled
        
    def set_strength(self, strength):
        """Set the transform strength"""
        self.strength = max(0.0, min(1.0, strength))
        
    def set_enabled(self, enabled):
        """Enable or disable the transform"""
        self.enabled = enabled
        
    def apply(self, image):
        """
        Apply the transform to an image
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            
        Returns:
            numpy.ndarray: Transformed image
        """
        if not self.enabled or self.strength <= 0:
            return image
        
        return self._apply_transform(image)
        
    def _apply_transform(self, image):
        """
        Apply the actual transform (to be implemented by subclasses)
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Transformed image
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_name(self):
        """Get the name of the transform"""
        return self.__class__.__name__.replace("Transform", "")


class GaussianNoiseTransform(ImageTransform):
    """Add gaussian noise to the image"""
    
    def _apply_transform(self, image):
        """Apply gaussian noise"""
        # Map strength 0-1 to noise level 0-50
        noise_level = int(self.strength * 50)
        
        if noise_level <= 0:
            return image
            
        # Create a copy to avoid modifying the original
        result = image.copy()
        
        # Generate gaussian noise
        height, width, channels = image.shape
        noise = np.random.normal(0, noise_level, (height, width, channels))
        
        # Add noise to the image
        result = result + noise
        
        # Clip values to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


class BlurTransform(ImageTransform):
    """Apply gaussian blur to the image"""
    
    def _apply_transform(self, image):
        """Apply gaussian blur"""
        # Map strength 0-1 to kernel size 1-25 (must be odd)
        kernel_size = int(self.strength * 24) + 1
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        if kernel_size <= 1:
            return image
            
        # Apply gaussian blur
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


class ContrastTransform(ImageTransform):
    """Adjust the contrast of the image"""
    
    def _apply_transform(self, image):
        """Apply contrast adjustment"""
        # Map strength 0-1 to factor 0.5-2.0
        factor = 0.5 + self.strength * 1.5
        
        if abs(factor - 1.0) < 0.05:
            return image
            
        # Convert to PIL Image
        pil_img = Image.fromarray(image)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        result = enhancer.enhance(factor)
        
        # Convert back to numpy array
        return np.array(result)


class BrightnessTransform(ImageTransform):
    """Adjust the brightness of the image"""
    
    def _apply_transform(self, image):
        """Apply brightness adjustment"""
        # Map strength 0-1 to factor 0.5-1.5
        factor = 0.5 + self.strength
        
        if abs(factor - 1.0) < 0.05:
            return image
            
        # Convert to PIL Image
        pil_img = Image.fromarray(image)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(pil_img)
        result = enhancer.enhance(factor)
        
        # Convert back to numpy array
        return np.array(result)


class PosterizeTransform(ImageTransform):
    """Reduce the number of bits per color channel"""
    
    def _apply_transform(self, image):
        """Apply posterize effect"""
        # Map strength 0-1 to bits 8-1 (inverted, more strength = fewer bits)
        bits = 8 - int(self.strength * 7)
        
        if bits >= 8:
            return image
            
        # Convert to PIL Image
        pil_img = Image.fromarray(image)
        
        # Apply posterize
        result = ImageOps.posterize(pil_img, bits)
        
        # Convert back to numpy array
        return np.array(result)


class SharpnessTransform(ImageTransform):
    """Adjust the sharpness of the image"""
    
    def _apply_transform(self, image):
        """Apply sharpness adjustment"""
        # Map strength 0-1 to factor 0-3.0
        factor = self.strength * 3.0
        
        if factor < 0.05:
            return image
            
        # Convert to PIL Image
        pil_img = Image.fromarray(image)
        
        # Adjust sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        result = enhancer.enhance(factor)
        
        # Convert back to numpy array
        return np.array(result)


class PixelizeTransform(ImageTransform):
    """Pixelize the image by downsampling and upsampling"""
    
    def _apply_transform(self, image):
        """Apply pixelization effect"""
        # Map strength 0-1 to scale factor 1.0-0.05 (inverted, more strength = smaller scale)
        scale = 1.0 - self.strength * 0.95
        
        if scale > 0.95:
            return image
            
        height, width = image.shape[:2]
        
        # Calculate new size for downsampling
        small_width = max(1, int(width * scale))
        small_height = max(1, int(height * scale))
        
        # Downsample
        small_img = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        
        # Upsample with nearest neighbor to maintain pixelated look
        return cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)


class TransformManager:
    """Manages and applies multiple image transformations"""
    
    def __init__(self):
        """Initialize the transform manager with default transforms"""
        self.transforms = {
            "GaussianNoise": GaussianNoiseTransform(),
            "Blur": BlurTransform(),
            "Contrast": ContrastTransform(),
            "Brightness": BrightnessTransform(),
            "Posterize": PosterizeTransform(),
            "Sharpness": SharpnessTransform(),
            "Pixelize": PixelizeTransform()
        }
        
    def get_transform(self, name):
        """Get a transform by name"""
        return self.transforms.get(name)
        
    def set_transform_strength(self, name, strength):
        """Set the strength of a transform"""
        if name in self.transforms:
            self.transforms[name].set_strength(strength)
            
    def set_transform_enabled(self, name, enabled):
        """Enable or disable a transform"""
        if name in self.transforms:
            self.transforms[name].set_enabled(enabled)
            
    def apply_transforms(self, image):
        """Apply all enabled transforms to an image"""
        result = image.copy()  # Create a copy to avoid modifying the original
        
        # Apply each enabled transform
        for transform in self.transforms.values():
            if transform.enabled:
                result = transform.apply(result)
                
        return result
        
    def get_transform_names(self):
        """Get a list of all transform names"""
        return list(self.transforms.keys())
        
    def reset_all(self):
        """Reset all transforms to default values"""
        for transform in self.transforms.values():
            transform.set_strength(0.0)
            transform.set_enabled(False)
            
    def has_active_transforms(self):
        """Check if any transforms are currently active
        
        Returns:
            bool: True if any transforms are enabled with non-zero strength
        """
        for transform in self.transforms.values():
            if transform.enabled and transform.strength > 0:
                return True
        return False
        
    def get_transform_params(self):
        """Get current transform parameters for caching
        
        Returns:
            tuple: Tuple of (transform_name, enabled, strength) for each transform
        """
        return tuple(
            (name, transform.enabled, transform.strength)
            for name, transform in self.transforms.items()
        ) 