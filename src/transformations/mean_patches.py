import numpy as np
import random
from albumentations.core.transforms_interface import ImageOnlyTransform

class MeanPatches(ImageOnlyTransform):
    """
    Adds rectangular patches filled with mean color values to the image.
    
    Args:
        num_patches (int, int): Min and max number of patches to add.
        patch_size (int, int): Min and max patch size in pixels.
        use_image_mean (bool): If True, use the mean of the entire image.
                              If False, use the mean of the patch area.
        p (float): Probability of applying the transform.
    """
    def __init__(
        self,
        num_patches=(3, 8),
        patch_size=(16, 64),
        use_image_mean=True,
        always_apply=False,
        p=0.5,
    ):
        super(MeanPatches, self).__init__(always_apply, p)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.use_image_mean = use_image_mean
        
    def apply(self, img, **params):
        img_copy = img.copy()
        height, width, channels = img.shape
        
        # Determine global mean if requested
        global_mean = np.mean(img, axis=(0, 1)) if self.use_image_mean else None
        
        # Add random patches
        num_to_add = random.randint(self.num_patches[0], self.num_patches[1])
        
        for _ in range(num_to_add):
            # Random patch size
            patch_h = random.randint(self.patch_size[0], self.patch_size[1])
            patch_w = random.randint(self.patch_size[0], self.patch_size[1])
            
            # Random position
            y = random.randint(0, height - patch_h)
            x = random.randint(0, width - patch_w)
            
            if self.use_image_mean:
                mean_value = global_mean
            else:
                # Calculate mean of the patch area
                mean_value = np.mean(img[y:y+patch_h, x:x+patch_w], axis=(0, 1))
            
            # Fill the patch with mean values
            img_copy[y:y+patch_h, x:x+patch_w] = mean_value
            
        return img_copy
    
    def get_transform_init_args_names(self):
        return ("num_patches", "patch_size", "use_image_mean")