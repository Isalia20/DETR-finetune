import random
from torchvision.transforms.functional import gaussian_blur

class BlurImage:
    def __call__(self, img, blur_bool):
        if blur_bool:
            kernel_size_int = random.randrange(3, 11, 2)
            img = gaussian_blur(img, kernel_size=(kernel_size_int, kernel_size_int), sigma=(0.1, 2.0))
        return img
