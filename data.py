import os
import math
import random
import numbers
import collections
from PIL import ImageOps
from dataset import DatasetFromFolder

from torchvision.transforms import ToTensor, Scale, CenterCrop, Normalize

def _iterate_transforms_(transforms_, x):
    if isinstance(transforms_, collections.Iterable):
        for i, transform in enumerate(transforms_):
            x[i] = _iterate_transforms_(transform, x[i])
    else:
            x = transforms_(x)
    return x

# we can pass nested arrays inside Compose
# the first level will be applied to all inputs
# and nested levels are passed to nested transforms_
class Compose(object):
    def __init__(self, transforms_):
        self.transforms_ = transforms_

    def __call__(self, x):
        for transform in self.transforms_:
            x = _iterate_transforms_(transform, x) 
        return x

class Nothing(object):
    def __call__(self, img):
        return img

class RandomCropGenerator(object):
    def __call__(self, img):
        self.x1 = random.uniform(0, 1)
        self.y1 = random.uniform(0, 1)
        return img
        
class RandomCrop(object):
    def __init__(self, size, padding=0, gen=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self._gen = gen

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        if self._gen is not None:
            x1 = math.floor(self._gen.x1 * (w - tw))
            y1 = math.floor(self._gen.y1 * (h - th))
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

        return img.crop((x1, y1, x1 + tw, y1 + th))


def get_image_dir(dest):
    if not os.path.exists(dest):
        print("current {} not exists!".format(dest))
        exit(-1)
    return dest


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_data_transform(crop_size, upscale_factor):
    gen = RandomCropGenerator()
    nothing = Nothing()
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize_size = crop_size // upscale_factor
    return Compose([
        [gen],
        [RandomCrop(crop_size, gen=gen), RandomCrop(crop_size, gen=gen)],
        [Scale(resize_size, 3), nothing], # 3 means Image.BICUBIC; 2 means Image.BILINEAR
        [ToTensor(), ToTensor()],
        [normalize, normalize],
    ])


def test_data_transform(crop_size, upscale_factor):
    gen = RandomCropGenerator()
    nothing = Nothing()
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize_size = crop_size // upscale_factor
    return Compose([
        [gen],
        [CenterCrop(crop_size), CenterCrop(crop_size)],
        [Scale(resize_size, 3), nothing],
        [ToTensor(), ToTensor()],
        [normalize, normalize],
    ])


def get_training_set(dest, crop_size, upscale_factor, jpeg, noise, blur):
    root_dir = get_image_dir(dest)
    train_dir = os.path.join(root_dir, "train")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             data_transform=train_data_transform(crop_size, upscale_factor))


def get_test_set(dest, crop_size, upscale_factor, jpeg, noise=None, blur=None):
    root_dir = get_image_dir(dest)
    test_dir = os.path.join(root_dir, "test")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             data_transform=test_data_transform(crop_size, upscale_factor))
