import torch.utils.data as data

import io
import random
import numpy as np
from os import listdir
from os.path import join

import scipy.misc
import PIL
from PIL import Image, ImageFilter

PIL.Image.MAX_IMAGES_PIXELS = None

def GaussianNoise(img, noise):
    h, w = img.size
    img_arr = scipy.misc.fromimage(img).astype(np.float32)
    img_arr += scipy.random.normal(scale=noise, size=(w, h, 1))
    img     = scipy.misc.toimage(img_arr)
    return img


def GaussianBlur(img, blur):
    imgfilter = ImageFilter.GaussianBlur(radius=random.randint(0, 2*blur))
    img = img.filter(imgfilter)
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "tif"])


def load_img(filepath, jpeg):
    img = Image.open(filepath).convert('RGB')
    if jpeg > 0:
        buffer = io.BytesIO()
        img.save(buffer, format='jpeg', quality=random.randrange(75-jpeg, 76)) # default quality=75
        img = Image.open(buffer)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, data_transform=None, jpeg=0, noise=0.0, blur=0):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])

        self.jpeg = jpeg
        self.noise = noise
        self.blur  = blur
        self.transform = data_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index], self.jpeg)
        target = input.copy()
        
        if self.noise > 0:
            input = GaussianNoise(input, self.noise)
        if self.blur > 0:
            input = GaussianFilter(input, self.blur)

        if self.transform:
            input, target= self.transform([input, target])
        return input, target

    def __len__(self):
        return len(self.image_filenames)
