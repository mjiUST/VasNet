###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.Bmp',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                # if root.endswith('/begin'): 
                path = os.path.join(root, fname)
                images.append(path)

    return images

def suppress_half_to_0(array_u8):
    """
    增强angiography对比度: [median, 255] -> [0, 255]
    """
    value_to_0 = np.median(array_u8)
    array2_u8 = ((array_u8 - value_to_0) * 255.0 / (255-value_to_0))
    array2_u8[array2_u8<0] = 0
    array2_u8 = array2_u8.astype(np.uint8)
    return array2_u8

def default_loader(path, preproc='None'):
    """
    preproc: 
        None: load as RGB
        suppress_half: suppress half of the pixel: map [median, 255] to [0, 255]
    """
    im_pil = Image.open(path).convert('RGB')
    if 'None' in preproc:
        im_out = im_pil
    elif 'suppress_half' in preproc:
        im_out = Image.fromarray(suppress_half_to_0(np.array(im_pil)))
    else:
        raise Warning("Do not know this img_loader_func: " + preproc)
    return im_out


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
