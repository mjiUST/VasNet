import torch.utils.data as data
import PIL
import math
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from datasets.transforms import AlignedRandomAffine, AlignedRandomFlip, RandomAdjustColor


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def get_transform_list(opt, disable_data_aug):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize, pad_if_needed=True))
    elif opt.resize_or_crop == 'crop_and_scale':  # crop loadSize w/o black border before scale to fineSize
        """
        For different shape input images, we need to keep the ratio between the scene and the pixel size
        """
        transform_list.append(transforms.Lambda(
            lambda img: pad_to_get_valid_shape(img, min_size = opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.loadSize, pad_if_needed=True))
        transform_list.append(transforms.Resize(opt.fineSize, interpolation=PIL.Image.BICUBIC))
    elif opt.resize_or_crop == 'crop_and_rotate_and_scale':  # crop loadSize w/o black border before scale to fineSize
        """
        For different shape input images, we need to keep the ratio between the scene and the pixel size
        """
        size_larger_patch = opt.loadSize * 1.416
        transform_list.append(transforms.Lambda(
            lambda img: pad_to_get_valid_shape(img, min_size = size_larger_patch)))  # make sure the rotated image does not have black borders.
        transform_list.append(transforms.RandomCrop(size_larger_patch, pad_if_needed=True))  # square patch (larger)
        transform_list.append(transforms.RandomRotation(degrees =[-180, 180], resample=PIL.Image.BICUBIC))
        transform_list.append(transforms.CenterCrop(size = opt.loadSize))  # square patch with size = opt.loadSize
        transform_list.append(transforms.Resize(opt.fineSize, interpolation=PIL.Image.BICUBIC))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize, pad_if_needed=True))
    elif opt.resize_or_crop == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    elif 'aligned' in opt.resize_or_crop: # 'aligned_crop_and_rotate_and_scale'  # crop loadSize w/o black border before scale to fineSize
        """
        ***** opt.loadSize*2 --> opt.loadSize --> opt.fineSize --> tensor *****
        1. based on the mask, a larger path with size of opt.loadSize*2 will be cropped
        2. this aligned transformation will be applied on all the modalities. 
        3. after random_rotate_shear_flip, crop the center patch with shape of opt.loadSize
        4. scale to opt.fineSize
        
        aligned_affine_flip_and_scale
        """
        # size_larger_patch = opt.loadSize * 1.415
        # transform_list.append(transforms.Lambda(
        #     lambda img: pad_to_get_valid_shape(img, min_size = size_larger_patch)))  # make sure the rotated image does not have black borders.
        if not disable_data_aug:
            if 'affine' in opt.resize_or_crop:  # aligned (rotate, scale, shear), [-5,5], [-180,180]
                transform_list.append(AlignedRandomAffine(degrees=[-180,180], scale_ranges=(0.98,1.25), shears=(-10.,10.)))
            if 'crop' in opt.resize_or_crop:  # aligned random crop 
                raise Warning("To be implementated: aligned random crop.")  # TODO
            if 'flip' in opt.resize_or_crop:
                transform_list.append(AlignedRandomFlip())
        
        # crop to opt.loadSize and scale to opt.fineSize
        transform_list.append(transforms.CenterCrop(size = opt.loadSize))  # square patch with size = opt.loadSize
        # transform_list.append(transforms.CenterCrop(size = opt.loadSize*1.5))  # square patch with size = opt.loadSize
        # transform_list.append(transforms.RandomCrop(opt.loadSize, pad_if_needed=True))  # randomly crop patch of size opt.loadSize*2
        ## transform_list.append(transforms.RandomCrop(opt.loadSize, pad_if_needed=True))  # randomly crop patch of size opt.loadSize*2
        ## transform_list.append(transforms.CenterCrop(size = opt.loadSize))  # square patch with size = opt.loadSize
        transform_list.append(transforms.Resize(opt.fineSize, interpolation=Image.BICUBIC))

    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)
    return transform_list




def compose_transform_list(transform_list):
    """
    Do not use `transform_list+=`, otherwise, it will take the **reference** of the variable and change its original value.
    """
    all_transform_list = transform_list + [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))]
    return transforms.Compose(all_transform_list)

def get_transform(opt): 
    transform_list = get_transform_list(opt, opt.disable_data_aug)
    # raise Warning("Not defined method: get_transform")
    return compose_transform_list(transform_list)

class AlignedTransformer():
    def __init__(self, opt, disable_data_aug):
        self.aligned_transform_list = get_transform_list(opt, disable_data_aug)

    def get_aligned_transform(self, with_color_jitter = True):
        if with_color_jitter:  # apply different color jitter for each function call
            adjust_range = [0.8,1.2]  #  [0.8,1.2]  /  [0.5,2.]
            transform_list = self.aligned_transform_list + [RandomAdjustColor(brightness_range=adjust_range, contrast_range=adjust_range, gamma_range=adjust_range),]
        else:
            transform_list = self.aligned_transform_list
        return compose_transform_list(transform_list)
    

# just modify the width and height to be multiple of 4
def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def pad_to_get_valid_shape(img, min_size):
    """
    Pad in mirror mode. (Only repeat once) Do not has black border.
    """
    padding_hw = tuple([max(math.ceil((min_size - _size)/2), 0) for _size in img.size])  # pad only when the h/w size is smaller than the min_size
    ## If a single int is provided this is used to pad all borders. 
    ## If tuple of length 2 is provided this is the padding on top/bottom and left/right respectively. 
    ## If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
    img = TF.pad(img, padding=padding_hw, padding_mode='reflect')  # make sure the cropped image do not contain black border.
    return img


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True




