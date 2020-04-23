import os
import numpy as np
import torch
import torch.nn as nn

from dataset import dataset_single
from model import UID
from networks import PerceptualLoss16,PerceptualLoss
from options import TestOptions
from saver import Saver, save_imgs
from shutil import copyfile
from skimage.measure import compare_psnr as PSNR
from skimage.measure import compare_ssim as SSIM
from skimage.io import imread
from skimage.transform import resize
from data import CreateDataLoader
from util.util import return_center_crop_slices


def main():
    
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    orig_dir = opts.orig_dir
    blur_dir = opts.dataroot


    saver = Saver(opts)

    # data loader
    print('\n--- load dataset ---')
    dataset_domain = 'A' if opts.a2b else 'B'
    #     dataset = dataset_single(opts, 'A', opts.input_dim_a)
    # else:
    #     dataset = dataset_single(opts, 'B', opts.input_dim_b)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)
    loader = CreateDataLoader(opts)

    # model
    print('\n--- load model ---')
    model = UID(opts)
    model.setgpu(opts.gpu)  ## comment for cpu mode
    model.resume(opts.resume, train=False)
    model.eval()

    # test
    print('\n--- testing ---')
    for idx1, data in enumerate(loader):
        # img1, img_name_list = data[dataset_domain], data[dataset_domain+'_paths']
        # img1 = img1.cuda(opts.gpu).detach()
        images_b = data['B']
        images_a = images_b  # should in the same shape (This is only for the case `resize_or_crop="none"`)
        img_name_list = data['B_paths']  # B is the fluorescence image
        center_crop_shape = data['B_size_WH'][::-1]  # B is the fluorescence image
        if len(img_name_list) > 1:
            print("Warning, there are more than 1 sample in the test batch.")
        images_a = images_a.cuda(opts.gpu).detach()  ## comment for cpu mode
        images_b = images_b.cuda(opts.gpu).detach()  ## comment for cpu mode
        images_a = torch.cat([images_a]*2, dim=0)  # because half of the batch is used as real_A_random
        images_b = torch.cat([images_b]*2, dim=0)  # because half of the batch is used as real_B_random
        print('{}/{}'.format(idx1, len(loader)))
        with torch.no_grad():
            model.inference(images_a, images_b)
            # img = model.test_forward(img1, a2b=opts.a2b)
        img_name = img_name_list[0].split('/')[-1]
        saver.write_img(idx1, model, img_name=img_name, inference_mode=True, mask_path = '../input/testB_mask/'+img_name)  # True
        # saver.save_img(img=model.fake_I_encoded[[np.s_[:]]*2 + return_center_crop_slices(input_shapes=images_b.shape[-2:], 
        #                                                                                  output_shapes=center_crop_shape, 
        #                                                                                  input_scale=1.0, 
        #                                                                                  output_scale=opts.fineSize*1.0/opts.loadSize)], 
        #                img_name=img_name,
        #                subfolder_name="fake_A") #'gen_%05d.png' % (idx1),
        
    return

if __name__ == '__main__':
  main()
