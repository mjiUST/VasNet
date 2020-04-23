import re
import os
import cv2
import numpy as np 
import imageio
from scipy import signal
from scipy import ndimage
from skimage import exposure

root = '/home/mengqi/fileserver/results/fluorescence/20190301_retina_blood_vessel/20190313_retina_to_quantum20190311_AHE_fuse_5_residual-2k-only_test/results_test_1024_512_crop_and_scale/nc1_resnet_9blocks_256_128_crop_and_rotate_and_scale/test_240/selected/fake_A_subfolders/20190311_diff_21-filter-fuse-color_code/median_fuse_n_5-[80_150]'
scale = 1.
rotation_degree = 0.  # 0., counter-clock-wise
output_shape = None  #(128, 128)  # or None
output_folder = '/home/mengqi/fileserver/results/fluorescence/20190301_retina_blood_vessel/20190313_retina_to_quantum20190311_AHE_fuse_5_residual-2k-only_test/results_test_1024_512_crop_and_scale/nc1_resnet_9blocks_256_128_crop_and_rotate_and_scale/test_240/selected/fake_A_subfolders/20190311_diff_21-filter-fuse-color_code/median_fuse_n_5-[80_150]-median'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if rotation_degree != 0:    
    rows, cols, _ = output_shape
    affine_mat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)    

for curdir, subdirs, files in os.walk(root, followlinks=True):
    files.sort()
    for ind, file in enumerate(files):
        img_path = os.path.join(curdir, file)

        img = cv2.imread(img_path, flags = cv2.IMREAD_GRAYSCALE)  #, flags = cv2.IMREAD_GRAYSCALE
        if scale != 1.:
            img = cv2.resize(img, dsize=(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        if output_shape is not None:
            output_img = np.ones(output_shape, dtype=img.dtype)
            height = min(img.shape[0], output_shape[0])
            width = min(img.shape[1], output_shape[1])
            output_img[:height, :width] = img[:height, :width]
            img = output_img

        if rotation_degree != 0:    # rotate image
            img = cv2.warpAffine(img, affine_mat, (cols,rows))


        # name_digits = re.findall(r"[-+]?\d*\.\d+|\d+", file)
        # # name_output = '{}-{}_{}_fake_A.png'.format(name_segs[0], name_segs[2], name_segs[1])
        # # subfolder_name = '{}_diff_{}'.format(name_digits[0], int(name_digits[2])- int(name_digits[1]))
        # # subfolder = os.path.join(output_folder, subfolder_name)
        # # if not os.path.exists(subfolder):
        # #     os.makedirs(subfolder)

        # img_orig = cv2.imread('/home/mengqi/temp/svr2/fileserver/mengqi/results/fluorescence/20190311_quantum/1_crop_orig_color/{}.png'.format(name_digits[2]), 
        #         flags = cv2.IMREAD_GRAYSCALE)  #, flags = cv2.IMREAD_GRAYSCALE
        # img_orig = cv2.resize(img_orig, dsize=(0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        # output_img = np.ones((512, 512*2), dtype=img.dtype)
        # height = min(img_orig.shape[0], 512)
        # width = min(img_orig.shape[1], 512)
        # output_img[70:70+height, :width] = img_orig[:height, :width]
        # output_img[:, 512:] = img

        # # if file.startswith('201903'):
        # #     name_segs = file.split('_')
        # #     name_output = '{}-{}_{}_fake_A.png'.format(name_segs[0], name_segs[2], name_segs[1])
        # # else:
        # #     name_segs = file.split('_')
        # #     name_output = '20190311-{}_{}_fake_A.png'.format(name_segs[1], name_segs[0])
        # output_img_path = os.path.join(output_folder, file)  # "{:06}.png".format(ind))
        # cv2.imwrite(output_img_path, output_img)


        # img[img<128] = 0  #
        # img[img>=128] = 255
        # img = signal.medfilt(img, kernel_size=5)
        # img = ndimage.morphology.grey_opening(img, size=15)
        # img = exposure.equalize_hist(img) * 255  # histogram equalization
        # img = exposure.equalize_adapthist(img, kernel_size=None, clip_limit=0.01) * 255  # adaptive histogram equalizations
           
        output_img_path = os.path.join(output_folder, file)
        cv2.imwrite(output_img_path, img)
        print(output_img_path)
    # print("cur", curdir)
    # print("sub", subdirs)
    # print("files", files)