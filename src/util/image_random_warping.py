import os
import numpy as np
from PIL import Image


input_folder = '/home/mengqi/fileserver/results/fluorescence/20190421_ICG_brainBloodBarrier/mouse6_w_brainBloodBarrier/3-None_HE-diff/wo_fuse'
output_folder = '/home/mengqi/fileserver/results/fluorescence/20190421_ICG_brainBloodBarrier/mouse6_w_brainBloodBarrier/3-None_HE-diff/wo_fuse_contrast10'





for curdir, subdirs, files in os.walk(input_folder, followlinks=True):
    for file_name in files:
        input_img_name = os.path.join(curdir, file_name)
        W, H = Image.open(input_img_name).size
        shift = W / 10

        output_subfolder = curdir.replace(input_folder, output_folder)
        output_img_name = os.path.join(output_subfolder, file_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        rand_warp_img_cmd = 'convert {} -roll {:+}{:+} -swirl {} -roll {:+}{:+} -implode {} -roll {:+}{:+} -contrast-stretch 10% {}'.format(
            input_img_name,
            np.random.randint(-1*shift, shift), np.random.randint(-1*shift, shift),  # roll
            np.random.randint(-100, 100),  # swirl
            np.random.randint(-1*shift, shift), np.random.randint(-1*shift, shift),  # roll
            np.random.rand() - 0.5,  # implode
            np.random.randint(-1*shift, shift), np.random.randint(-1*shift, shift),  # roll
            output_img_name
        )
        cnvt_colorspace_cmd = 'convert {} -colorspace Gray {}'.format(input_img_name, output_img_name)
        cnvt_to_bw_cmd = 'convert {} -threshold 0% {}'.format(input_img_name, output_img_name)
        cnvt_intensity_cmd = 'convert {} -brightness-contrast 0x10 {}'.format(input_img_name, output_img_name)

        os.system(cnvt_intensity_cmd)
        print(output_img_name)


