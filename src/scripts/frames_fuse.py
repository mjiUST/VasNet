import os
import cv2
import numpy as np 
#from skimage import exposure

frame_range = [60, 200]
# fuse_n_frames = 2 # range(2, frame_range[1]-frame_range[0]+1)
for fuse_n_frames in [3, 5]:  #range(3, 5): # range(2, frame_range[1]-frame_range[0]+1)

    root = '/home/mengqi/temp/svr2/fileserver/mengqi/results/fluorescence/20190311_quantum/1.1_AHE_gray'
    output_folder = '/home/mengqi/temp/svr2/fileserver/mengqi/results/fluorescence/20190311_quantum/2_AHE_fuse_gray/{}_fuse_n_{}'.format(frame_range, fuse_n_frames) 

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for f_i in range(frame_range[0], frame_range[1]-fuse_n_frames+1):
        img_fuse = cv2.imread(os.path.join(root, "{:06}.png".format(f_i)), flags = cv2.IMREAD_GRAYSCALE).astype(np.uint32)  # , flags = cv2.IMREAD_GRAYSCALE
        imgs = np.repeat(img_fuse[..., None], fuse_n_frames, axis=-1)
        for _shift in range(1, fuse_n_frames):
            imgs[..., _shift] = cv2.imread(os.path.join(root, "{:06}.png".format(f_i + _shift)), flags = cv2.IMREAD_GRAYSCALE).astype(np.uint32)  # , flags = cv2.IMREAD_GRAYSCALE

        # img_fuse = np.median(imgs, axis=-1)
        img_fuse = np.mean(imgs, axis=-1)

        output_img_path = os.path.join(output_folder, "{:06}.png".format(f_i))
        cv2.imwrite(output_img_path, img_fuse)
        print(output_img_path)

