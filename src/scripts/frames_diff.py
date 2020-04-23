import os
import cv2
import numpy as np 

frame_range = [22, 108]
max_frame_ind_diff = 50
test_every_N_frame = 5
# 1: [1293, 1313], 2: [1063, 1171], 3: [561, 666], 4: [232, 356]

root = '/home/mengqi/temp/svr2/fileserver/mengqi/results/fluorescence/20190312_quantum/2_fuse_gray/[22, 108]_fuse_n_5'
# subfolders = os.listdir(root)

# for subfolder in subfolders:

input_folder = root  #os.path.join(root, subfolder)
# output_folder = input_folder + '/../../frames_residual/frame_diff_{}/{}'.format(frame_range, subfolder)
output_folder = '/home/mengqi/temp/svr2/fileserver/mengqi/results/fluorescence/20190312_quantum/2_fuse_5_gray-residual_50max'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



for f_i in range(frame_range[0], frame_range[1]):
    img_i = cv2.imread(os.path.join(input_folder, "{:06}.png".format(f_i)), flags = cv2.IMREAD_GRAYSCALE)
    for f_j in range(f_i + 1, min(frame_range[1], f_i + max_frame_ind_diff), test_every_N_frame):
        try:
            img_j = cv2.imread(os.path.join(input_folder, "{:06}.png".format(f_j)), flags = cv2.IMREAD_GRAYSCALE)
            img_diff = img_j.astype(np.int32) - img_i.astype(np.int32)
            output_img_path = os.path.join(output_folder, "{:06}_{:06}.png".format(f_j, f_i))
            cv2.imwrite(output_img_path, img_diff)
            print(output_img_path)
        except:
            print('Error')
            break

