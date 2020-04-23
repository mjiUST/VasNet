import os
import cv2
import numpy as np 
from skimage.measure import compare_ssim, compare_psnr


#root = '/home/mengqi/fileserver/results/cross_scale/train-L_HR-L_LR-pix2pix/results_test/1215_0100_nc3/test_latest/images'
root = '/home/mengqi/fileserver/results/cross_scale/train-L_HR-L_LR-cyclegan/results_test/1215_0100_nc3/test_3000epoch/images'


R_real_HR = cv2.imread(os.path.join(root, 'R_real_B.png'))
R_fake_HR = cv2.imread(os.path.join(root, 'R_fake_B.png'))
R_real_LR = cv2.imread(os.path.join(root, 'R_real_A.png'))

bicubic_psnr = compare_psnr(im_true = R_real_HR, im_test = R_real_LR)
synth_psnr = compare_psnr(im_true = R_real_HR, im_test = R_fake_HR)
print("PSNR: bicubic {}, synth {}".format(bicubic_psnr, synth_psnr))

bicubic_ssim = compare_ssim(X = R_real_HR, Y = R_real_LR, multichannel=True)
synth_ssim = compare_ssim(X = R_real_HR, Y = R_fake_HR, multichannel=True)
print("SSIM: bicubic {}, synth {}".format(bicubic_ssim, synth_ssim))
