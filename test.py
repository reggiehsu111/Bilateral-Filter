from joint_bilateral_filter import *
import numpy as np
import cv2
import time
import pickle
from tqdm import tqdm
import threading
import sys
import os

bgr_img = cv2.imread('testdata/0b.png').astype(int)

# img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
jbf = Joint_bilateral_filter(2, 0.2)
bf_img = jbf.joint_bilateral_filter(bgr_img, bgr_img)


sigma_s = 2
sigma_r = 0.2*255
print("==============")
print("SIGMA_S {} SIGMA_R {}".format(sigma_s, sigma_r/255))

r = 3*sigma_s # 對應的 window size 是 2*r + 1

bgr_img_padded = cv2.copyMakeBorder(bgr_img,r,r,r,r,cv2.BORDER_REPLICATE)

# Create spacial kernel here, can be shared accross both range kernels
spacial_kernel = np.zeros((2*r+1, 2*r+1))
for i in range(1,r+1):
    spacial_kernel[r+i] += i**2
    spacial_kernel[r-i] += i**2
    spacial_kernel[:,r+i] += i**2
    spacial_kernel[:,r-i] += i**2
spacial_kernel = np.exp(-spacial_kernel / (2*sigma_s**2))
#         print('Spacial kernel OK')

# Create color_output_image (bilateral filtered image)
color_output_image = np.zeros((bgr_img.shape[0], bgr_img.shape[1], bgr_img.shape[2]))
for i in range(bgr_img.shape[0]):
    for j in range(bgr_img.shape[1]):
        center_pixel = bgr_img_padded[i+r][j+r] # 3 channels
        range_kernel = bgr_img_padded[i:i+2*r+1, j:j+2*r+1] - center_pixel # 3 channels
        range_kernel = np.exp(-np.sum(range_kernel**2, axis=2) / (2*sigma_r**2))

        kernel = spacial_kernel * range_kernel # 1 channel only

        color_output_image[i][j] = np.sum(bgr_img_padded[i:i+2*r+1, j:j+2*r+1] * np.expand_dims(kernel,2), axis=(0,1)) / np.sum(kernel)
cv2.imwrite('out_color_filtered.png', color_output_image) # sanity check
print("BF_Error:", np.sum(np.abs(bf_img - color_output_image)))
print("color_filtered generated")

for a in range(11): # 0~10
    for b in range(11-a):
        w_b = a/10
        w_g = b/10
        w_r = round(1 - w_b - w_g, 2)
        
        grayscale_img = (w_b*bgr_img[:,:,0] + w_g*bgr_img[:,:,1] + w_r*bgr_img[:,:,2]).astype(int)
        grayscale_img_padded = cv2.copyMakeBorder(grayscale_img,r,r,r,r,cv2.BORDER_REPLICATE).astype(int)
        cv2.imwrite('out_gray.png', grayscale_img)

        output_image = np.zeros((bgr_img.shape[0], bgr_img.shape[1], bgr_img.shape[2]))
        for i in range(bgr_img.shape[0]):
            for j in range(bgr_img.shape[1]):
                center_pixel = grayscale_img_padded[i+r][j+r]
                range_kernel = grayscale_img_padded[i:i+2*r+1, j:j+2*r+1] - center_pixel
                range_kernel = np.exp(-range_kernel**2 / (2*sigma_r**2))

                kernel = spacial_kernel * range_kernel

                output_image[i][j] = np.sum(bgr_img_padded[i:i+2*r+1, j:j+2*r+1] * np.expand_dims(kernel,2), axis=(0,1)) / np.sum(kernel)
#                 print('Finished output')
        jbf_img = jbf.joint_bilateral_filter(bgr_img, grayscale_img)
        print("JBF_Error:", np.sum(np.abs(jbf_img-output_image)))
        cv2.imwrite('out_gray_filtered.png', output_image)
        error = np.sum(abs(color_output_image - output_image))
        test_error = np.sum(abs(bf_img - jbf_img))
        print("test_error:", test_error)
        # print("color_output_image", color_output_image[0,0])
        # print("output_image", output_image[0,0])
        # print("bf_img", bf_img[0,0])
        # print("jbf_img", jbf_img[0,0])
        # print("abs(color_output_image-output_image)",np.abs(color_output_image-output_image)[0,0])
        # print("abs(bf_img-jbf_img)", np.abs(bf_img-jbf_img)[0,0])
        
        print("w_b {} w_g {} w_r {} : error {}".format(w_b, w_g, w_r, error))
        with open("log_sigmaS{}_sigmaR{}.txt".format(sigma_s, sigma_r/255), "a") as myfile:
            myfile.write("w_b {} w_g {} w_r {} : error {}\n".format(w_b, w_g, w_r, error))