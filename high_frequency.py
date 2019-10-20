import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

eindex = 1
srcdir = '/home/mingqing/InvSR/results/' + str(eindex) + '_InvbiGANSR_learnedy_block16_x4_gan/val_DIV2K/'
tardir = '/home/mingqing/InvSR/results_highfrequency/'
if not os.path.exists(tardir):
    os.mkdir(tardir)

for i in range(100):
    print(i)
    index = i + 801
    imgpth = srcdir + '0' + str(index) + '_forw.png'
    imgpth_save = tardir + str(index) + '_' + str(eindex) + '.png'
    img = cv2.imread(imgpth, 0)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    cv2.imwrite(imgpth_save, np.uint8(iimg))
    
