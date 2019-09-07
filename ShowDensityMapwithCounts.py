from matplotlib import cm 
import h5py
import numpy as np 
import os
import glob
import PIL.Image as Image
import matplotlib.pyplot as plt


#show the density map corresponding to the image.
#density map
file_path = 'path of your .h5 file'
gt_file = h5py.File(file_path,'r')
groundtruth = np.asarray(gt_file['density'])
image = plt.imshow(groundtruth,cmap=cm.CMRmap)
plt.show(image)
print("Sum = " ,np.sum(groundtruth))

#open the image corresponding to the density map.
file = 'path of your jpg file of correspnding .h5 file'
imag = Image.open(file.replace('.h5', '.jpg').replace('ground-truth', 'images'))
a = plt.imshow(imag)
plt.show(a)
