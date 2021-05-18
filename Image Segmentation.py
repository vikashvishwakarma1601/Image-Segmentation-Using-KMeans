import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy

im = cv2.imread('butterfly.jpg') 
print(im.shape)
original_shape = im.shape
print(original_shape[0])
cv2.imshow('Original Image',im) 

#  Flatten Each Channel of The Image
all_pixels = im.reshape(( -1,3))
print(all_pixels.shape) 

dominant_colors = 4#  Number of Main Color Out of Entire Picture

km = KMeans(n_clusters =dominant_colors )
km.fit(all_pixels) 

#  Main Colors in Floating 
centers = km.cluster_centers_
print("Dominant Colors in Floating") 
print(centers,end="\n\n")

#  Main Colors in Integer
centers = np.array(centers,dtype='uint8')
print("Dominant Colors In Integer" ) 
print( centers)

#  Printing Subplot Of The Dominant Colors
i = 1

plt.figure( figsize=(4,2) )

colors = []

for each_col in centers:
     plt.subplot(1,dominant_colors,i)
     plt.axis('off') 
     i+=1
     
     colors.append( each_col)

     a= np.zeros( (100,100,3),dtype='uint8')
     a[:,:,:] = each_col
     plt.imshow(a)
plt.show( )


new_img = np.zeros((original_shape[0]* original_shape[1],3),dtype='uint8')


for ix in range(new_img.shape[0]) :
     new_img[ix] = colors[km.labels_[ix]]

new_img = new_img.reshape( ( original_shape) )
cv2.imshow('new_img',new_img) 
