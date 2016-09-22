# -*- coding: utf-8 -*-
import Mypc as pc 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import cv2
import pylab as pltss
from pylab import *
import scipy 
import scipy.io
import os.path

img_root = '/home/caojiajiong/DataSets/CelebA/CelebA/img_align_celeba/'
align_root = '/home/caojiajiong/DataSets/CelebA/CelebA/img_align_celeba_resize/'
N = 202599
Landmarks = np.zeros((202599,11))
mat_file = '/home/caojiajiong/attribute_learning/Landmarks.mat'
mat_contents = scipy.io.loadmat(mat_file)
Landmarks = mat_contents['Landmarks']

for i in range(0,202599):
	img_name = img_root + str('%06d' % (i+1)) + '.jpg'
	input_Y_img = cv2.imread(img_name)
	X_pts = np.asarray([[53,90],[101,90],[77,110],[63,140],[91,140]])
	Y_pts = np.asarray([Landmarks[i,1:3], Landmarks[i,3:5], Landmarks[i,5:7], Landmarks[i,7:9], Landmarks[i,9:11]])
	d,Z_pts,Tform = pc.procrustes(X_pts,Y_pts)
	R = np.eye(3)
	R[0:2,0:2] = Tform['rotation']
	S = np.eye(3) * Tform['scale'] 
	S[2,2] = 1
	t = np.eye(3)
	t[0:2,2] = Tform['translation']
	M = np.dot(np.dot(R,S),t.T).T
	tr_Y_img = cv2.warpAffine(input_Y_img,M[0:2,:],(159,190))
	align_name = align_root+str('%06d' % (i+1))+'.jpg'
	cv2.imwrite(align_name, tr_Y_img)
	if i % 1000 == 0:
		print i
	

# # Open images...
# #target_X_img = cv2.imread('arnie1.jpg',0)
# input_Y_img = cv2.imread('/home/caojiajiong/DataSets/CelebA/CelebA/img_align_celeba/000001.jpg',0)

# # Landmark points - same number and order!
# # l eye, r eye, nose tip, l mouth, r mouth
# X_pts = np.asarray([[53,90],[101,90],[77,110],[63,140],[91,140]])
# Y_pts = np.asarray([[69,109],[106,113],[77,142],[73,152],[108,153]])

# # Calculate transform via procrustes...
# d,Z_pts,Tform = pc.procrustes(X_pts,Y_pts)

# # Build and apply transform matrix...
# # Note: for affine need 2x3 (a,b,c,d,e,f) form
# R = np.eye(3)
# R[0:2,0:2] = Tform['rotation']
# S = np.eye(3) * Tform['scale'] 
# S[2,2] = 1
# t = np.eye(3)
# t[0:2,2] = Tform['translation']
# M = np.dot(np.dot(R,S),t.T).T
# tr_Y_img = cv2.warpAffine(input_Y_img,M[0:2,:],(190,159))

# cv2.imwrite('/home/caojiajiong/DataSets/CelebA/CelebA/img_align_celeba_resize/000001.jpg', tr_Y_img)

# Confirm points...
# aY_pts = np.hstack((Y_pts,np.array(([[1,1,1,1,1]])).T))
# tr_Y_pts = np.dot(M,aY_pts.T).T

# Show result - input transformed and superimposed on target...
# plt.figure() 
# plt.subplot(1,3,1)
# plt.imshow(target_X_img,cmap=cm.gray)
# plt.plot(X_pts[:,0],X_pts[:,1],'bo',markersize=5)
# plt.axis('off')
# plt.subplot(1,3,2)
# plt.imshow(input_Y_img,cmap=cm.gray)
# plt.plot(Y_pts[:,0],Y_pts[:,1],'ro',markersize=5)
# plt.axis('off')
# plt.subplot(1,3,3)
# plt.imshow(target_X_img,cmap=cm.gray)
# plt.imshow(tr_Y_img,alpha=0.6,cmap=cm.gray)
# plt.plot(X_pts[:,0],X_pts[:,1],'bo',markersize=5) 
# plt.plot(Z_pts[:,0],Z_pts[:,1],'ro',markersize=5) # same as...
# plt.plot(tr_Y_pts[:,0],tr_Y_pts[:,1],'gx',markersize=5)
# plt.axis('off')
# plt.show()
