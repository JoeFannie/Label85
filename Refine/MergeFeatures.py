# -*- coding: utf-8 -*-
caffe_root = '/home/caojiajiong/caffe-master/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import lmdb
import pylab as pltss
from pylab import *
import numpy as np
import matplotlib.pyplot as plt 
import scipy 
import scipy.io
import os.path

N = 17311							# Number of data instances  
M = 80										# Number of possible labels for each data instance

output_lmdb_path = '/home/caojiajiong/attribute_learning/AUX_test_42500_lmdb'
label = lmdb.open('/home/caojiajiong/attribute_learning/test_label')
label_txn = label.begin()
label_cursor = label_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
data_label = np.zeros((N, M,1,1))
count = 0

for key, value in label_cursor:
  datum.ParseFromString(value)
  data_label[count, :] = caffe.io.datum_to_array(datum).astype(float).reshape(M,1,1)
  count = count + 1

loss_list = ['loss_5_o_Clock_Shadow', 'loss_Arched_Eyebrows', 'loss_Attractive', 'loss_Bags_Under_Eyes', \
'loss_Bald', 'loss_Bangs', 'loss_Big_Lips', 'loss_Big_Nose', \
'loss_Black_Hair', 'loss_Blond_Hair', 'loss_Blurry', 'loss_Brown_Hair', \
'loss_Bushy_Eyebrows', 'loss_Chubby', 'loss_Double_Chin', 'loss_Eyeglasses', \
'loss_Goatee', 'loss_Gray_Hair', 'loss_Heavy_Makeup', 'loss_High_Cheekbones', \
'loss_Male', 'loss_Mouth_Slightly_Open', 'loss_Mustache', 'loss_Narrow_Eyes', \
'loss_No_Beard', 'loss_Oval_Face', 'loss_Pale_Skin', 'loss_Pointy_Nose', \
'loss_Receding_Hairline', 'loss_Rosy_Cheeks', 'loss_Sideburns', 'loss_Smiling', \
'loss_Straight_Hair', 'loss_Wavy_Hair', 'loss_Wearing_Earrings', 'loss_Wearing_Hat', \
'loss_Wearing_Lipstick', 'loss_Wearing_Necklace', 'loss_Wearing_Necktie', 'loss_Young']

loss_count = 0;
data_loss = np.zeros((N, M,1,1))
for loss in loss_list:
	loss_name = '/home/caojiajiong/attribute_learning/features/MCNN_42500_test/' + loss
	feature = lmdb.open(loss_name)
	feature_txn = feature.begin()
	feature_cursor = feature_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()
	count = 0

	for key, value in feature_cursor:
	  datum.ParseFromString(value)
	  data_loss[count, loss_count*2:loss_count*2+2] = caffe.io.datum_to_array(datum).reshape(2,1,1)
	  count = count + 1
	  if count>=N:
	    break;
	loss_count = loss_count + 1
	print loss

np.savetxt('/home/caojiajiong/attribute_learning/features/MCNN_42500_test/loss_all.txt', data_loss, delimiter = ' ')

data_merge = np.zeros((N, M*2, 1, 1))
data_merge[:, 0:M, :, :] = data_loss
data_merge[:, M:2*M, :, :] = data_label

for idx in range(int(math.ceil(N/1000.0))):
	in_db_label = lmdb.open(output_lmdb_path, map_size=int(1e12))
	with in_db_label.begin(write=True) as in_txn:
		for label_idx, label_ in enumerate(data_merge[(1000*idx):(1000*(idx+1))]):
			im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(2*M,1,1))
			in_txn.put('{:0>10d}'.format(1000*idx+label_idx), im_dat.SerializeToString())
			string_ = str(1000*idx+label_idx+1) + ' / ' + str(N)
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
	in_db_label.close()
	
	
  
  
  
  
  
  
