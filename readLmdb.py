caffe_root = '/home/caojiajiong/caffe-master/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import lmdb
import numpy as np

label = lmdb.open('/home/caojiajiong/attribute_learning/test_label')
label_txn = label.begin()
label_cursor = label_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
data_label = np.zeros((17311, 80))
count = 0

for key, value in label_cursor:
  datum.ParseFromString(value)
  data_label[count, :] = caffe.io.datum_to_array(datum).reshape(80)
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
accuracy = np.zeros((41))
for loss in loss_list:
	loss_name = '/home/caojiajiong/attribute_learning/features/MCNN_32500_test/' + loss
	save_name = '/home/caojiajiong/attribute_learning/features/MCNN_32500_test/' + loss + '.txt'
	feature = lmdb.open(loss_name)
	feature_txn = feature.begin()
	feature_cursor = feature_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()
	data = np.zeros((17400, 2))
	count = 0

	for key, value in feature_cursor:
	  datum.ParseFromString(value)
	  data[count, :] = caffe.io.datum_to_array(datum).reshape(2)
	  count = count + 1
	data[0:-1, :] = data[0:-1, :] > 0.5
	data = data.astype(float)
	bool_matrix = data[0:17311, 0] == data_label[0:17311, loss_count*2]
	bool_matrix = bool_matrix.astype(float)
	accuracy[loss_count] = bool_matrix.sum() / 17311.0
	loss_count = loss_count + 1
	print loss
	np.savetxt(save_name, data, delimiter = ' ')
accuracy[-1] = accuracy.sum()/40
np.savetxt('/home/caojiajiong/attribute_learning/features/MCNN_32500_test/accuracy.txt', accuracy, delimiter = ' ')
	
  
  
  
  
  
  
