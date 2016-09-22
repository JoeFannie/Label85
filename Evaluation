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

for k in range(0,3):
	AUX_features = lmdb.open('/home/caojiajiong/attribute_learning/features/AUX_features_' + str(10000+2500*k))
	AUX_features_txn = AUX_features.begin()
	AUX_features_cursor = AUX_features_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()
	data_AUX_features = np.zeros((17311, 80))
	count = 0

	for key, value in AUX_features_cursor:
	  datum.ParseFromString(value)
	  data_AUX_features[count, :] = caffe.io.datum_to_array(datum).reshape(80)
	  count = count + 1
	  if count >= 17311:
		break;
		
	accuracy = np.zeros((41))
	bool_matrix = (data_AUX_features>0.5).astype(float)
	for i in range(0,40):
	  accuracy[i] = (bool_matrix[:, 2*i] == data_label[:, 2*i]).sum() / 17311.0
	accuracy[-1] = accuracy.sum() / 40.0
	np.savetxt('/home/caojiajiong/attribute_learning/features/AUX_features_' + str(10000+2500*k) + '/accuracy.txt', accuracy, delimiter = ' ')
  
  
  
  
  
  
