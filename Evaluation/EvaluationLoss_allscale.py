caffe_root = '/home/zhangyuan/caojiajiong/caffe-HIK-crop/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import lmdb
import numpy as np

label = lmdb.open('/home/zhangyuan/caojiajiong/attribute_learning/data/test_all_label')
label_txn = label.begin()
label_cursor = label_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
data_label = np.zeros((19962, 80))
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

file = open('accuracy_ensemble_branch_conv5_fusion_7W_allscale.txt', 'a')
feat_file = open('feat_branch_conv5_fusion_7W_allscale.txt', 'w')
for i in range(0, 1):
	loss_count = 0;
	accuracy = np.zeros((41))
	if i >= 1:
		file.writelines('\n')
	file.writelines(str(i*10000+60000)+'\n')
	for loss in loss_list:
	    # ensemble number
		feat_file.writelines(loss+'\n')
		N = 10
		data = np.zeros((20000, 2))
		name_list = ['_wh112_7W', '_wh116_7W', '_wh120_7W', '_wh124_7W', '_6W']
		for n in range(0, N):
			for l in range(0, 5):
				dir_name = '/home/zhangyuan/caojiajiong/attribute_learning/features_all/branch_conv5_fusion' + name_list[l] + '_' + str(1 + n*1) + '/'
				loss_name = dir_name + loss
				#save_name = dir_name + loss + '.txt'
				feature = lmdb.open(loss_name)
				feature_txn = feature.begin()
				feature_cursor = feature_txn.cursor()
				datum = caffe.proto.caffe_pb2.Datum()
				count = 0

				for key, value in feature_cursor:
				  datum.ParseFromString(value)
				  data[count, :] = data[count, :] + caffe.io.datum_to_array(datum).reshape(2)
				  count = count + 1
		for k in range(0, 19963):
			feat_file.writelines(str(data[k, 0]) + ' ' + str(data[k, 1]) + '\n')
		data[0:-1, :] = data[0:-1, :] > 0.5 * 5 * N
		data = data.astype(float)
		bool_matrix = data[0:19962, 0] == data_label[0:19962, loss_count*2]
		bool_matrix = bool_matrix.astype(float)
		accuracy[loss_count] = bool_matrix.sum() / 19962.0
		loss_count = loss_count + 1
		print loss
		#np.savetxt(save_name, data, delimiter = ' ')
	accuracy[-1] = accuracy.sum()/40
	for j in range(0, 41):
		file.writelines(str(accuracy[j])+'\n')
file.close()
feat_file.close()
	#accuracy_name = dir_name + 'accuracy.txt'
	#np.savetxt(accuracy_name, accuracy, delimiter = ' ')
	
  
  
  
  
  
  
