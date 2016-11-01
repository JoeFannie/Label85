#!/bin/bash

for i in {20000..100000..5000}
do
  model=/home/caojiajiong/attribute_learning/snapshot/Res_refine_iter_$i.caffemodel.h5
  /home/caojiajiong/caffe-HIK/build/tools/extract_features.bin \
$model \
/home/caojiajiong/attribute_learning/prototxt/Res_refine_deploy.prototxt \
res1,res2,res3,res4,res5 \
/home/caojiajiong/attribute_learning/features/res1_15p_alldata_2W_$i,\
/home/caojiajiong/attribute_learning/features/res2_15p_alldata_2W_$i,\
/home/caojiajiong/attribute_learning/features/res3_15p_alldata_2W_$i,\
/home/caojiajiong/attribute_learning/features/res4_15p_alldata_2W_$i,\
/home/caojiajiong/attribute_learning/features/res5_15p_alldata_2W_$i \
200 lmdb GPU 5
echo "$i done"
done
