#!/bin/bash

/home/caojiajiong/caffe-master/build/tools/caffe train \
	-model "/home/caojiajiong/attribute_learning/attr_id/ClassificationCNN_bn_attr_v4_bn.prototxt" \
	-solver "/home/caojiajiong/attribute_learning/attr_id/ClassificationCNN_bn_attr_solver.prototxt" \
	-gpu 4
