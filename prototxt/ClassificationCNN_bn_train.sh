#!/bin/bash

/home/caojiajiong/caffe-master/build/tools/caffe train \
	-model "/home/caojiajiong/attribute_learning/ClassificationCNN_bn.prototxt" \
	-solver "/home/caojiajiong/attribute_learning/ClassificationCNN_bn_solver.prototxt" \
  -snapshot "/home/caojiajiong/attribute_learning/snapshot/ClassificationCNN_bn_iter_150000.solverstate.h5" \
	-gpu 3,4
