#!/bin/bash

for i in {50000..100000..10000}
do
  name=/home/zhangyuan/caojiajiong/attribute_learning/features_all/dropout_inception_conv23attr_111_decay7_$i
  mkdir $name
  model=/home/zhangyuan/caojiajiong/attribute_learning/snapshot/dropout_inception_conv23attr_111_decay7_iter_$i.caffemodel.h5
  /home/zhangyuan/caojiajiong/caffe-HIK/build/tools/extract_features.bin \
$model \
/home/zhangyuan/caojiajiong/attribute_learning/prototxt/dropout_inception_conv23attr_111_deploy.prototxt \
loss_5_o_Clock_Shadow,loss_Arched_Eyebrows,loss_Attractive,loss_Bags_Under_Eyes,\
loss_Bald,loss_Bangs,loss_Big_Lips,loss_Big_Nose,\
loss_Black_Hair,loss_Blond_Hair,loss_Blurry,loss_Brown_Hair,\
loss_Bushy_Eyebrows,loss_Chubby,loss_Double_Chin,loss_Eyeglasses,\
loss_Goatee,loss_Gray_Hair,loss_Heavy_Makeup,loss_High_Cheekbones,\
loss_Male,loss_Mouth_Slightly_Open,loss_Mustache,loss_Narrow_Eyes,\
loss_No_Beard,loss_Oval_Face,loss_Pale_Skin,loss_Pointy_Nose,\
loss_Receding_Hairline,loss_Rosy_Cheeks,loss_Sideburns,loss_Smiling,\
loss_Straight_Hair,loss_Wavy_Hair,loss_Wearing_Earrings,loss_Wearing_Hat,\
loss_Wearing_Lipstick,loss_Wearing_Necklace,loss_Wearing_Necktie,loss_Young \
$name/loss_5_o_Clock_Shadow,$name/loss_Arched_Eyebrows,$name/loss_Attractive,$name/loss_Bags_Under_Eyes,\
$name/loss_Bald,$name/loss_Bangs,$name/loss_Big_Lips,$name/loss_Big_Nose,\
$name/loss_Black_Hair,$name/loss_Blond_Hair,$name/loss_Blurry,$name/loss_Brown_Hair,\
$name/loss_Bushy_Eyebrows,$name/loss_Chubby,$name/loss_Double_Chin,$name/loss_Eyeglasses,\
$name/loss_Goatee,$name/loss_Gray_Hair,$name/loss_Heavy_Makeup,$name/loss_High_Cheekbones,\
$name/loss_Male,$name/loss_Mouth_Slightly_Open,$name/loss_Mustache,$name/loss_Narrow_Eyes,\
$name/loss_No_Beard,$name/loss_Oval_Face,$name/loss_Pale_Skin,$name/loss_Pointy_Nose,\
$name/loss_Receding_Hairline,$name/loss_Rosy_Cheeks,$name/loss_Sideburns,$name/loss_Smiling,\
$name/loss_Straight_Hair,$name/loss_Wavy_Hair,$name/loss_Wearing_Earrings,$name/loss_Wearing_Hat,\
$name/loss_Wearing_Lipstick,$name/loss_Wearing_Necklace,$name/loss_Wearing_Necktie,$name/loss_Young \
200 lmdb GPU 3
echo "$i"
done
