load Patches_align.mat;
img_root = 'D:\DataSets\img_align_celeba';
write_dir = 'D:\DataSets';
for i = 1 : 100
   I = imread(fullfile(img_root, Patches{i}.img_name));
   d_eye = max(40, Patches{i}.righteye_x - Patches{i}.lefteye_x);
   d_mouth = Patches{i}.rightmouth_x - Patches{i}.leftmouth_x;
   d_eyenose = (2*Patches{i}.nose_y-Patches{i}.righteye_y-Patches{i}.lefteye_y)/2.0;
   d_eyemouth = ((Patches{i}.leftmouth_y - Patches{i}.lefteye_y)+(Patches{i}.rightmouth_y - Patches{i}.righteye_y))/2.0;
   %left eye
   bbox_lefteye = [Patches{i}.lefteye_x-0.6*d_eye, Patches{i}.lefteye_y-0.6*d_eye, 1.2*d_eye, 1.2*d_eye];
   I_lefteye = imcrop(I, bbox_lefteye);
   if(~exist(fullfile(write_dir, 'LeftEye'), 'dir'))
     mkdir(fullfile(write_dir, 'LeftEye'));
   end
   imwrite(I_lefteye, fullfile(write_dir, 'LeftEye', Patches{i}.img_name));
   %right eye
   bbox_righteye = [Patches{i}.righteye_x-0.6*d_eye, Patches{i}.righteye_y-0.6*d_eye, 1.2*d_eye, 1.2*d_eye];
   I_righteye = imcrop(I, bbox_righteye);
   if(~exist(fullfile(write_dir, 'RightEye'), 'dir'))
     mkdir(fullfile(write_dir, 'RightEye'));
   end
   imwrite(I_righteye, fullfile(write_dir, 'RightEye', Patches{i}.img_name));
   
   %eye nose
   bbox_eyenose = [Patches{i}.lefteye_x-0.2*d_eye, Patches{i}.lefteye_y-0.5*d_eye, 1.4*d_eye, 1.4*d_eye];
   I_eyenose = imcrop(I, bbox_eyenose);
   if(~exist(fullfile(write_dir, 'EyeNose'), 'dir'))
     mkdir(fullfile(write_dir, 'EyeNose'));
   end
   imwrite(I_eyenose, fullfile(write_dir, 'EyeNose', Patches{i}.img_name));

   % eye hair
   bbox_eyehair = [Patches{i}.lefteye_x-0.7*d_eye, Patches{i}.lefteye_y-2*d_eye, 2.4*d_eye, 2.4*d_eye];
   I_eyehair = imcrop(I, bbox_eyehair);
   if(~exist(fullfile(write_dir, 'EyeHair'), 'dir'))
     mkdir(fullfile(write_dir, 'EyeHair'));
   end
   imwrite(I_eyehair, fullfile(write_dir, 'EyeHair', Patches{i}.img_name));
   
   %nose mouth
   bbox_nosemouth = [Patches{i}.lefteye_x-0.2*d_eye, Patches{i}.lefteye_y+0.3*d_eye, 1.4*d_eye, 1.4*d_eye];
   I_nosemouth = imcrop(I, bbox_nosemouth);
   if(~exist(fullfile(write_dir, 'NoseMouth'), 'dir'))
     mkdir(fullfile(write_dir, 'NoseMouth'));
   end
   imwrite(I_nosemouth, fullfile(write_dir, 'NoseMouth', Patches{i}.img_name));
   
   %eye nose mouth
   bbox_eyenosemouth = [Patches{i}.lefteye_x-0.4*d_eye, Patches{i}.lefteye_y-0.4*d_eye, 1.8*d_eye, 1.8*d_eye];
   I_eyenosemouth = imcrop(I, bbox_eyenosemouth);
   if(~exist(fullfile(write_dir, 'EyeNoseMouth'), 'dir'))
     mkdir(fullfile(write_dir, 'EyeNoseMouth'));
   end
   imwrite(I_eyenosemouth, fullfile(write_dir, 'EyeNoseMouth', Patches{i}.img_name));
   
   %face 
   bbox_face = [Patches{i}.lefteye_x-0.6*d_eye, Patches{i}.lefteye_y-0.6*d_eye, 2.2*d_eye, 2.2*d_eye];
   I_face = imcrop(I, bbox_face);
   if(~exist(fullfile(write_dir, 'Face'), 'dir'))
     mkdir(fullfile(write_dir, 'Face'));
   end
   imwrite(I_face, fullfile(write_dir, 'Face', Patches{i}.img_name));

end
