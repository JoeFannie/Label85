%The code generates id labels and attribute labels for CelebA dataset
%based on attribute labels from the original dataset as well as the score
%file generated from the trained Classification network.
clear all;
close all;
clc;
score_file = fopen('../list/CelebA_label.txt', 'r');
train_file = fopen('../list/train_10p.txt', 'w');
val_file = fopen('../list/val_10p.txt', 'w');
data = textscan(score_file, '%d %f');
identity = data{1};
score = data{2};
identity = identity(1:184884);
score = score(1:184884);
identity_70 = [];
name_70 = [];
for i = 1 : size(identity)
   if(score(i) >= 0.1)
      identity_70(end+1) = identity(i);
      name_70(end+1) = i;
   end
end

[identity_70, I] = sort(identity_70);
name_70 = name_70(I);
count = -1;
previous_id = -1;
id_count = 0;
for i = 1 : size(identity_70, 2)
    if(identity_70(i) ~= previous_id)
        count = count + 1;
        id_count = 1;
    else
        id_count = id_count+1;
    end
       id_sum = sum(identity_70==identity_70(i));
   if(id_sum >= 8 && id_count <= id_sum-2)
         fprintf(train_file, '%s %d\n', [sprintf('%06d',name_70(i)), '.jpg'], count);
   end
   if(id_sum >= 8 && id_count >= id_sum-1)
         fprintf(val_file, '%s %d\n', [sprintf('%06d',name_70(i)), '.jpg'], count);
   end
   if(id_sum < 8)
          fprintf(train_file, '%s %d\n', [sprintf('%06d',name_70(i)), '.jpg'], count);

   end 
   previous_id = identity_70(i);
end
fclose all;

load attributes_new.mat;;
train_file = fopen('../list/train_10p.txt', 'r');
train_file_shuffle = fopen('../list/train_10p_shuffle.txt', 'w');
train_data = textscan(train_file, '%s %d');
names = train_data{1};
ids = train_data{2};
attr_train_10p = zeros(length(names), 80);
perm = randperm(length(names));
names = names(perm);
ids = ids(perm);
for i = 1 : length(names)
   fprintf(train_file_shuffle, '%s %d\n', names{i}, ids(i));
   name = names{i};
   attr_train_10p(i, :) = attributes_new(str2double(name(1:end-4)), :);
end

val_file = fopen('../list/val_10p.txt', 'r');
val_data = textscan(val_file, '%s %d');
names = val_data{1};
ids = val_data{2};
attr_val_10p = zeros(length(names), 80);
for i = 1 : length(names)
   name = names{i};
   attr_val_10p(i, :) = attributes_new(str2double(name(1:end-4)), :);
end

fclose all;
save attr_train_10p.mat attr_train_10p;
save attr_val_10p.mat attr_val_10p;
