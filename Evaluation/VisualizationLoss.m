train_file = fopen('../attr_id/log/v4_bn.log.train', 'r');
train_log = textscan(train_file, '%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
only_file = fopen('../attr_id/log/only.log.train', 'r');
only_log = textscan(only_file, '%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
x = only_log{1};

for i = 1 : 40
loss_train = train_log{i+4};
loss_only = only_log{i+4};
% for i = 1 : 39
%    loss_train = train_log{i+5} + loss_train; 
% end
% for i = 1 : 39
%    loss_only = only_log{i+5} + loss_only; 
% end
[x_sample, loss_train] = filtData(x, loss_train(1:size(x,1)), 16);
[~, loss_only] = filtData(x, loss_only, 16);

plot(x_sample, loss_train, 'r');
hold on;
plot(x_sample, loss_only, 'g');
hold off;
disp(num2str(i));
pause;
end
% axis([x(1), x(end), 10, 20]);

% train_file = fopen('train_shuffle.txt', 'r');
% new_file = fopen('train_shuffle_new.txt', 'w');
% train_list = textscan(train_file, '%s %d');
% attr_train_new = zeros(35996, 80);
% names = train_list{1};
% ids = train_list{2};
% overlap_id = [];
% count = 0;
% j = 1;
% for i = 1 : length(names)
%    id = str2double(names{i}(1:end-4));
%    if(id >= 184885)
%       overlap_id(end+1) = id;
%       count = count + 1;
%    else
%       fprintf(new_file, '%s %d\n', names{i}, ids(i));
%       attr_train_new(j, :) = attr_train(i, :);
%       j = j + 1;
%    end
% end
% 
% disp(num2str(count));
% save attr_train_new.mat attr_train_new;
fclose all;
