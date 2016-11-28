function [ flag ] = VisualizeLoss( varargin )
%UNTITLED10 此处显示有关此函数的摘要
%   此处显示详细说明
N = nargin-1;
color_table = ['r', 'g', 'b', 'k', 'y', 'm', 'c'];
sample_rate = varargin{end};
loss_gender = cell(N, 1);
loss_nose = cell(N, 1);
loss_mouth = cell(N, 1);
loss_eyes = cell(N, 1);
loss_face = cell(N, 1);
loss_rest = cell(N, 1);
x_sample_all = cell(N, 1);
figure;
hold on;
for i = 1 : N
    file = fopen(varargin{i}, 'r');
    line = strsplit(fgets(file));
    switch length(line)
        case 45
            log = textscan(file, '%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
        case 46
            log = textscan(file, '%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
            [~, loss_mid] = filtData(log{1}, log{end}, sample_rate);
        case 47
            log = textscan(file, '%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
            [~, loss_mid] = filtData(log{1}, log{end-1}, sample_rate);
    end
    loss_attr = log{5};
    for j = 1 : 39
       loss_attr = loss_attr + log{j+5}; 
    end
    [x_sample, loss_attr] = filtData(log{1}, loss_attr, sample_rate);
    if(length(line) >= 46)
%         [AX, H1, H2] = plotyy(x_sample, loss_attr, x_sample, loss_mid);
%         set(H1, 'Linestyle', '-', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5);
%         set(H2, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5);
        plot(x_sample, loss_mid, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5);
    end
    plot(x_sample, loss_attr, 'Linestyle', '-', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5);
    fclose all;
    % group loss
    x_sample_all{i, 1} = log{1};
    loss_gender{i, 1} = log{21+4};
    loss_nose{i, 1} = log{8+4} + log{28+4};
    loss_mouth{i, 1} = log{7+4} + log{32+4} + log{37+4} + log{22+4};
    loss_eyes{i, 1} = log{2+4} + log{4+4} + log{13+4} + log{16+4} + log{24+4};
    loss_face{i, 1} = log{3+4} + log{11+4} + log{26+4} + log{19+4} + log{27+4} + log{40+4};
    loss_rest{i, 1} = log{1+4} + log{5+4} + log{6+4} + log{9+4} + log{10+4} + log{12+4} + log{14+4} ...
         + log{15+4} + log{17+4} + log{18+4} + log{20+4} + log{23+4} + log{25+4} + log{29+4} ...
          + log{30+4} + log{31+4} + log{33+4} + log{34+4} + log{35+4} + log{36+4} + log{38+4} + log{39+4};
end
hold off;
%gender group
figure;
hold on;
for i = 1 : N
   [x_sample, loss_attr] = filtData(x_sample_all{i, 1}, loss_gender{i, 1}, sample_rate); 
   plot(x_sample, loss_attr, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5); 
end
title('gender group');
hold off;
%nose group
figure;
hold on;
for i = 1 : N
   [x_sample, loss_attr] = filtData(x_sample_all{i, 1}, loss_nose{i, 1}, sample_rate); 
   plot(x_sample, loss_attr, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5); 
end
title('nose group')
hold off;
%mouth group
figure;
hold on;
for i = 1 : N
   [x_sample, loss_attr] = filtData(x_sample_all{i, 1}, loss_mouth{i, 1}, sample_rate); 
   plot(x_sample, loss_attr, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5);
end
title('mouth group')
hold off;
%eyes group
figure;
hold on;
for i = 1 : N
   [x_sample, loss_attr] = filtData(x_sample_all{i, 1}, loss_eyes{i, 1}, sample_rate); 
   plot(x_sample, loss_attr, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5); 
end
title('eyes group')
hold off;
%face group
figure;
hold on;
for i = 1 : N
   [x_sample, loss_attr] = filtData(x_sample_all{i, 1}, loss_face{i, 1}, sample_rate); 
   plot(x_sample, loss_attr, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5);
end
title('face group')
hold off;
%rest group
figure;
hold on;
for i = 1 : N
   [x_sample, loss_attr] = filtData(x_sample_all{i, 1}, loss_rest{i, 1}, sample_rate); 
   plot(x_sample, loss_attr, 'Linestyle', '--', 'color', color_table(mod(i-1, 7) + 1), 'Linewidth', 2.5); 
end
title('rest group')
hold off;
end

