function [ flag ] = VisualizeLoss( varargin )
%UNTITLED10 此处显示有关此函数的摘要
%   此处显示详细说明
N = nargin-1;
color_table = ['r', 'g', 'b', 'k', 'y'];
sample_rate = varargin{end};
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
%         set(H1, 'Linestyle', '-', 'color', color_table(mod(i, 5)), 'Linewidth', 2.5);
%         set(H2, 'Linestyle', '--', 'color', color_table(mod(i, 5)), 'Linewidth', 2.5);
        plot(x_sample, loss_mid, 'Linestyle', '--', 'color', color_table(mod(i, 5)), 'Linewidth', 2.5);
    end
    plot(x_sample, loss_attr, 'Linestyle', '-', 'color', color_table(mod(i, 5)), 'Linewidth', 2.5);
    fclose all;
end
hold off;
end

