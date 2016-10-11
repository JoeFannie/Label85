function [ x_sample, y_new ] = filtData( x, y, n )
   l = length(y) / n;
   y = y(1:l*n);
   x = x(1:l*n);
   for i = 1 : l
      y_new(i) = sum(y((i-1)*n+1:i*n)) / n;
      x_sample(i) = x(i*n/2);
   end

end

