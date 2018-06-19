function y = output( i,w,b )
%OUTPUT 返回某一层神经元的输出
%   参数顺序为i,w,b;i代表input，w代表weights，b代表biases
temp=w*i+b;
y=sigmoid(temp);
end

