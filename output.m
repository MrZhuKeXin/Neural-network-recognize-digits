function y = output( i,w,b )
%OUTPUT ����ĳһ����Ԫ�����
%   ����˳��Ϊi,w,b;i����input��w����weights��b����biases
temp=w*i+b;
y=sigmoid(temp);
end

