%获取数据并将数据拆分成训练数据和测试数据
%前50000个为训练数据，后10000个为测试数据
IMGS=loadMNISTImages('train-images.idx3-ubyte');
TRAIN_IMGS=IMGS(:,1:50000);
TEST_IMGS=IMGS(:,50001:60000);
LABS=loadMNISTLabels('train-labels.idx1-ubyte');
TRAIN_LABS=LABS(1:50000,:);
TEST_LABS=LABS(50001:60000,:);
%%将识别标签转换为向量形式
%比如7则转换为[0,0,0,0,0,0,0,1,0,0]
%TRAIN_LABS
TEMP=zeros(5003000,10);
for i=1:50000
    TEMP(i,TRAIN_LABS(i,1)+1)=1;
end
TRAIN_LABS=TEMP.';
%TEST_LABS
TEMP=zeros(10000,10);
for i=1:10000
    TEMP(i,TEST_LABS(i,1)+1)=1;
end
TEST_LABS=TEMP.';
%定义网络大小w2
NetSize=[784,30,10];
%使用randn初始化biases和weights
biases_h=randn(30,1);%隐层biases
biases_o=randn(10,1);%输出层biases
weights_ih=randn(30,784);%连接输入层和隐层的权值
weights_ho=randn(10,30);%连接隐层和输出层的权值
%这里定义了out_o是给后面测试集的验证使用的，先分配了空间
out_o=zeros(10,10000);
%%现在我们的网络已经初始化完成，下面采用mini批BP算法来学习
%在这里我们设置每一批的数目为10
step=3;%设置学习率
for j=1:30 %j代表训练次数，暂设置最大训练数为100
    %%下面的FOR循环为对数据集的一次mini批BP算法应用
    %首先随机打乱我们的数据集  
    r=randperm(50000);
    TRAIN_IMGS=TRAIN_IMGS(:,r);
    TRAIN_LABS=TRAIN_LABS(:,r);
    for i=0:4999
        %计算一个mini-batch的输出层神经元的梯度项G，即把该mini-batch的每一项求得的G相加
        G=zeros(10,1);%分配空间
        E=zeros(30,1);%分配空间
        %给各个参数的delta值分配空间，delta即变化量
        weights_ho_delta=zeros(10,30);
        biases_o_delta=zeros(10,1);
        weights_ih_delta=zeros(30,784);
        biases_h_delta=zeros(30,1);
        for l=(i*10+1):((i+1)*10)
            %首先由一个样本获得输出
            tem_i=TRAIN_IMGS(:,l);
            out_h= output(tem_i,weights_ih,biases_h);
            tem_o=output(out_h,weights_ho,biases_o);
            tem_labs=TRAIN_LABS(:,l);
            %计算输出层神经元的梯度项G
            G=tem_o.*(1-tem_o).*(tem_labs-tem_o);
            %计算隐层神经元的梯度项E
            E=out_h.*(1-out_h).*((weights_ho.')*G);
            %获得各个参数的delta值，原理是把mini-batch里的每一样本求delta值然后累加，再/10
            weights_ho_delta=weights_ho_delta+(step/10)*(G*(out_h.'));
            biases_o_delta=biases_o_delta-(step/10)*G;
            weights_ih_delta=weights_ih_delta+(step/10)*(E*(tem_i.'));
            biases_h_delta=biases_h_delta-(step/10)*E;
        end
        %%接下来由上面求出的E和G更新参数
            %更新隐层到输出层的权值
            weights_ho=weights_ho+weights_ho_delta;
            %更新输出层的biases
            biases_o=biases_o+biases_o_delta;
            %更新输入层和隐层的权值
            weights_ih=weights_ih+weights_ih_delta;
            %更新隐层的biases
            biases_h=biases_h+biases_h_delta;
    end
    %%每一次迭代运用测试数据集来验证准确率，输出测试集的正确输出数目
    %对于神经网络的输出做如下处理：
    %由测试集的输入通过神经网络获得输出（10*10000的矩阵）,取每一列的最大值为1，其他为0。
    %比如某一列为[0.1,0.3,0.2,0.2,0.5,0.7,0.8,0.1,0,0].'将会被转成[0,0,0,0,0,0,1,0,0,0].'
    for i=1:10000
        %获得一个样本输入的神经网络输出
        tem_i=TEST_IMGS(:,i);
        out_h= output(tem_i,weights_ih,biases_h);
        tem_o=output(out_h,weights_ho,biases_o);
        out_o(:,i)=tem_o;
        %转换（即将最大值所在的位置取1，其他取0）
        tem_row=1;
        tem_max=out_o(tem_row,i);
        for k=1:10
            if out_o(k,i)>tem_max
                tem_row=k;
                tem_max=out_o(tem_row,i);
            end
        end
        for k=1:10
            if k==tem_row
                out_o(k,i)=1;
            else
                out_o(k,i)=0;
            end
        end
    end
    %与标准输出对比，获得准确率
    sum=0;
    for i=1:10000
        if out_o(:,i)==TEST_LABS(:,i)
           sum=sum+1; 
        end
    end
    fprintf('迭代%d: %d/10000\n',j,sum);
end

