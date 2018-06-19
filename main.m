%��ȡ���ݲ������ݲ�ֳ�ѵ�����ݺͲ�������
%ǰ50000��Ϊѵ�����ݣ���10000��Ϊ��������
IMGS=loadMNISTImages('train-images.idx3-ubyte');
TRAIN_IMGS=IMGS(:,1:50000);
TEST_IMGS=IMGS(:,50001:60000);
LABS=loadMNISTLabels('train-labels.idx1-ubyte');
TRAIN_LABS=LABS(1:50000,:);
TEST_LABS=LABS(50001:60000,:);
%%��ʶ���ǩת��Ϊ������ʽ
%����7��ת��Ϊ[0,0,0,0,0,0,0,1,0,0]
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
%���������Сw2
NetSize=[784,30,10];
%ʹ��randn��ʼ��biases��weights
biases_h=randn(30,1);%����biases
biases_o=randn(10,1);%�����biases
weights_ih=randn(30,784);%���������������Ȩֵ
weights_ho=randn(10,30);%���������������Ȩֵ
%���ﶨ����out_o�Ǹ�������Լ�����֤ʹ�õģ��ȷ����˿ռ�
out_o=zeros(10,10000);
%%�������ǵ������Ѿ���ʼ����ɣ��������mini��BP�㷨��ѧϰ
%��������������ÿһ������ĿΪ10
step=3;%����ѧϰ��
for j=1:30 %j����ѵ�����������������ѵ����Ϊ100
    %%�����FORѭ��Ϊ�����ݼ���һ��mini��BP�㷨Ӧ��
    %��������������ǵ����ݼ�  
    r=randperm(50000);
    TRAIN_IMGS=TRAIN_IMGS(:,r);
    TRAIN_LABS=TRAIN_LABS(:,r);
    for i=0:4999
        %����һ��mini-batch���������Ԫ���ݶ���G�����Ѹ�mini-batch��ÿһ����õ�G���
        G=zeros(10,1);%����ռ�
        E=zeros(30,1);%����ռ�
        %������������deltaֵ����ռ䣬delta���仯��
        weights_ho_delta=zeros(10,30);
        biases_o_delta=zeros(10,1);
        weights_ih_delta=zeros(30,784);
        biases_h_delta=zeros(30,1);
        for l=(i*10+1):((i+1)*10)
            %������һ������������
            tem_i=TRAIN_IMGS(:,l);
            out_h= output(tem_i,weights_ih,biases_h);
            tem_o=output(out_h,weights_ho,biases_o);
            tem_labs=TRAIN_LABS(:,l);
            %�����������Ԫ���ݶ���G
            G=tem_o.*(1-tem_o).*(tem_labs-tem_o);
            %����������Ԫ���ݶ���E
            E=out_h.*(1-out_h).*((weights_ho.')*G);
            %��ø���������deltaֵ��ԭ���ǰ�mini-batch���ÿһ������deltaֵȻ���ۼӣ���/10
            weights_ho_delta=weights_ho_delta+(step/10)*(G*(out_h.'));
            biases_o_delta=biases_o_delta-(step/10)*G;
            weights_ih_delta=weights_ih_delta+(step/10)*(E*(tem_i.'));
            biases_h_delta=biases_h_delta-(step/10)*E;
        end
        %%�����������������E��G���²���
            %�������㵽������Ȩֵ
            weights_ho=weights_ho+weights_ho_delta;
            %����������biases
            biases_o=biases_o+biases_o_delta;
            %���������������Ȩֵ
            weights_ih=weights_ih+weights_ih_delta;
            %���������biases
            biases_h=biases_h+biases_h_delta;
    end
    %%ÿһ�ε������ò������ݼ�����֤׼ȷ�ʣ�������Լ�����ȷ�����Ŀ
    %�������������������´���
    %�ɲ��Լ�������ͨ���������������10*10000�ľ���,ȡÿһ�е����ֵΪ1������Ϊ0��
    %����ĳһ��Ϊ[0.1,0.3,0.2,0.2,0.5,0.7,0.8,0.1,0,0].'���ᱻת��[0,0,0,0,0,0,1,0,0,0].'
    for i=1:10000
        %���һ��������������������
        tem_i=TEST_IMGS(:,i);
        out_h= output(tem_i,weights_ih,biases_h);
        tem_o=output(out_h,weights_ho,biases_o);
        out_o(:,i)=tem_o;
        %ת�����������ֵ���ڵ�λ��ȡ1������ȡ0��
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
    %���׼����Աȣ����׼ȷ��
    sum=0;
    for i=1:10000
        if out_o(:,i)==TEST_LABS(:,i)
           sum=sum+1; 
        end
    end
    fprintf('����%d: %d/10000\n',j,sum);
end

