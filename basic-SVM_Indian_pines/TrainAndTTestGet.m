function [train_set,train_label,train_index,test_set,test_label,test_index]=TrainAndTestGet(image,image_label,classnum,ratio)
%���ݴ���----��ԭʼ����---->���ѵ�����Ͳ��Լ�
%ԭʼ����Ҫ��
% image   145*145*200     200������������ά��    image_label 145*145��С ��������ǩ 1,2,3,4,5,6��...16 
%classnum �������������Ŀ
%��ȡ�������ݼ� �Ѿ�



label=image_label;
image=permute(image,[3,1,2]);


classnum=16;

%ѡȡ���е� 10%��Ϊѵ������
par=ratio;

train_index=[];
train_set=[];
train_label=[];

test_index=[];
test_set=[];
test_label=[];

for i=1:1:classnum
    index=find(label==i);
    len=length(index);
    
    t_index=randsample(index,round(par*len));
    train_index=[train_index;t_index];
    train_label=[train_label;label(t_index)];
    train_set=[train_set;image(:,t_index)'];
    
    tt_index=setdiff(index,t_index);
    test_index=[test_index;tt_index];
    test_label=[test_label;label(tt_index)];
    test_set=[test_set;image(:,tt_index)'];
    
end




for i=1:length(train_label)
    train_set(i,:)=train_set(i,:)/norm(train_set(i,:));    
end


for i=1:length(test_label)
test_set(i,:)=test_set(i,:)/norm(test_set(i,:));

end

