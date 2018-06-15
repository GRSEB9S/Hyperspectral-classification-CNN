function [x_tr,x_vl,x_ts,y_tr,y_vl,y_ts] = STN_India_Datapreprocessing(Proportional_vector)
%CNN_HSIDATEPROCESS Summary of this function goes here
%   Detailed explanation goes here
load Indian_pines_corrected
load Indian_pines_gt
data_3D = indian_pines_corrected;
label_3D = indian_pines_gt; 
[ m,n,d] = size(data_3D);
label_number = max(max(label_3D));
d1 = (floor(sqrt(d)))^2;
label_list = 1:label_number;
one = zeros(1,label_number);
for i= 1:1:label_number
    one(i)=length(find(label_3D==label_list(i)));
end
data_2D = hyperConvert2d(data_3D(:,:,1:d1));
%%                πÈ“ªªØ
data_2D = hyperNormalize(data_2D); % 0 ~ 1
%data_2D = mapminmax(data_2D);      %-1 ~ 1
%%             
label_2D = reshape(label_3D,m*n,1);
label_data = [label_2D,data_2D'];
data_label_order_train = [];
data_label_order_val = [];
data_label_order_test = [];
for i=1:1:label_number
    row_index = label_data(:,1) == label_list(i);
    data_label_mid = label_data(row_index,:);
    train_one = floor(one(i)*Proportional_vector(1)/sum(Proportional_vector));
    val_one = floor(one(i)*Proportional_vector(2)/sum(Proportional_vector));
       
    RandIndex = randperm(one(i)); 
    data_label_mid = data_label_mid( RandIndex,: );
    mid_train = data_label_mid(1:train_one,:);
    mid_val= data_label_mid(train_one+1:train_one+val_one,:);
    mid_test= data_label_mid(train_one+val_one+1:one(i),:);
    data_label_order_train = [data_label_order_train;mid_train];
    data_label_order_val = [data_label_order_val;mid_val];
    data_label_order_test = [data_label_order_test;mid_test];    
end
RandIndex = randperm( length( data_label_order_train ) ); 
data_label_random_train = data_label_order_train( RandIndex,: );
RandIndex = randperm( length( data_label_order_test ) );
data_label_random_test = data_label_order_test( RandIndex,: );
RandIndex = randperm( length( data_label_order_val ) );
data_label_random_val = data_label_order_val( RandIndex,: );
train_num = size(data_label_order_train,1);
val_num = size(data_label_order_val,1);
test_num = size(data_label_order_test,1);
train_data = (data_label_random_train(:,2:d1+1))';
y_tr = data_label_random_train(:,1)';
val_data = (data_label_random_val(:,2:d1+1))';
y_vl = data_label_random_val(:,1)';
test_data = (data_label_random_test(:,2:d1+1))';
y_ts = data_label_random_test(:,1)';
x_tr = reshape(train_data,sqrt(d1),sqrt(d1),1,train_num);
x_vl = reshape(val_data,sqrt(d1),sqrt(d1),1,val_num);
x_ts = reshape(test_data,sqrt(d1),sqrt(d1),1,test_num);
end
