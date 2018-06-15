 load matlab
 label_data = GSE87304seriesmatrix;
 find_array = [];
for i=0:1:4
     index= find(label_data(1,:)==i);
    find_array =[find_array,label_data(:,index(10))];
end
find_array = find_array(2:40001,:);
show_3D = reshape(find_array,200,200,[]);
%find_array = [x1(:,113),x1(:,166),x1(:,199),x1(:,16),x1(:,8)];
%show_3D = reshape(find_array,14,14,[]);
for j=1:1:5
    subplot(2,3,j);
    imagesc(show_3D(:,:,j));
end