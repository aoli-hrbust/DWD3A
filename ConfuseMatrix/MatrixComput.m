function MatrixComput(gnd_train,predLabel,type)
%type 1: Show percentage; type 0: Show sample number;
% A=[1 2 1 1 2 1 2 3 3 3 3 4 4 1 4];  % predicted label 
% num=ones(1,9)*100;  % total number of test samples in each class
name=cell(1,9);  
name{1}='1';name{2}='2';name{3}='3';name{4}='4';  
name{5}='5';name{6}='6';name{7}='7';name{8}='8'; 
name{9}='9';
figure,my_confusion_matrix(double(gnd_train),predLabel,type); 
% draw_cm(confusion_matrix,name,9); 