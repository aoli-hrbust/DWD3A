clear all; close all; clc;
addpath('liblinear-2.1/matlab');
addpath('datasets');addpath('Utilies');addpath('ConfuseMatrix');

Source = 'paviaU7'; Source_gt = 'paviaU_7gt';
Target = 'paviaC7'; Target_gt = 'paviaC_7gt';
load(Source);S_img = eval('ori_data');
load(Source_gt);S_gt = eval('map');
load(Target);T_img = eval('ori_data');
load(Target_gt);T_gt = eval('map');

% Optional parameters for Pavia
opt.nnClass = size(1 : max(unique(S_gt)), 2);
opt.dim = 60;
opt.pca_flag = 0;
opt.norm_flag = 0;
opt.sel_source = [1 2 3 4 5 6 7];
opt.sel_target = [1 2 3 4 5 6 7];
opt.source_class_num = 200;
opt.target_class_rate = 0.1;


% DataPrepare
[i_row, i_col] = size(T_gt);
[DataTrain,DataTest,CTrain,CTest,~,im_gt_1d] = myDataPrepare(S_img,S_gt,T_img,T_gt,opt);

%========== Transfer Learning ================%
disp('---------Transfer Learning Beginning---------')
nnClass = length(CTrain);
Train_Lab = [];
Test_Lab = [];
class_acc = [];
for j = 1:nnClass 
   Train_Lab = [Train_Lab;j*ones(CTrain(j),1)];
   Test_Lab = [Test_Lab;j*ones(CTest(j),1)];
end
label = unique(Train_Lab);
Y_src = double(bsxfun(@eq, Train_Lab, label'));
Y_tar = double(bsxfun(@eq, Test_Lab, label'));
Train_Ma = DataTrain'; 
X_src = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma = DataTest';
X_tar = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]); 

% Divide data an label into 10 cells
num = 10;
X_src_all = cell(1,num); Train_Lab_all = cell(1,num);
X_tar_all = cell(1,num); Test_Lab_all = cell(1,num);

%Filling cells
a = 0; b = 0;
for j = 1:length(CTest) 
    b = b + CTest(j);
    for i = 1:num
        temp = floor(CTest(j)*size(DataTest,1)/num/size(DataTest,1));
        if i == num
            X_tar_all{i} = [X_tar_all{i},X_tar(:,a+1:b)];
            Test_Lab_all{i} = [Test_Lab_all{i};Test_Lab(a+1:b)];
        else
            X_tar_all{i} = [X_tar_all{i},X_tar(:,a+1:a+temp)];
            Test_Lab_all{i} = [Test_Lab_all{i};Test_Lab(a+1:a+temp)];
        end
        a = a + temp;
    end
    a = b;
end
a = 0; b = 0;
for j = 1:length(CTrain) 
    for i = 1:num
        temp = floor(CTrain(j)*size(DataTrain,1)/num/size(DataTrain,1));
        X_src_all{i} = [X_src_all{i},X_src(:,a+1:a+temp)];
        Train_Lab_all{i} = [Train_Lab_all{i};Train_Lab(a+1:a+temp)];
        a = a + temp;
    end
    b = b + CTrain(j);
    a = b;
end

%Train parameters
% beta = [1e-5 1e-4 1e-3 1e-2 1e-1 1e0 5e0 1e1 1e2 1e3 5e2 ];
beta = [1e0 1e0 1e0 1e0 1e0 5e0];
opts.Train_Lab = Train_Lab;
opts.Test_Lab_all = Test_Lab_all;
opts.Test_Lab = Test_Lab;
opts.max_iter = 200;
opts.dim = 20;
iter = 0;

class_indices = cell(length(opt.sel_target), 1);
class_index_split = cell(length(opt.sel_target), 1);
for j = 1:length(opt.sel_target)
    class_index = find(im_gt_1d == opt.sel_target(j));
    class_index_split{j} = class_index;
    class_indices{j} = class_index(1:CTest(j));
end

for i = 1:length(beta)
        iter = iter + 1
        opts.beta = beta(i);
        [class,obj,Ws{iter},Wt{iter}] = DWD3A(X_src,Y_src,X_tar_all,X_tar,opts);
        [OA,AA,kappa,CA,Yt] = svm_pred(Ws{iter},Wt{iter},X_src,X_tar,Train_Lab,class,nnClass);
        result_OA(i) = OA
        result_AA(i) = AA;
        result_Kappa(i) = kappa
end

disp('---------Transfer Learning END---------')