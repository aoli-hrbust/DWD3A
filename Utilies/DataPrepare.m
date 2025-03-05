function [Source_data,Target_data,Source_label,Target_label,temp_train,im_gt] = DataPrepare(Source_img,Source_gt,Target_img,Target_gt,opt)
% ================== Source Domain ======================= %
% load(Source)
% Source_img = eval(Source);
if opt.pca_flag
   [PC, ~] = PCA_img(Source_img,opt.dim);
   im_2d = ToVector(PC)';
else
   band = min(size(Source_img,3),size(Target_img,3));
   im_2d = ToVector(Source_img(:,:,1:band))';
end
im_gt = Source_gt;
[i_row, i_col] = size(im_gt);
im_gt_1d = reshape(im_gt,1,i_row*i_col);
C = opt.nnClass;
% index_map = reshape(1:length(im_gt_1d),[i_row,i_col]);
%======================================================
index = [];label = [];
num_class = [];
for i = 1:1:C
    index_t =  find(im_gt_1d == i);
    index = [index index_t];
    label_t = ones(1,length(index_t))*i;
    label = [label label_t];
    num_class_t = length(index_t);
    num_class = [num_class num_class_t];
end
% num_tr = [1048,12,140,83,20,50,73,5,50,5,100,250,60,20,130,40,10];  %10%
sel_c = opt.sel_source;
nnClass = length(sel_c);
train_class_num = opt.source_class_num;
num_tr = train_class_num*ones(1,C);
CTrain = num_tr(sel_c);
num_tr = [sum(CTrain) num_tr];
D = [];D_label = [];tt_data = [];tt_label = [];
tt_index = [];temp_train = [];temp_test = [];

% for i = 1:1:C
for i = 1:1:nnClass  
    label_c = find(label == sel_c(i));
    random_index = label_c(randperm(length(label_c)));
    temp = index(random_index(1:num_tr(sel_c(i)+1)));
    temp_train = [temp_train temp];
    D_i = im_2d(:,temp);
    D = [D D_i];
    D_label_i = ones(1,length(temp))*i;
    D_label = [D_label D_label_i];
    temp = index(random_index(num_tr(sel_c(i)+1)+1:end));
    tt_data_i = im_2d(:,temp);
    temp_test = [temp_test temp];
    tt_data = [tt_data tt_data_i];
    tt_label_i = ones(1,length(temp))*i;
    tt_label = [tt_label tt_label_i];
    tt_index = [tt_index temp];
end
% data_all = [D,tt_data ];
% labels = [D_label,tt_label];
if (~opt.norm_flag)
   Source_data = D;
else
   Source_data = D./repmat(sqrt(sum(D.*D)),[size(D,1) 1]);
end
Source_label = D_label;
% label_result = zeros(size(tt_label));
% Source_train_data_ori = data_all(:, (1:num_tr(1)))';
% Source_test_data_ori = data_all(:, (num_tr(1)+1:end))';
disp('----------Source Data Preparing Finish------------');
% ================ Target Domain ==================== %
if opt.pca_flag
   [PC, ~] = PCA_img(Target_img,opt.dim);
   im_2d = ToVector(PC)';
else
   band = min(size(Source_img,3),size(Target_img,3));
   im_2d = ToVector(Target_img(:,:,1:band))';
end
im_gt = Target_gt;
[i_row, i_col] = size(im_gt);
im_gt_1d = reshape(im_gt,1,i_row*i_col);
% index_map = reshape(1:length(im_gt_1d),[i_row,i_col]);
%=====================================================
index = [];label = [];
num_class = [];
for i = 1:1:C
    index_t =  find(im_gt_1d == i);
    index = [index index_t];
    label_t = ones(1,length(index_t))*i;
    label = [label label_t];
    num_class_t = length(index_t);
    num_class = [num_class num_class_t];
end
% num_tr = [1048,12,140,83,20,50,73,5,50,5,100,250,60,20,130,40,10];  %10%
sel_c = opt.sel_target;
nnClass = length(sel_c);
train_rate_class = opt.target_class_rate;
num_tr = floor(num_class*train_rate_class);
CTest = num_tr(sel_c);
num_tr = [sum(CTest),num_tr];
D = [];D_label = [];tt_data = [];tt_label = [];
tt_index = [];temp_train = [];temp_test = [];
% for i = 1:1:C
for i = 1:1:nnClass
    label_c = find(label == sel_c(i));
    random_index = label_c(randperm(length(label_c)));
    temp = index(random_index(1:num_tr(sel_c(i)+1)));
    temp_train = [temp_train temp];
    D_i = im_2d(:,temp);
    D = [D D_i];
    D_label_i = ones(1,length(temp))*i;
    D_label = [D_label D_label_i];
    temp = index(random_index(num_tr(sel_c(i)+1)+1:end));
    tt_data_i = im_2d(:,temp);
    temp_test = [temp_test temp];
    tt_data = [tt_data tt_data_i];
    tt_label_i = ones(1,length(temp))*i;
    tt_label = [tt_label tt_label_i];
    tt_index = [tt_index temp];
end
% data_all = [D,tt_data ];
% labels = [D_label,tt_label];
if (~opt.norm_flag)
   Target_data = D;
else
   Target_data = D./repmat(sqrt(sum(D.*D)),[size(D,1) 1]);
end
Target_label = D_label;
% label_result = zeros(size(tt_label));
% Target_train_data_ori = data_all(:, (1:num_tr(1)))';
% Target_test_data_ori = data_all(:, (num_tr(1)+1:end))';
disp('----------Target Data Preparing Finish------------');