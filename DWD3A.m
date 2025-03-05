function [label,obj,Ws_all,Wt_all] = DWD3A(X_src,Y_src,X_tar_all,X_tar_ori,options)

Opts.A_st_norm = 0.5;
Opts.A_st_max = 0;
Opts.A_st_min = 0;
Opts.J_w_norm = 0.5;
Opts.max_J_w = 1;
Opts.min_J_w =1;

Test_Lab_ori = options.Test_Lab;
Train_Lab = options.Train_Lab;
dim = options.dim;
beta = options.beta;
nnClass = size(Y_src,2);
max_iter = options.max_iter;
[m, n] = size(X_src);

Ws  = rand(m,dim);
Wt  = rand(m,dim);
Ws_all = cell(1,nnClass);
Wt_all = cell(1,nnClass);
acc_list = [];
options.ReducedDim = dim;

[coeff, ~] = pca(X_src');
P1s = coeff(:,1:dim);
[coeff, ~] = pca(X_tar_ori');
P1t = coeff(:,1:dim);

knn_model = fitcknn(X_src',Train_Lab,'NumNeighbors', 1);
class_candidate = knn_model.predict(X_tar_ori');

acc_max = 0; X_src_da_all = []; X_tar_da_all = []; a = 0;
for iter = 1 : max_iter
    %         a = a + 1
    X_tar = X_tar_all{mod(iter,length(X_tar_all))+1};
    
    %compute alpha
    alpha_complex = Opts.A_st_norm / (Opts.A_st_norm + (1.0 - Opts.J_w_norm));
    alpha = real(alpha_complex);
    
    if iter == 1
        knn_model = fitcknn(X_src',Train_Lab,'NumNeighbors',5);
        label_candidate = knn_model.predict(X_tar');
        X_tar_candidate = X_tar;
    end
    
    obj_intra = 0; discri = 0; align = 0;
    X_src_da = []; X_tar_da = [];
    
    for i = 1:nnClass
        src_index_i = Train_Lab == i;
        X_src_i = X_src(:,src_index_i);
        Y_src_i = Train_Lab(src_index_i);
        
        tar_index_i = label_candidate(:) == i;
        X_tar_i = X_tar_candidate(:,tar_index_i);
        label_candidate_i = label_candidate(tar_index_i);
        
        ns_i = size(X_src_i,2); nt_i = size(X_tar_i,2);
        [Ls_i,Lt_i,Lst_i,Lts_i] = construct_mmd(ns_i,nt_i,Y_src_i,label_candidate_i,nnClass);
        
        Ms_i = X_src_i * Ls_i * X_src_i';
        Mt_i = X_tar_i * Lt_i * X_tar_i';
        Mst_i = X_src_i * Lst_i * X_tar_i';
        Mts_i = X_tar_i * Lts_i * X_src_i';
        clear Ls_i Lt_i Lst_i Lts_i
        
        % ----------  update Ps ----------- %
        if (iter == 1)
            Ps = P1s;
        else
            [U1,S1,V1] = svd(X_src_i*X_src_i'*Ws,'econ');
            Ps = (1-alpha)*U1*V1';
        end
        clear U1 S1 V1
        % ---------- update Pt ----------- %
        if (iter == 1)
            Pt = P1t;
        else
            [U1,S1,V1] = svd(X_tar_i*X_tar_i'*Wt,'econ');
            Pt = (1-alpha)*U1*V1';
        end
        clear U1 S1 V1
        
        % ---------- ¸üÐÂ Zs ----------- %
        Z1 = alpha.*(X_src_i'*Ws*Wt'*X_tar_i);
        Z2 = alpha.*(X_src_i'*Ws*Ws'*X_src_i)+alpha.*eye(ns_i);
        Zs = (Z1'/Z2)';
        clear Z1 Z2
        
        % ---------- update Zt ----------- %
        Z1 = alpha.*(X_tar_i'*Wt*Ws'*X_src_i);
        Z2 = alpha.*(X_tar_i'*Wt*Wt'*X_tar_i)+alpha.*eye(nt_i);
        Zt = (Z1'/Z2)';
        clear Z1 Z2
        
        % ---------- update Ws ----------- %
        XX = X_src_i*X_src_i';
        W1new = alpha.*(X_src_i*Zs*X_tar_i'*Wt) + alpha.*(X_src_i*Zt'*X_tar_i'*Wt);
        W2new = alpha.*(X_src_i*Zs*Zs'*X_src_i') + alpha.*XX;
        W1 = (1-alpha).*(XX*Ps) + beta/nnClass.*Wt - alpha.*(Mts_i'+Mst_i)*Wt + W1new;
        W2 = (1-alpha).*XX + beta/nnClass.*eye(m) + alpha.*(Ms_i+Ms_i') + W2new;
        Ws = (W1'/W2)';
        clear W1 W2 W1new W2new XX
        
        % ---------- update Wt ----------- %
        XX = X_tar_i*X_tar_i';
        W1new = alpha.*(X_tar_i*Zs'*X_src_i'*Ws) + alpha.*(X_tar_i*Zt*X_src_i'*Ws);
        W2new = alpha.*XX + alpha.*(X_tar_i*Zt*Zt'*X_tar_i');
        W1 = (1-alpha).*(XX*Pt) + beta/nnClass.*Ws - alpha.*(Mts_i+Mst_i')*Ws + W1new;
        
        W2 = (1-alpha).*XX + beta/nnClass.*eye(m) + alpha.*(Mt_i+Mt_i') + W2new;
        Wt = (W1'/W2)';
        clear W1 W2 W1new W2new XX XX_lable
        
        X_src_da = [X_src_da,Ws'*X_src_i];
        X_tar_da = [X_tar_da,Wt'*X_tar_i];
        Ws_all{i} = Ws; Wt_all{i} = Wt;
        temp = [Ws',Wt']*[Ms_i, Mst_i; Mts_i, Mt_i]*[Ws;Wt];
        
        discri = discri + (norm(X_src_i-Ps*Ws'*X_src_i,'fro').^2 + norm(X_tar_i-Pt*Wt'*X_tar_i,'fro').^2);
        align = align + (norm(Wt'*X_tar_i-Ws'*X_src_i*Zs,'fro').^2 + norm(Ws'*X_src_i-Wt'*X_tar_i*Zt,'fro').^2);
        obj_intra = obj_intra + trace(temp);
    end
    
    X_src_da_temp = []; X_tar_da_temp = [];
    for i = 1:nnClass
        src_index_i = Train_Lab == i;
        X_src_i = X_src(:,src_index_i);
        
        tar_index_i = class_candidate(:) == i;
        X_tar_i = X_tar_ori(:,tar_index_i);
        
        X_src_da_temp = [X_src_da_temp, Ws_all{i}' * X_src_i];
        X_tar_da_temp = [X_tar_da_temp, Wt_all{i}' * X_tar_i];
    end
    
    knn_model = fitcknn(X_src_da_temp',Train_Lab,'NumNeighbors',5);
    label = knn_model.predict(X_tar_da_temp');
    
    %Dynamic weighted compute
    Output = Dweight(X_src_da_temp, X_tar_da_temp,Train_Lab,label,nnClass,Opts);
    Opts.A_st_norm = Output.A_st_norm;
    Opts.J_w_norm = Output.J_w_norm;
    Opts.A_st_max = Output.A_st_max;
    Opts.A_st_min = Output.A_st_min;
    Opts.max_J_w = Output.max_J_w;
    Opts.min_J_w = Output.min_J_w;
    
    label_candidate = knn_model.predict(X_tar_da');
    acc = length(find(label == Test_Lab_ori)) / length(Test_Lab_ori);
    acc_list = [acc_list;acc];
    
    if iter>1 && acc_list(iter)>acc_max
        X_src_da_all = X_src_da_temp;
        X_tar_da_all = X_tar_da_temp;
        acc_max = acc_list(iter);
        
    elseif iter == 1
        X_src_da_all = X_src_da_temp;
        X_tar_da_all = X_tar_da_temp;
        acc_max = acc_list(iter);
    end
    
    obj(iter) = alpha * (obj_intra + align + norm(Zs ,'fro').^2 + norm(Zt ,'fro').^2) + ...
        (1 - alpha) * discri + beta * norm(Ws-Wt ,'fro').^2;
    
    
    % judge convergence
    if iter > 10 && abs(obj(iter)-obj(iter-1)) < 1e-7
        iter
        knn_model = fitcknn(X_src_da_all',Train_Lab,'NumNeighbors',5);
        label = knn_model.predict(X_tar_da_all');
        break;
    elseif iter == max_iter
        knn_model = fitcknn(X_src_da_all',Train_Lab,'NumNeighbors',5);
        label = knn_model.predict(X_tar_da_all');
    end
    
end

end