function [OA,AA,kappa,CA,Yt] = svm_pred(Ws_all,Wt_all,X_src,X_tar,Train_Lab, Test_Lab,nnClass)
S_v = [];
for i = 1:nnClass
    idx_s = find(Train_Lab==i);
    idx_t = find(Test_Lab==i);
    % Ws = Ws_all{i};
    % Wt = Wt_all{i};
    S_v(:,idx_s) = Ws_all{i}'* X_src(:,idx_s); %question ����������ʽ��ô�����ۼӵ�  ------------------ Ȼ���԰���Щ�𿪵�S_v�ϳ�һ���ܵ�
    % S_v = [S_v, Ws_all{i}'* X_src(:,idx_s)];
    T_v(:,idx_t) = Wt_all{i}'* X_tar(:,idx_t);
end
[Yt,acc] = target_svm_pred(S_v, T_v, Train_Lab, Test_Lab);
[OA,AA,kappa,CA] = eval_confusion(Test_Lab, Yt);