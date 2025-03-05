function [Yt,acc] = target_svm_pred(Xs, Xt, Xs_label, Xt_label)
C = [0.001 0.01 0.1 1.0 10 100 1000 10000];  
for chsvm = 1 :length(C)
    tmd = ['-s 3 -c ' num2str(C(chsvm)) ' -B 1 -q'];
    model(chsvm) = train(Xs_label, sparse(double(Xs')),tmd);
    [~,acc, ~] = predict(Xt_label, sparse(double(Xt')), model(chsvm), '-q');
    acc1(chsvm)=acc(1);
end	
[acc,bestsvm_id]=max(acc1);
fprintf(' svm acc=%2.2f %%\n',acc);
model=model(bestsvm_id);
c=C(bestsvm_id);
score = model.w * [Xt; ones(1, size(Xt, 2))];

th = mean(score, 2)';
[confidence, C] = max(score, [], 1);
Yt = C';
end