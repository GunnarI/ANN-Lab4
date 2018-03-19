v_trn = hist(digtargets_trn, [0:1:9]);
bar(0:9, v_trn./sum(v_trn))
hold on
plot(-1:10, 0.1*ones(1,12))
title('Digit distribution for training set')
xlabel('Digit')
ylabel('Percentage')

figure
v_tst = hist(digtargets_tst, 0:1:9);
bar(0:9, v_tst./sum(v_tst))
hold on
plot(-1:10, 0.1*ones(1,12))
title('Digit distribution for test set')
xlabel('Digit')
ylabel('Percentage')