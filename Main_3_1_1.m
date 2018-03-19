importdata('binMNIST');

inputnum = size(bindata_trn, 2);
outputnum = [50 75 100 150];

epochs = 20;

opt.Verbose = false;
opt.CalcError = true;
opt.MaxIter = epochs;
opt.StepRatio = 0.1;

for h_nodes = 1:length(outputnum)
    rbm = randRBM(inputnum, outputnum(h_nodes));
    [rbm, error1, error2] = pretrainRBM(rbm, bindata_trn, opt);

    figure
    plot(error1); hold on; plot(error2);
end

%imshow(reshape(rbm.W(:,6)+rbm.b(6),28,28)')
%in = round(rand(outputnum(3),1));
%H = v2h(rbm, bindata_tst);
%V = h2v(rbm, H);

%imshow(reshape(V(1,:)', 28, 28))