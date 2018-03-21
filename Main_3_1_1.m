importdata('binMNIST');

%plotDistributionHist;

inputnum = size(bindata_trn, 2);
outputnum = [50 75 100 150];

epochs = 20;

opt.Verbose = false;
opt.CalcError = true;
opt.MaxIter = epochs;
opt.StepRatio = 0.1;

figSMSE = zeros(length(outputnum), epochs);
figMMSE = zeros(length(outputnum), epochs);
timeVec = zeros(1,length(outputnum));
for h_nodes = 1:length(outputnum)
    tic
    rbm = randRBM(inputnum, outputnum(h_nodes));
    [rbm, figSMSE(h_nodes,:), figMMSE(h_nodes,:)] = pretrainRBM(rbm, bindata_trn, opt);
    timeVec(h_nodes) = toc;

    if h_nodes == 1 || h_nodes == 3
        numHidden = outputnum(h_nodes);
        numPlotLines = numHidden/10;
        
        figure
        for i = 1:numHidden
            subplot(numPlotLines,10,i)
            imshow(reshape(rbm.W(:,i),28,28)');
            title(['Node ' num2str(i)])
        end
    end
end

figure
plot(figSMSE')
legend('50 hidden nodes','75 hidden nodes',...
    '100 hidden nodes','150 hidden nodes')
title('RBM Training Error: Sum of MSE for each image')
xlabel('Epochs')
ylabel('Error')

figure
plot(figMMSE')
legend('50 hidden nodes','75 hidden nodes',...
    '100 hidden nodes','150 hidden nodes')
title('RBM Training Error: Mean of MSE for each image')
xlabel('Epochs')
ylabel('Error')

figure
plotDigComparison(bindata_tst, digtargets_tst, rbm)