importdata('binMNIST');

%plotDistributionHist;

runPreTraining = false;

inputnum = size(bindata_trn, 2);
outputnum = [50 75 100 150];

epochs = 20;

numLayers = [1 2 3];
layerDims = cell(1,3);
layerDims{1} = [inputnum 150];
layerDims{2} = [inputnum 150 125];
layerDims{3} = [inputnum 150 125 100];

opt.Verbose = false;
opt.CalcError = true;
opt.MaxIter = epochs;
opt.StepRatio = 0.1;

figSMSE = zeros(length(outputnum), epochs);
figMMSE = zeros(length(outputnum), epochs);

if runPreTraining
    dbn = cell(length(numLayers), 1);
end
W = cell(length(numLayers), 1);
V = cell(length(numLayers), 1);
MMSE = cell(length(numLayers), 1);

for n_layer = 1:length(numLayers)
    if runPreTraining
        opt.LayerNum = numLayers(n_layer);
        %layerDim = [inputnum ones(1,opt.LayerNum)*outputnum(h_nodes)];

        dbn_temp = randDBN(layerDims{n_layer});
        dbn{n_layer} = pretrainDBN(dbn_temp, bindata_trn, opt);
    end
        
    pretrainOutput = v2h(dbn{n_layer}, bindata_trn);

    eta = 0.01;
    epochs = 300;
    weights = randn(10,size(pretrainOutput,2))*2; % Weights
    %weights = weights/norm(weights);

%     W{n_layer, h_nodes} = trainSingleLayerDelta(pretrainOutput,...
%        digtargets_trn, eta, epochs);
    [W{n_layer}, V{n_layer}, MMSE{n_layer}] = ...
        trainMultiLayerDelta(pretrainOutput,digtargets_trn,eta,epochs);
end

pretrainOutput = v2h(dbn{3}, bindata_trn);
%output_digits = feedforwardSingleLayerDelta(pretrainOutput, W{3,4});
output_digits = feedforwardMultiLayerDelta(pretrainOutput, W{3}, V{3});