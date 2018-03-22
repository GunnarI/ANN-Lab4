clear all; close all; clc

%% Import original data
importdata('binMNIST');

%% Define constants and Parameters
epochs = 50;
eta = 0.01;
hidden_size1 = 300;
hidden_size2 = 250;
hidden_size3 = 200;

%network features
enc.MaxEpochs = epochs;
%enc.EncoderTransferFunction = 'logsig';
%enc.DecoderTransferFunction = 'purelin';
%enc.L2WeightRegularization = 0.004;
%enc.SparsityRegularization = 4;
%enc.SparsityProportion = 0.15;

%Turn data
bindata_trn = bindata_trn';
bindata_tst = bindata_tst';
digtargets_trn = digtargets_trn';
digtargets_tst = digtargets_tst';

%% Pretrain the network
tic
%train network with i hidden layers
autoenc1 = trainAutoencoder(bindata_trn,hidden_size1,enc);
features1 = encode(autoenc1,bindata_trn);

autoenc2 = trainAutoencoder(features1,hidden_size2,enc);
features2 = encode(autoenc2,features1);

autoenc3 = trainAutoencoder(features2,hidden_size3,enc);
features3 = encode(autoenc3,features2);

%decode wit the last autoencoder
%features = decode(autoenc3,features3);

%classification layer MATLAB
%training parameters in deep network as in this layer
digtargets_trn = convertDigits(digtargets_trn);
softnet = trainSoftmaxLayer(features3,digtargets_trn,'MaxEpochs',epochs);

%% Training of the network
%stack the different prelearned layers MATLAB
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);

%train the classification layer
%[W, V, MMSE] = trainMultiLayerDelta(features3',digtargets_trn,eta,epochs);

%Train the whole network together MATLAB
%deepnet = train(deepnet,bindata_trn,digtargets_trn);
toc
%% Testing the trained deep autoencoder

%test_features1 = encode(autoenc1,bindata_tst);
%test_features2 = encode(autoenc2,test_features1);
%test_features3 = encode(autoenc3,test_features2);

%decode data with the last autoencoder
%test_features = decode(autoenc3,test_features3);

output_digits = deepnet(bindata_tst);

%output_digits = feedforwardMultiLayerDelta(test_features3', W, V);

%class = sum(output_digits == digtargets_tst');

%convert output
%output_digits = convertDigits(output_digits');
%convert targets
digtargets_tst = convertDigits(digtargets_tst);

%plot classification performance
plotconfusion(digtargets_tst,output_digits);
