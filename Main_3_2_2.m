clear all; close all; clc

%% Import original data
importdata('binMNIST');

%% Define constants
epochs = 10;
hidden_size = 10;
hidden_layers = 1;

%% Train the network

%network features
enc.MaxEpochs = epochs;
%enc.EncoderTransferFunction = 'logsig';
enc.DecoderTransferFunction = 'purelin';
%enc.L2WeightRegularization = 0.004;
%enc.SparsityRegularization = 4;
%enc.SparsityProportion = 0.15;

%Turn data
bindata_trn = bindata_trn';
bindata_tst = bindata_tst';
digtargets_trn = digtargets_trn';
digtargets_tst = digtargets_tst';

%set variable for error measurement
bindata_err = zeros(size(hidden_size));

%train network with i hidden layers
autoenc1 = trainAutoencoder(bindata_trn,hidden_size,enc);
features1 = encode(autoenc1,bindata_trn);

autoenc2 = trainAutoencoder(features1,hidden_size,enc);
features2 = encode(autoenc2,features1);

autoenc3 = trainAutoencoder(features2,hidden_size,enc);
features3 = encode(autoenc3,features2);

%classification layer
softnet = trainSoftmaxLayer(features3,digtargets_trn,'LossFunction','crossentropy');

%stack the different layers
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);

%train the full network with targets
deepnet = train(deepnet,bindata_trn,digtargets_trn);

%% Testing the trained deep autoencoder

testdata_class = deepnet(bindata_tst);
plotconfusion(digtargets_tst,testdata_class);
