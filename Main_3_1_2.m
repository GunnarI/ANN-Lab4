clear all; close all; clc

%% Import and Plot original data

importdata('binMNIST');
for i = 1:20
    %original data
    figure(1)
    subplot(4,5,i)
    imshow(reshape(bindata_trn(i,:),28,28)')
end
%% Define constants

epochs = 75;
%hidden_size = [50 75 100 150];
hidden_size = 150;
bindata_err = zeros(size(hidden_size));

%% Train the network

enc.MaxEpochs = epochs;
%enc.EncoderTransferFunction = 'logsig';
%enc.DecoderTransferFunction = 'logsig';
%enc.L2WeightRegularization = 0.004;
%enc.SparsityRegularization = 4;
%enc.SparsityProportion = 0.15;

%Turn data
bindata_trn = bindata_trn';
bindata_tst = bindata_tst';

for h = 1:length(hidden_size)
    autoenc = trainAutoencoder(bindata_trn,hidden_size(h),enc);
    traindata_rec = predict(autoenc, bindata_trn);
    %bindata_err(h) = mean(mean((bindata_trn - bindata_rec).^2));
    
    %Test of the autoencoder
    testdata_rec = predict(autoenc, bindata_tst);
    
    %Plot the images
    digit = 0:9;
    %find targets for each digit
    for k = 1:length(digtargets_tst)
        digidata = bindata_tst(:,find(digtargets_tst(i) == digit,1));
        recdata = testdata_rec(:,find(digtargets_tst(i) == digit,1));
    end
    
    for i = 1:10
        figure(h+1)
        subplot(5,5,i)
        imshow(reshape(digidata(:,i),28,28)')
        subplot(5,5,10+i)
        imshow(reshape(recdata(:,i),28,28)')
        hold on
    end
    hold off
end


%% Plot the weights
%autoenc.EncoderWeights 