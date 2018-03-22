clear all; close all; clc

%% Import and Plot original data

importdata('binMNIST');
% for i = 1:20
%     %original data
%     figure(1)
%     subplot(4,5,i)
%     imshow(reshape(bindata_trn(i,:),28,28)')
% end
%% Define constants

epochs = 75;
hidden_size = [50 75 100 150];
%hidden_size = 100;
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
    weights = autoenc.EncoderWeights;
    traindata_rec = predict(autoenc, bindata_trn);
    bindata_err(h) = mean(mean((bindata_trn - traindata_rec).^2));
    
    %Test of the autoencoder
    testdata_rec = predict(autoenc, bindata_tst);
    
    %Plot the images
    %find targets for each digit
%     digdata = zeros(784,10);
%     recdata = zeros(784,10);
%     plotvec = [1 6 11 16 21 4 9 14 19 24];
%     for d = 1:10                             %digits
%         k = find(digtargets_tst == d-1,1);   %indices for every digit
%         digdata(:,d) = bindata_tst(:,k);
%         recdata(:,d) = testdata_rec(:,k);
%         
%         figure(h+1)
%         subplot(5,5,plotvec(d))
%         imshow(reshape(digdata(:,d),28,28)')
%         subplot(5,5,plotvec(d)+1)
%         imshow(reshape(recdata(:,d),28,28)')
%     end
end


%% Plot the weights
% for i = 1:hidden_size
%     figure(3)
%     subplot(10,10,i)
%     imshow(reshape(weights(i,:),28,28)')
%     title(['Node ',num2str(i)],'FontSize',6)
% end
