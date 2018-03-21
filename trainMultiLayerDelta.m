function [w, v, MMSE] =trainMultiLayerDelta(input,target,eta,epochs)

% input is a N x M vector where N is the number of images and M is the
% number of nodes in the last hidden layer of the pretraining network
% input contains the data representation given by the pretraining network

input = input';
ndata = size(input,2);
nhidden = size(input,1);
noutput = 10;

targetVecs = ones(noutput,ndata)*(-1);
for t = 1:ndata
    targetVecs(target(t)+1,t) = 1; % targetVecs is 10 x ndata matrix
end

%d = mean(sum(input));
w = randn(nhidden,nhidden+1)*(2/sqrt(nhidden+1))-(1/sqrt(nhidden+1));
v = randn(noutput,nhidden+1)*(2/sqrt(nhidden+1))-(1/sqrt(nhidden+1));

alpha = 0.9;
dw = zeros(nhidden, nhidden+1);
dv = zeros(noutput, nhidden+1);

MMSE = zeros(1,epochs);
for i = 1:epochs
    for j = 1:ndata
        hin = w * [input(:,j); 1];
        hout = [2 ./ (1+exp(-hin)) - 1 ; 1];

        oin = v * hout;
        out = 2 ./ (1+exp(-oin))-1;

        delta_o = (out - targetVecs(:,j)) .* ((1+out) .* (1-out)) * 0.5;
        delta_h = (v' * delta_o) .* ((1+hout) .* (1-hout)) * 0.5;
        delta_h = delta_h(1:nhidden, :);

        dw = (dw .* alpha) - (delta_h * [input(:,j); 1]') .* (1-alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
        w = w + dw .* eta;
        v = v + dv .* eta;
    end
    
    hin = w * [input; ones(1, size(input,2))];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1, size(input,2))];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin))-1;
    error = (out - targetVecs).^2;
    MMSE(i) = mean(mean(error));
end

end