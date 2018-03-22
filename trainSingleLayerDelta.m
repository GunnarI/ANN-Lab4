function w =trainSingleLayerDelta(input,target,eta,epochs)

% input is a N x M vector where N is the number of images and M is the
% number of nodes in the last hidden layer, i.e. the number of nodes in the
% input layer here

% weights should be 10 x M 

input = input';
ndata = size(input,2);
nhidden = size(input,1);
noutput = 10;

targetVecs = ones(noutput,ndata)*(-1);
for t = 1:ndata
    targetVecs(target(t)+1,t) = 1; % targetVecs is N x 10 matrix
end

%input = [ones(size(input,1),1), input]; % Add bias column
alpha = 0.9;
w = rand(noutput,nhidden+1)*(2/sqrt(nhidden+1))-(1/sqrt(nhidden+1));
dw = zeros(noutput, nhidden+1);
for i = 1:epochs
    for j = 1:ndata
        oin = w * [input(:,j); 1];
        out = 2 ./ (1+exp(-oin))-1;

        delta_o = (out - targetVecs(:,j)) .* ((1+out) .* (1-out)) * 0.5;

        dw = (dw .* alpha) - (delta_o * [input(:,j); 1]') .* (1-alpha);
        w = w + dw .* eta;
    end
end

end