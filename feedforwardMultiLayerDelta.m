function digit = feedforwardMultiLayerDelta(data,w,v)

data = data'; %ones(size(data,1),1)]';

hin = w * [data; ones(1, size(data,2))];
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1, size(data,2))];
oin = v * hout;
out = 2 ./ (1+exp(-oin))-1;

digit = ones(size(out,2),1)*(-1);
for i = 1:length(digit)
    num = find(out(:,i) == max(out(:,i)))-1;
    if length(num) == 1
        digit(i) = num;
    end
end
end