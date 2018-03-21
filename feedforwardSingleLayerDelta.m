function digit = feedforwardSingleLayerDelta(data,w)

data = data'; %ones(size(data,1),1)]';

oin = w * [data; ones(1, size(data,2))];
out = 2 ./ (1+exp(-oin))-1;

digit = ones(size(out,2),1)*(-1);
for i = 1:length(digit)
    %out(:,i) == max(out(:,i));
    num = find(out(:,i) == max(out(:,i)))-1;
    if length(num) == 1
        digit(i) = num;
    end
end
end