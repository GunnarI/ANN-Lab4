function plotDigComparison(data, targets, rbm)

digits = 0:9;
for i = 1:length(digits)
    digdata = data(find(targets == digits(i), 1), :);
    if i < 6
        subplot(5,5,i*5-4)
        imshow(reshape(digdata,28,28)')
        subplot(5,5,i*5-3)
        H_dig = v2h(rbm, digdata);
        V_dig = h2v(rbm, H_dig);
        imshow(reshape(V_dig,28,28)')
    else
        subplot(5,5,(i-5)*5-1)
        imshow(reshape(digdata,28,28)')
        subplot(5,5,(i-5)*5)
        H_dig = v2h(rbm, digdata);
        V_dig = h2v(rbm, H_dig);
        imshow(reshape(V_dig,28,28)')
    end
end

end