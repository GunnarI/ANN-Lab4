function result = convertDigits(digits)

result = zeros(10, length(digits));

for i = 1:length(digits)
    result(digits(i),i) = 1;
end

end