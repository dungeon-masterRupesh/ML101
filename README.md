# ML101
This part and commit was done by Ubuntu environment of Windows

## ex1 -> linear regression
## ex2 -> logistic regression
## ex3 -> multi classification and predicting value from an already trained neural net
## ex4 -> neural net and back propagation
Implemented back propagation using vectorisation 
```
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2,1), 1) sigmoid(z2)];
%a3 == h
h = sigmoid(a2 * Theta2');
y_logic = zeros(m,size(Theta2,1));
for i = 1:m
    y_logic(i,y(i))=1;
end
J = log(h).*y_logic + log(1-h) .* (1 - y_logic);
J = sum(sum(J));
J = J*(-1/m);
J = J + (lambda/(2*m))*(norm(Theta1,'fro')^2 ...
    + norm(Theta2,'fro')^2 - norm(Theta1(:,1),'fro')^2 - norm(Theta2(:,1),'fro')^2);
%%%%%%%%%%%%%%%%%%%%
d3 = h - y_logic; %each layers delta for all data points
d2 = (d3 * Theta2) .* a2 .* (1 - a2);
% d1 = (d2 * Theta1) .* a1 .* (1 - a1);
Theta2_grad = d3' * a2; %this multiplication of (a*m) X (m*p) is as summation over all m data points
Theta1_grad = d2(:,2:end)' * a1;
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
regularised1 = Theta1*lambda/m;
regularised2 = Theta2*lambda/m;
regularised(:,1) = 0;
regularised(:,1) = 0;
Theta1_grad = Theta1_grad + regularised1;
Theta2_grad = Theta2_grad + regularised2;
```

## ex6 -> Svm along with kernels and spam filtering
## ex7 -> K-mean clustering algorithm and PCA implementation
