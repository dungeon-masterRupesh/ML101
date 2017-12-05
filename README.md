# ML101
This part and commit was done by Ubuntu environment of Windows

## ex1 -> linear regression
Implementation of linear regression using gradient descend and normal equation
## ex2 -> logistic regression
Implementation of a logistic regression of training set using Gradient Descent algorithm and also using fminunc having single as well as multiple features. Regularization implementation to reduce the variance by addition of a little bias.
## ex3 -> multi classification and predicting value from an already trained neural net
Multiple class logistic regression using OneVsAll technique for handwritten digits 20 X 20 pixels. And developing prediction function for a
trained neural net for the same.
## ex4 -> neural net and back propagation
Implemented back propagation using vectorisation for self-learning neural net for recognising handwritten digits 20 X 20 pixels.
```
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2,1), 1) sigmoid(z2)];
%a3 == h
h = sigmoid(a2 * Theta2');
y_logic = zeros(m,size(Theta2,1));
%% converting [1 2 1] to [1 0; 0 1; 1 0]. If anyone knows the matlab command to do so, Plz push your code.
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
Email spam filtering given a dictionary using linear as well as Gaussian Kernels and applied algorithm to find most optimal parameters for the Gaussian kernels among a given set of possible parameters.
## ex7 -> K-mean clustering algorithm and PCA implementation
K-mean clustering to a 2 dimention data as through unsupervised learning find a division. Application of k-mean clustering for image compression.And PCA for image compression as well as recontructing an approximation of data.
