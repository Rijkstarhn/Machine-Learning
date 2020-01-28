function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta;
thetaj = theta.^2;
J = 1/m * sum(-y .* log(sigmoid(z)) - (1-y) .* log(1 - sigmoid(z))) + 0.5 * lambda / m * sum(thetaj(2:length(thetaj),:));
grad_0 = 1/m * ((X(:,1))' * (sigmoid(z) - y));
grad_rest = 1/m * (X(:,2:size(X,2))' * (sigmoid(z) - y)) + lambda / m * theta(2:size(theta,1),:);
grad = [grad_0;grad_rest];





% =============================================================

end
