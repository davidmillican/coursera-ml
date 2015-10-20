function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = size(y, 1); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Create theta_reg - which has zero value for the first element
theta_reg = theta;
theta_reg(1) = 0;

h = sigmoid(X * theta);
J = -1/m * (y' * log(h) + (1 - y)' * log(1 - h)) ...
	+ lambda / (2 * m) * (theta_reg' * theta_reg);

grad = 1/m * X' * (h - y) + lambda/m * theta_reg;

% =============================================================

end
