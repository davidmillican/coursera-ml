function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
err = X * theta - y;
J = (0.5 / m) * (err' * err + lambda * theta(2:end)' * theta(2:end));

if nargout > 1
	grad = (1 / m) * (X' * err + [0; lambda * theta(2:end)]);
end
	
end
