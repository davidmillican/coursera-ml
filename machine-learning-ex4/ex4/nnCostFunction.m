function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
k = size(Theta2, 1);
         
% Feedforward
a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Calculate cost
y_full = zeros(m, k);
y_full(sub2ind([m, k], 1:m, y')) = 1;
J = -sum(sum(y_full .* log(a3) + (1 - y_full) .* log(1 - a3))) / m;

% Regularize
theta_sq = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2));
J = J + lambda * theta_sq / (2*m);

% Backpropagate errors
delta3 = a3 - y_full;
delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z2);

Theta2_grad = delta3' * a2 / m;
Theta1_grad = delta2' * a1 / m;

% Add regularization to the gradients
Theta2_grad = Theta2_grad + [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)] * lambda / m;
Theta1_grad = Theta1_grad + [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)] * lambda / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
