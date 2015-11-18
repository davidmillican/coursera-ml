function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


test_vals	= [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
n_tests		= numel(test_vals);
accuracy	= NaN(n_tests);

for i = 1:n_tests
	for j = 1:n_tests

		C_test = test_vals(i);
		sigma_test = test_vals(j);
		
		model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
		
		yval_pred = svmPredict(model, Xval);
		accuracy(i, j) = mean(double(yval_pred == yval));
		
	end
end
[~, i] = max(accuracy(:));
[C_i, sigma_i] = ind2sub([n_tests, n_tests], i);

C		= test_vals(C_i);
sigma	= test_vals(sigma_i);

fprintf('Best C is %g\n', C);
fprintf('Best sigma is %g\n', sigma);

end
