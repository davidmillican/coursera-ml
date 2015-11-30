function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S

covariance	= (1 / size(X, 1)) * (X' * X);
[U, S, V]	= svd(covariance);

end
