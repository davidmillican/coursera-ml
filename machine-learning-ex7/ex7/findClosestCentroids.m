function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Compute the squared distances between all points and centroids 
deltas = bsxfun(@minus, permute(X, [1, 3, 2]), permute(centroids, [3, 1, 2]));
distances = sum(deltas.^2, 3);

% Find the closest to each point
[~, idx] = min(distances, [], 2);

end

