function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
x1 = [1 2 1]; x2 = [0 4 -1];

C_set = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_set = [0.01;0.03;0.1;0.3;1;3;10;30];

J_cv = zeros(size(C_set,1) * size(sigma_set,1),1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

count = 1;
for i = 1 : length(C_set)
    C = C_set(i);
    for j = 1 : length(C_set)
       sigma = sigma_set(j);
       model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
       prediction = svmPredict(model, Xval);
       J_cv(count) = mean(double(prediction ~= yval));
       count = count +1; 
    end
end

[value,pos] = min(J_cv);
j = mod(pos,length(C_set));
sigma = sigma_set(j);
C = C_set((pos - j) / length(C_set));



% =========================================================================

end
