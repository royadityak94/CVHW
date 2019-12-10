function model = multiclassLRTrain(x, y, param)
classLabels = unique(y);
numClass = length(classLabels);
numFeats = size(x,1);
numData = size(x,2);

% Initialize weights randomly (Implement the gradient descent)
model.w = randn(numClass, numFeats)*0.01;
model.classLabels = classLabels;
