function ypred = multiclassLRPredict(model,x)
numData = size(x,2);

% Simply predict the first class (Implement this)
ypred = model.classLabels(1)*ones(1, numData);