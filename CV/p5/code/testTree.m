% There are three versions of MNIST dataset
dataTypes = {'digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat'};
featureType = 'lbpTokens';
% Accuracy placeholder
accuracy = zeros(length(dataTypes), 1);
trainSet = 1;
testSet = 2; % 2 = validation, 3 = test

for i = 1:length(dataTypes), 
    dataType = dataTypes{i};
    % load data
    load(fullfile('..', 'data', dataType));
    fprintf('+++ Loading digits of dataType: %s\n', dataType);
    
    tic;
    features = extractDigitFeatures(data.x, featureType);
    fprintf(' %.2fs to extract %s features for %i images\n', toc, featureType, size(features,2)); 

    % Train model
    tic;
    model = fitctree(features(:, data.set==trainSet)', data.y(data.set==trainSet)');
    fprintf(' %.2fs to train DT model\n', toc);

    % Test the model
    ypred = predict(model, features(:, data.set==testSet)');
    y = data.y(data.set==testSet);

    % Measure accuracy
    [acc, conf] = evaluateLabels(y, ypred', false);
    fprintf(' Accuracy [testSet=%i] %.2f%%\n\n', testSet, acc*100);
    accuracy(i) = acc;
end

%% Random forest
numTrees = 50;
for i = 1:length(dataTypes), 
    dataType = dataTypes{i};
    % load data
    load(fullfile('..', 'data', dataType));
    fprintf('+++ Loading digits of dataType: %s\n', dataType);
    
    features = extractDigitFeatures(data.x, 'pixel');
    % Train model
    tic;
    clear model;
    for m = 1:numTrees,
        model{m} = fitctree(features(:, data.set==trainSet)', data.y(data.set==trainSet)', 'NumVariablesToSample', 20);
    end
    fprintf(' %.2fs to train %i DT model\n',  toc, numTrees);
 
    % Test the model
    clear ypred;
    for m = 1:numTrees,
         ypred{m} = predict(model{m}, features(:, data.set==testSet)');
    end
    ypred = mode(cat(2, ypred{:}), 2);

    y = data.y(data.set==testSet);

    % Measure accuracy
    [acc, conf] = evaluateLabels(y, ypred', false);
    fprintf(' Accuracy [testSet=%i] %.2f%%\n\n', testSet, acc*100);
    accuracy(i) = acc;
end