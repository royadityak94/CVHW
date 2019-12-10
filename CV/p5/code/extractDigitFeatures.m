function features = extractDigitFeatures(x, featureType)
% EXTRACTDIGITFEATURES extracts features from digit images
%   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
%   images X of the provided FEATURETYPE. The images are assumed to the of
%   size [W H 1 N] where the first two dimensions are the width and height.
%   The output is of size [D N] where D is the size of each feature and N
%   is the number of images. 

switch featureType
    case 'pixel'
        features = zeroFeatures(x);  % implement this
    case 'hog'
        features = zeroFeatures(x);  % implement this
    case 'lbp'
        features = zeroFeatures(x);  % implement this
end

function features = zeroFeatures(x)
features = zeros(10, size(x,4));