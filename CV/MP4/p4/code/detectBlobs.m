function blobs = detectBlobs(im, param)
% DETECTBLOBS detects blobs in an image
%   BLOBS = DETECTBLOBSCALEIMAGE(IM, PARAM) detects multi-scale blobs in IM.
%   The method uses the Laplacian of Gaussian filter to find blobs across
%   scale space.
% 
% Input:
%   IM - input image
%   PARAM - struct containing the following fields
%       PARAM.SIGMA - sigma of the LoG filter (smallest blob desired)
%       PARAM.INTERVAL - number of intervals in an octave
%       PARAM.THRESHOLD - threshold for blob detection
%       PARAM.DISPLAY - if true then then shows intermediate results
%
% Ouput:
%   BLOBS - n x 4 array with blob in each row in (x, y, radius, score)
%

% Convert image to grayscale and convert it to double [0 1].
if size(im, 3) > 1
    im = rgb2gray(im);
end
if ~isfloat(im)
    im = im2double(im);
end

% If param is not specified use default
if nargin < 2
    param.sigma = 2;
    param.interval = 12;
    param.threshold = 1e-2;
    param.display = false;
end

% dummy blob
blobs = [size(im, 2)/2, size(im, 1)/2, 100, 1.0];


%% Implement these:

% Compute the scale space representation


% Compute the blob response


% Perform NMS (spatial and scale space)

