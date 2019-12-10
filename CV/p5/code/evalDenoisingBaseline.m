% Matlab code to compare errors using Gaussian and Median filters
% Load images
%im = im2double(imread('../data/denoising/saturn.png'));
%noise1 = im2double(imread('../data/denoising/saturn-noisy.png'));
im = im2double(imread('../data/denoising/lena.png'));
noise1 = im2double(imread('../data/denoising/lena-noisy.png'));

% Compute error
error1 = sum(sum((im - noise1).^2));
fprintf('Input, Errors: %.2f %.2f\n', error1, error2)

% Display the images
figure(1);
subplot(1,4,1); imshow(im); title('Clean');
subplot(1,4,2); imshow(noise1); title(sprintf('Noisy SE %.1f', error1));

%% Denoising algorithm (with Gaussian)
sigma = 2;
g = fspecial('Gaussian', ceil(6*sigma + 1), sigma);
denoise1 = imfilter(noise1, g, 'replicate', 'same');
error1 = sum(sum((im - denoise1).^2));
fprintf('Gaussian filter SE: %.2f\n', error1)
subplot(1,4,3); imshow(denoise1); title(sprintf('Gaussian SE %.1f', error1));

%% Denoising algorithm (median filter)
sz = [7 7];
denoise1 = medfilt2(noise1, sz);
error1 = sum(sum((im - denoise1).^2));
fprintf('Median filter SE: %.2f\n', error1)
subplot(1,4,4); imshow(denoise1); title(sprintf('Median SE %.1f', error1));



