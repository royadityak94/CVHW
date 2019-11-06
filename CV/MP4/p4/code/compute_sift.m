function sift_arr = compute_sift(I, circles, enlarge_factor)
% Modified from Lana Lazebnik's non rotational invariant SIFT 
% using VL-FEAT.
%
% I - image
% circles - Nx3 array where N is the number of circles, where the
%    first column is the x-coordinate, the second column is the y-coordinate,
%    and the third column is the radius
% enlarge_factor is by how much to enarge the radius of the circle before
%    computing the descriptor (a factor of 1.5 or larger is usually necessary
%    for best performance)
% The output is an Nx128 array of SIFT descriptors
% (c) Lana Lazebnik

if ndims(I) == 3
    I = im2double(rgb2gray(I));
else
    I = im2double(I);
end

% parameters (default SIFT size)
num_angles = 8;
num_bins = 4;
num_samples = num_bins * num_bins;
alpha = 9; % smoothing for orientation histogram

if nargin < 3
    enlarge_factor = 1.5;
end

angle_step = 2 * pi / num_angles;
angles = 0:angle_step:2*pi;
angles(num_angles+1) = []; % bin centers

[hgt wid] = size(I);
num_pts = size(circles,1);

sift_arr = zeros(num_pts, num_samples * num_angles);

% edge image
sigma_edge = 1;


[G_X,G_Y]=gen_dgauss(sigma_edge);
I_X = filter2(G_X, I, 'same'); % vertical edges
I_Y = filter2(G_Y, I, 'same'); % horizontal edges
I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
I_theta = atan2(I_Y,I_X);
I_theta(isnan(I_theta)) = 0; % necessary????

% make default grid of samples (centered at zero, width 2)
interval = 2/num_bins:2/num_bins:2;
interval = interval - (1/num_bins + 1);
[grid_x grid_y] = meshgrid(interval, interval);
grid_x = reshape(grid_x, [1 num_samples]);
grid_y = reshape(grid_y, [1 num_samples]);

% make orientation images
I_orientation = zeros(hgt, wid, num_angles);
% for each histogram angle
for a=1:num_angles    
    % compute each orientation channel
    tmp = cos(I_theta - angles(a)).^alpha;
    tmp = tmp .* (tmp > 0);
    
    % weight by magnitude
    I_orientation(:,:,a) = tmp .* I_mag;
end

% for all circles
orienStep = 2*pi/36;
orienGrid = -pi+orienStep/2:orienStep:pi;
theta = zeros(num_pts, 1);
for i=1:num_pts
    cx = circles(i,1);
    cy = circles(i,2);
    r = circles(i,3) * enlarge_factor;

    % find coordinates of sample points (bin centers)
    grid_x_t = grid_x * r + cx;
    grid_y_t = grid_y * r + cy;
    grid_res = grid_y_t(2) - grid_y_t(1);
    
    % find window of pixels that contributes to this descriptor
    x_lo = floor(max(cx - r - grid_res/2, 1));
    x_hi = ceil(min(cx + r + grid_res/2, wid));
    y_lo = floor(max(cy - r - grid_res/2, 1));
    y_hi = ceil(min(cy + r + grid_res/2, hgt));

    % compute the dominating orientation
    orien = I_theta(y_lo:y_hi, x_lo:x_hi);
    orienDiff = abs(bsxfun(@minus, orien(:), orienGrid));
    [~, oidx] = min(orienDiff, [], 2);
    orienHist = accumarray(oidx, 1);
    [~, maxtheta] = max(orienHist);
    theta(i) = orienGrid(maxtheta);

end
% compute the sift descriptor
circles = [circles, theta];
GRAD = zeros(2, hgt, wid, 'single');
GRAD(1,:,:) = I_mag;
GRAD(2,:,:) = I_theta;
sift_arr = vl_siftdescriptor(GRAD, circles');
sift_arr = single(sift_arr');


function [GX,GY]=gen_dgauss(sigma)

f_wid = 4 * floor(sigma);
G = normpdf(-f_wid:f_wid,0,sigma);
G = G' * G;
[GX,GY] = gradient(G); 

GX = GX * 2 ./ sum(sum(abs(GX)));
GY = GY * 2 ./ sum(sum(abs(GY)));



function pdf = normpdf (x, m, s)
% NORMPDF  PDF of the normal distribution
%  PDF = normpdf(X, M, S) computes the probability density
%  function (PDF) at X of the normal distribution with mean M
%  and standard deviation S.
%
%  PDF = normpdf(X) is equivalent to PDF = normpdf(X, 0, 1)

% Adapted for Matlab (R) from GNU Octave 3.0.1
% Original file: statistics/distributions/normpdf.m
% Original author: TT <Teresa.Twaroch@ci.tuwien.ac.at>

% Copyright (C) 1995, 1996, 1997, 2005, 2006, 2007 Kurt Hornik
% Copyright (C) 2008-2009 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <http://www.gnu.org/licenses/>.

if (nargin ~= 1 && nargin ~= 3)
    error('normpdf: you must give one or three arguments');
end

if (nargin == 1)
    m = 0;
    s = 1;
end

if (~isscalar (m) || ~isscalar (s))
    [retval, x, m, s] = common_size (x, m, s);
    if (retval > 0)
        error ('normpdf: x, m and s must be of common size or scalars');
    end
end

sz = size (x);
pdf = zeros (sz);

if (isscalar (m) && isscalar (s))
    if (find (isinf (m) | isnan (m) | ~(s >= 0) | ~(s < Inf)))
        pdf = NaN * ones (sz);
    else
        pdf = stdnormal_pdf ((x - m) ./ s) ./ s;
    end
else
    k = find (isinf (m) | isnan (m) | ~(s >= 0) | ~(s < Inf));
    if (any (k))
        pdf(k) = NaN;
    end

    k = find (~isinf (m) & ~isnan (m) & (s >= 0) & (s < Inf));
    if (any (k))
        pdf(k) = stdnormal_pdf ((x(k) - m(k)) ./ s(k)) ./ s(k);
    end
end

pdf((s == 0) & (x == m)) = Inf;
pdf((s == 0) & ((x < m) | (x > m))) = 0;




function pdf = stdnormal_pdf (x)
% STDNORMAL_PDF  PDF of the standard normal distribution
%  PDF = stdnormal_pdf(X)
%  For each element of X, compute the PDF of the standard normal
%  distribution at X.
% 
% Adapted for Matlab (R) from GNU Octave 3.0.1
% Original file: statistics/distributions/stdnormal_pdf.m
% Original author: TT <Teresa.Twaroch@ci.tuwien.ac.at>
% 
% Copyright (C) 1995, 1996, 1997, 1998, 2000, 2002, 2004, 2005, 2006,
%               2007 Kurt Hornik
% Copyright (C) 2008-2009 Dynare Team
% 
% This file is part of Dynare.
% 
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <http://www.gnu.org/licenses/>.

if (nargin ~= 1)
    error('stdnormal_pdf: you should provide one argument');
end

sz = size(x);
pdf = zeros (sz);

k = find (isnan (x));
if (any (k))
    pdf(k) = NaN;
end

k = find (~isinf (x));
if (any (k))
    pdf (k) = (2 * pi)^(- 1/2) * exp (- x(k) .^ 2 / 2);
end

