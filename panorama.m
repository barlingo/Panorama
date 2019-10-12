clear; clc;

%Least Square Fitting of a plane

n_samples = 500;

x = 10 * rand(1, n_samples);
y = 10 * rand(1, n_samples);

%Generate random alpha, beta and gamma values
alpha = 10* randn(1,1);
beta = 10* randn(1,1);
gamma = 10* randn(1,1);

%Calculate non noisy plane 
z_clean = alpha*x + beta*y + gamma;

%Add noise
z_noise = z_clean + randn(1, n_samples);
fprintf('Q1.1\n');
fprintf('Generated %d samples for z = %.3fx + %.3fy + %.3fz\n', n_samples, alpha, beta, gamma);
fprintf('\nPress any key to continue...\n');
input('');
%Part 1.2
%Form A matrix from ||Ax - b||^2
fprintf('Q1.2\n');
fprintf('Forming non-homogeneus matrix equation A x = b\n');
fprintf('    [ x1  y2  1 ]\t    [ alpha ]\t    [ z1 ]\n');
fprintf('A = [ ..  ..  ..]\tx = [ beta  ]\tb = [ .. ]\n');
fprintf('    [ xn  yn  1 ]\t    [ gamma ]\t    [ zn ]\n');
A = [x' y' ones([500 1])]; %Transpose require to transform x and y from row to columns

%Calculate x for dE/dx = 0 , x = (A'A)^-1 * b
estimate = A\z_noise'; %Transpose required to transform z to columns
fprintf('Estimating x values for dE/dx = 0 , x = (AtA)^-1 * b\n');
fprintf('\nPress any key to continue...\n');
input('');
%Take alpha beta and gamma values
alpha_calc = estimate(1);
beta_calc = estimate(2);
gamma_calc = estimate(3);

%Calculate errors
error_alpha = abs(alpha - alpha_calc);
error_beta = abs(beta - beta_calc);
error_gamma = abs(gamma - gamma_calc);

%absolute error found
fprintf('Q1.3 Estimated errors\n');
fprintf("Variable\tEstimated\tReal\tError\tError (%%)\n");
fprintf("Alpha:\t\t%.3f\t\t%.3f\t%.3f\t%.1f%%\n", alpha_calc, alpha, error_alpha, (error_alpha/alpha)*100);
fprintf("Beta :\t\t%.3f\t\t%.3f\t%.3f\t%.1f%%\n", beta_calc, beta, error_beta, (error_beta/beta)*100);
fprintf("Gamma:\t\t%.3f\t\t%.3f\t%.3f\t%.1f%%\n", gamma_calc, gamma, error_gamma, (error_gamma/gamma)*100);


fprintf('\nPress any key to continue...\n');
input(''); clc; close all;

%RANSAC-based image stitching
clear; clc;  close all;
%VLFEATROOT installed in the following path. Sustitude for your own path
%for the program to run correctly.
VLFEATROOT= 'C:\Program Files\MATLAB\R2018b\toolbox\vlfeat-0.9.21';
VLFEATRUN = strcat(VLFEATROOT,'/toolbox/vl_setup');
run(VLFEATRUN);

%Image pre-processing
fprintf('\nQ2.1. Preprocessing images...\n');
tic
img_left = imread('parliament-left.jpg');   
img_left = im2double(img_left);
%img_left = imresize(img_left, 0.3); %Resize for faster procesing
img_left_color = img_left;
img_left = single(rgb2gray(img_left));

img_right = imread('parliament-right.jpg');   
img_right = im2double(img_right);
%img_right = imresize(img_right, 0.3); %Resize for faster procesing
img_right_color = img_right;
img_right = single(rgb2gray(img_right));
toc
fprintf('\nPress any key to continue...\n');
input('');

%Detect keypoints and descriptor extraction.
fprintf('\nQ2.2. Detecting SIFT descriptors using vl_feat library...\n');
tic
[f_right,d_right] = vl_sift(img_right) ;
[f_left,d_left] = vl_sift(img_left) ;
toc
fprintf('\nPress any key to continue...\n');
input('');

%Feature matching. Returns the euclidian distance and also the
%matches between SIFT descriptors.
fprintf('\nQ2.3. Finding euclidian distance between matches...\n');
tic

[f_matches, distance] = vl_ubcmatch(d_right,d_left) ;
toc
fprintf('\nPress any key to continue...\n');
input('');

%Part 2.4. Prune features.
fprintf('\nQ2.4. Selecting and plotting best 100 points found...\n');
tic
n_sel = 100;
%Select the n_sel elements that have the smallest euclidian distance
[d_small, d_small_id] = getSmallest(distance, n_sel);
%Select matches with smalles distances
f_matches_sel = f_matches(:,d_small_id);

%Select corresponding features that were matched and have smallest factor
%on each image
f_right_sel = f_right(1:2,f_matches_sel(1,:)); 
f_left_sel = f_left(1:2,f_matches_sel(2,:));

%Plot 20 features out of the n_sel
figure(1); ax = axes;
%This function might give an error if the toolbox is not installed
%correctly
showMatchedFeatures(img_left, img_right, f_left_sel', f_right_sel', 'montage', 'Parent', ax);
title(ax, 'Point matches');
legend(ax, 'Matched points 1','Matched points 2');
toc
fprintf('\nPress any key to continue...\n');
input('');

%Robust transformation estimation RANSAC
fprintf('\nQ2.5. Estimating with RANSAC what model fits the data better...\n');
tic
%Adjusted iterations and threshold to maximize number of inliers.
n_ite = n_sel;
d_thre = 0.1; %0.1 Pixel radius define as threshold
c_inliers = zeros(n_ite,n_sel);


for i = 1:n_ite
    %Select 3 random sample points for each matching matrix.
    perm = randsample(n_sel,3);
    M = zeros(6,6);
    b = zeros(6,1);
    j = 0;
    %Construct matrices
    for i = 1:3 
        
        j = j + 1;
        row1_M = [ f_left_sel(1, perm(i)), f_left_sel(2, perm(i)), 1, 0, 0, 0];
        row2_M = [ 0, 0, 0, f_left_sel(1, perm(i)), f_left_sel(2, perm(i)), 1];
    
        row1_b = [ f_right_sel(1, perm(i))];
        row2_b = [ f_right_sel(2, perm(i))];

        M(j,:)= row1_M;
        b(j,:)= row1_b;
    
        j = j + 1;
        M(j,:)= row2_M;
        b(j,:)= row2_b;
    end
    
    %Calculate transformation matrix
    a = inv((M')*M)*(M')*b;
    a = [a(1,1), a(2,1), a(3,1);
         a(4,1), a(5,1), a(6,1)];
    a = [a; [0 0 1]];
        
    for j = 1 : n_sel
        %Take one of the matched features from both sides
        point_left = f_left_sel(:,j);
        point_right = f_right_sel(:,j);
        point_left_aff = a * [point_left;1]; %Calculate transformation
        
        %Calculate its sum square distance as measure point
        d_ssd = (point_right(1,1) - point_left_aff(1,1))^2 + (point_right(2,1) - point_left_aff(2,1))^2;
        
        %If the distance is within threshold increase inliner counter.
        if (d_ssd < d_thre) 
            c_inliers(i,1) = c_inliers(i,1) + 1;
            c_inliers(i,j+1) = j;
        end
    end
end
toc
fprintf('\nPress any key to continue...\n');
input('');

fprintf('\nQ2.6. Computing optimal transformation using all inliers\n');
tic
% Find maximum number of inliers.
[max_val, max_loc] = max(c_inliers(:,1));
c_inliers_max = c_inliers(max_loc, :);
c_inliers_max = c_inliers_max(c_inliers_max ~= 0); %Take only inlier values not corresponding to 0

%Allocate matrices/variables for calculation
M = zeros((size(c_inliers_max,2)-1)*2, 6);
b = zeros((size(c_inliers_max,2)-1)*2, 1);
j = 0;


for i = 2:size(c_inliers_max,2) %Discard first value since it is the number of inliers
    %form M matrix
    j = j + 1;
    inlier = c_inliers_max(i);
    row1_M = [ f_left_sel(1,inlier), f_left_sel(2,inlier), 1, 0, 0, 0];
    row2_M = [ 0, 0, 0, f_left_sel(1,inlier), f_left_sel(2,inlier), 1];
    
    row1_b = [ f_right_sel(1,inlier)];
    row2_b = [ f_right_sel(2,inlier)];

    M(j,:)= row1_M;
    b(j,:)= row1_b;
    
    j = j + 1;
    M(j,:)= row2_M;
    b(j,:)= row2_b;
end


%Result affine transformation
H=M\b; 

H = [H(1,1), H(2,1), H(3,1);
     H(4,1), H(5,1), H(6,1)];
H = [H; [0 0 1]];

tform = affine2d(H');
img_left_warp = imwarp(img_left,tform);
img_right_size = size(img_right);
img_left_warp_size = size(img_left_warp);

figure(2);
subplot(1,2,1);
imshow(img_left_warp); title('Left side transformed');
img_left_warp = imwarp(img_left,tform);
figure(2);
subplot(1,2,2);
imshow(img_right); title('Right side');
toc
fprintf('\nPress any key to continue...\n');
input('');

fprintf('\nQ2.7. Creating panorama \n');
%Calculate limits on new coordinate system of canvas
img_left_limits = [1 1 1; 
                    size(img_left_warp,2) 1 1;
                    size(img_left_warp,2) size(img_left_warp,1) 1;
                    1 size(img_left_warp,1) 1]; 
img_left_limits = inv(H) * img_left_limits'; % Transform coordinates.


%Get regular coordinates
for i = 1:4
    img_left_limits(1:2,i) = img_left_limits(1:2,i) / img_left_limits(3,i);
end

panorama_x = round(max([img_left_limits(1,:) size(img_right,2)]));
panorama_y = round(max([img_left_limits(2,:) size(img_right,1)]));

canvas_panorama = zeros(panorama_y,panorama_x);

[xcoor, ycoor] = meshgrid([1:panorama_x],[1:panorama_y]);

xtrans = (H(1,1)*xcoor + H(1,2)*ycoor + H(1,3)); 
ytrans = (H(2,1)*xcoor + H(2,2)*ycoor + H(2,3)); 

%Place image left onto canvas
img_left_canvas = canvas_panorama; 
img_left_canvas(1:size(img_left,1),1:size(img_left,2)) = img_left;

%Transform img_right to the new canvas annd get rid of NaN values
img_right_canvas = vl_imwbackward(im2double(img_right),xtrans,ytrans);
img_right_canvas(isnan(img_right_canvas)) = 0;

canvas_panorama = imadd(img_left_canvas,img_right_canvas);

%Take pixels with maximum value
for i = 1:panorama_y
    for j = 1:panorama_x
        if (img_left_canvas(i,j) > img_right_canvas(i,j))
            canvas_panorama(i,j) = img_left_canvas(i,j);
        elseif(img_left_canvas(i,j) < img_right_canvas(i,j))
            canvas_panorama(i,j) = img_right_canvas(i,j);
        else
            canvas_panorama(i,j) =canvas_panorama(i,j);
        end
    end
end

figure(3), imshow(canvas_panorama);
fprintf('\nPress any key to continue...\n');
input('');

img_left_color_canvas = zeros(panorama_y,panorama_x,3);
%Take image 1 color in panorama canvas
img_left_color_canvas(1:size(img_left_color,1),1:size(img_left_color,2),:) = img_left_color; % copy im1 into composite

%Take transformed of image 2 in color
img_right_color_canvas_trans = vl_imwbackward(img_right_color,xtrans,ytrans);
img_right_color_canvas_trans (isnan(img_right_color_canvas_trans)) = 0;

canvas_panorama_color = imadd(img_left_color_canvas,img_right_color_canvas_trans);

%Take pixels with maximum value
for i = 1:panorama_y
    for j = 1:panorama_x
        for d = 1:3
            if (img_left_color_canvas(i,j,d) > img_right_color_canvas_trans(i,j,d))
                canvas_panorama_color(i,j,d) = img_left_color_canvas(i,j,d);
            elseif(img_left_color_canvas(i,j,d) < img_right_color_canvas_trans(i,j,d))
                canvas_panorama_color(i,j,d) = img_right_color_canvas_trans(i,j,d);
            else
                canvas_panorama_color(i,j,d) =canvas_panorama_color(i,j,d);
            end
        end    
    end
end

figure(4), imshow(canvas_panorama_color);


function [smallest, smallestId] = getSmallest(A, n)
     [ASorted, AIdx] = sort(A);
     smallest = ASorted(1:n);
     smallestId = AIdx(1:n);
end


