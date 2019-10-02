% Prepare AVIRIS measured radiances for input to Neural Network
% Cuts AVIRIS Scene to standard size and normalizes radiance for SZA and sun-earth distance 
% Removes noise from AVIRIS radiances with a PCA
%
% input:    AVIRIS observed radiances for a given scene
%           
% output:   Neural Network inputs for AVIRIS scene
%
%
% Steffen Mauceri, Jun 2018

clear

%load AVIRIS scan file
load('/Volumes/2TB/AVIRIS-NG/AvirisScan20160101.mat') 
% specify day of year from AVIRIS flight
day = 1;


%load wavelength grid of MODTRAN radiative transfer calculations
load('/Users/stma4117/Studium/LASP/Hyper/NN/GeneratedData/SurfaceGenerated4','wl')
%load wavelength grid of AVIRIS-NG
load('/Users/stma4117/Studium/LASP/Hyper/NN/Aviris/AvirisWL.mat')
%inspect image
X = img(:,:,[45 25 5]);
meanX = mean(mean(X));
stdX = mean(std(X));
Y = (X-(meanX-3*stdX))./(meanX+3*stdX);
imagesc(Y)
title('False Color Image')

%% cut smaller 
% choose image crop for neural network analysis
height = 4400:4905;
width = 201:300;

img = img(height, width, :); %504x100x425
info = info(height, width, :);

%make matrices flat
img = reshape(img, size(img,1)*size(img,2),425);
info = reshape(info, size(info,1)*size(info,2),3);

%Interpolate AVIRIS wavelength grid to MODTRAN wavelength grid
input_ = zeros(length(img),319);
for i=1:length(img)
    input_(i,:) = interp1(AVIRISwl(:,1)*1000, img(i,:), wl);
end

%remove noise in AVIRIS-NG with PCA
[COEFF, SCORE, LATENT] = pca(input_', 'Centered', false);

%inspect first 20 principal components
for i=1:20
    subplot(2,10,i)
    imagesc(reshape(COEFF(:,i), length(pieces), 100))
end

%remove principal components greater than 16
COEFF = COEFF(:,1:16);
SCORE = SCORE(:,1:16);
A = SCORE*COEFF';
input_ = A';

%make input matrix for neural network with cos normalized radiance,
%cos(SZA), ground elevation, distance sensor-ground
inputAVIRIS = [input_./cosd(info(:,1)) cosd(info(:,1)) info(:,2:3)/1000];

d = 1 - 0.01672*cosd(0.9856*(day - 4));
inputAVIRIS(:,1:wl_bands) = inputAVIRIS(:,1:length(wl)).*d^2;

imagesc(reshape(inputAVIRIS(:,50), length(pieces), 100)) %sanity check

%save everything for matlab and as csv for python
save('/Users/stma4117/Studium/LASP/Hyper/NN/GeneratedData/AVIRIS_20160204_refl.mat', 'inputAVIRIS', 'wl')
dlmwrite('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/data/AVIRIS_20160204_refl.csv', inputAVIRIS, 'delimiter', ',', 'precision', 9)

