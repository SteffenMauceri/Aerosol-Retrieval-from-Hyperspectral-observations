% Loads predictions of aerosol optical thickness (AOT) for AIVIRS flights 
% and calculates mean, median and standard deviation
%
% input:    name of neural network to load retrieved AOT from AVIRIS
% output:   Mean, Median, Standard deviation of retrieved AOT from AIVIRS
%           flights
% 
%
% Steffen Mauceri, March 2018


%% START Make Changes .................
% visualize some results after processing
visualize_IO = false;
% use a set percent of the retrieved AVIRIS AOT pixel instead of a fixed threshold 
cutOff_perc = true;

% name of neural network that generated AOT for AVIRIS flights
name = '5.1_05_128x32_noise1_reg5000_2000_ind_relu_refl'
%% END Make Changes .................

load(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/trained/prediction_AVIRIS' , name,'.mat'))

% calculate combined AOT of all three aerosol types for comparison to MODIS
% and AERONET
prediction(:,2:4) = prediction(:,1:3);
prediction(:,1) = sum(prediction(:,2:4),2);
AVIRIS = prediction;

% load output from verification neural network to mask AVIRIS pixels that
% are to different to the training-set. Radiance at every wavelengt was
% compared and difference was calculated
name = '5.1_05_512x32_noise1_verification_reg5000_100';
load(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/trained/prediction_AVIRIS' , name,'.mat'), 'verification')
% concaternate with AOT predictions
AVIRIS(:,5:322+4) = verification;

% calculate mean squared error
MEAE = mean((AVIRIS(:,5:322+4)).^2,2);
% save mean squared error to AOT predictions
AVIRIS(:,5) = MEAE*10;
% remove calculated differences for individual wavelengths
AVIRIS(:,6:end) = [];

% cacluated median, mean and standard deviation for the 21 AVIRIS flights
% after removing pixels that are different to the training-set
for j=1:21
    % subset a scene from the 21 AVIRIS-eval scenes
    scene_j = AVIRIS(j*10000-9999:j*10000, :);
    % cut off AVIRIS pixel above a threshold
    if cutOff_perc %can be done in percent
        cutOff = prctile(scene_j(:,5), 50);
    else % or at a fixed value
        cutOff = 0.4; 
    end
    scene_j(scene_j(:,5)>cutOff,1:4) = NaN;
    
    MED(j,:) = nanmedian(scene_j,1);
    MEA(j,:) = nanmean(scene_j,1);
    STD(j,:) = nanstd(scene_j);
end

%visualize some results if visualize_IO==True
if visualize_IO
    %histogram of AOT for the flights
    hist(MED(:,1))
    avir = reshape(AVIRIS,100, 2100, 5);
    subplot(2,1,1)
    imshow(squeeze(avir(:,:,1)))
    subplot(2,1,2)
    imshow(squeeze(avir(:,:,5)*0.2))
    
    %visualize retrieved AOT and pixel mask spatially resolved
    figure
    leg = {'Carbon', 'Dust', 'Sulfate'};
    for i=1:3
        subplot(1,3,i)
        hist(MED(:,i+1), 10)
        legend(leg(i))
        ylabel('Number of Flights')
        xlabel('AOT')
    end
end


