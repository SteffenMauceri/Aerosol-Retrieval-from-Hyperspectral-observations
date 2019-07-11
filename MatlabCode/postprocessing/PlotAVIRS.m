% Plot aerosol optical thickness (AOT) retrieval from AVIRIS-NG for a single
% scene
%
% input:    name of neural network that made the prediction and AVIRIS scene name 
%
% output:   Spatially resolved plot of AOT and Median, Standard deviation
%           per aerosol type for the analyzed scene
%
% Steffen Mauceri, Sept 2018

clear

% START Change Code .............
% choose neural network and scene to plot
name = '5.1_05_128x64_noise1_reg5000_2000_ind_relu_refl';
% choose scene
scene = '204';
% choose what percentage of pixels we want to remove from evaluation
% set to 0 to use a fixed threshold
cutOff_perc = 50;
% apply median filter to output. Set to 0 to not apply filter
Median_Filter = 0;
% END Change Code ...................

% load AOT predictions
load(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/trained/prediction_AVIRIS' , name,'_',scene,'.mat'))
prediction(:,2:4) = prediction(:,1:3); 
prediction(:,1) = sum(prediction(:,2:4),2); %calculate combined AOT of all aerosol types

% load output from verification neural network to mask AVIRIS pixel that
% are to different to the training set. Radiance at every wavelengt was
% compared and difference was calculated
name = '5.1_05_512x32_noise1_verification_reg5000_500';
load(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/trained/prediction_AVIRIS' , name,'_',scene,'.mat'), 'verification')
% concaternate with AOT predictions
prediction(:,5:322+4) = verification;

% calculate mean squared error
MEAE = mean((prediction(:,5:322+4)).^2,2); %calculate difference ob prediction-NG observations to training-set
% save mean squared error to AOT predictions
prediction(:,5) = MEAE*10;
% remove calculated differences for individual wavelengths
prediction(:,6:end) = [];

% cut off AVIRIS pixel above a threshold
if cutOff_perc > 0 %can be done in percent
    cutOff = prctile(prediction(:,5), cutOff_perc);
else % or at a fixed value
    cutOff = 0.1;
end
prediction(prediction(:,5)>cutOff,1:4) = NaN;

% calculate Median and Standard deviation
MED = round(nanmedian(prediction,1),2);
STD = round(nanstd(prediction),2);

%reshape prediction retrieval for ploting 
prediction = reshape(prediction, 504, 100, 5);  

% apply median filter to AOT retrieval if Median_Filter == True
if Median_Filter>0
    for i = 1:4
        prediction(:,:,i) = movmedian(movmedian(prediction(:,:,i),Median_Filter,1, 'omitnan'),Median_Filter,2, 'omitnan');
    end
end

%% Plot
title_i = {'AOT comb.', 'Carbon', 'Dust', 'Sulfate', 'Validation'};
set(0, 'DefaultFigureRenderer', 'painters');
figure('position', [0, 0, 1500, 600])
% plot AOT retrieval
for i=1:4
    subplot(1,6,i)
    imshow(cat(3, ones(size(prediction,1), size(prediction,2))*1, ones(size(prediction,1), size(prediction,2))*0.5, ones(size(prediction,1), size(prediction,2))*0.5))
    hold on
    h = imshow(squeeze(prediction(:,:,i)), 'DisplayRange', [0 0.5]);
    colormap(gca, flipud(brewermap([],'GnBu'))) %uses Brewermap for colors
    set(h, 'AlphaData', ~isnan(squeeze(prediction(:,:,i))));
    
    title({string(title_i(i)), strcat(string(MED(i)), '{ \pm }', string(STD(i)))},'fontsize', 15)
    colorbar('fontsize', 12)
    hold on
end

% plot output of novelty detection
subplot(1,6,5)
imshow(squeeze(prediction(:,:,5))*1.2,'DisplayRange', [0 1]);
colormap(gca, flipud(brewermap([],'YlGn')))
colorbar('fontsize', 12)
title({'Novelty', 'Detection'},'fontsize', 15)

% plot RGB image of scene for reference
load(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/GeneratedData/AVIRIS_20160',scene,'.mat'), 'inputAVIRIS')
X = reshape(inputAVIRIS(:,[45 25 5]), 504, 100,3); %pick wavelength for Red Green Blue
subplot(1,6,6)
% Normalize radiance for RGB plot
meanX = mean(mean(X));
stdX = mean(std(X));
Y = (X-(meanX-3*stdX))./(meanX+3*stdX);
% plot RGB image
imshow(Y)
axis on
xticks([1 100])
xticklabels({'0 m', '500 m'});
yticks(1:100:501)
yticklabels({'0 m', '500 m', '1 km', '1.5 km', '2 km', '2.5 km'});
title({'True Color', 'Image'},'fontsize', 15)

