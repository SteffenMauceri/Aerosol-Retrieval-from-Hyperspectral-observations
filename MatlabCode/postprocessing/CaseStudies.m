% Compare retrieval accuracy for different amounts of noise in the data and 
% different sampling of the available wavelengths
%
% input:    Predictions and GroundTruth values for TestCases
% 
% output:   Heatmap (plot) of accuracy
%
% Steffen Mauceri, April 2019


clear
k=0;
for n= [0, 1, 3, 9]             % amount of AVIRIS-NG equivalent noise
    for s=[319, 107, 36, 12, 4] % number of wavlength bands
        k=k+1;
        name = strcat('4.4_05_128x32_noise_' , string(n) , '_sampling_' , string(s) , '_reg5000_ind_relu');
        load(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/trained/prediction' , name,'.mat'))
        
        prediction_c(:,k*3-2:k*3) = prediction(:,1:3);
        target_c(:,k*3-2:k*3) = target(:,1:3);
        c(:,k*3-2:k*3) = repmat([n; s], 1,3);
    end
end

%calculate difference between predictions and GroundTruth
diff = nanstd(abs(prediction_c - target_c));    

% remove predictions where GroundTruth label has AOT < 0.3 to check how the
% neural network performs for low AOT
cutoff = 0.3;
prediction_c(target_c>cutoff) = nan;
target_c(target_c>cutoff) = nan;
%calculate difference between predictions and GroundTruth for AOT < 0.3
diff2 = nanstd(abs(prediction_c - target_c));

%calculate baseline values for comparison. See paper for details
prediction_r = rand(10000,60)*cutoff/3;%guess between 0 and 1*cutoff and devide by 3
diff_random = mean(nanstd(abs((prediction_r) - target_c)));
prediction_r = rand(10000,60)*cutoff/3;%know AOT and devide by 3
diff_random2 = mean(nanstd(abs(repmat(sum(target_c(:,1:3),2),1,60)/3 - target_c)));    %abs

%% plot
for i=1:3
    conf_i(:,:,i) = reshape(diff(i:3:end), 5,4);
    conf_i_2(:,:,i) = reshape(diff2(i:3:end), 5,4);
end
n_i(:,:) = reshape(c(1,1:3:end), 5,4);
s_i(:,:) = reshape(c(2,1:3:end), 5,4);

labelsx={'0','1','3','9'};
labelsy={'319','107','36','12' ,'4'};
title_i={'Carbon','Dust','Sulfate'};

% combine all aerosols
conf_mean = mean(conf_i,3);
conf_i(:,:,1) = conf_mean;

for i=1:3
    for j=1:2
        subplot(3,2,j+2*(i-1))
        if j==1
            h = heatmap(labelsx, labelsy, conf_i(:,:,i));
            h.Title = strcat(title_i(i), ' (High AOT [0 to 1])');
            h.ColorLimits = [0.015 0.094];
            h.Colormap = brewermap([],'PuBuGn');
            h.YLabel = 'Number of Wavelength Bands';
            
        else
            h = heatmap(labelsx, labelsy, conf_i_2(:,:,i));
            h.Title = strcat(title_i(i),' (Low AOT [0 to 0.3])');
            h.ColorLimits = [0.015 0.037];
            h.Colormap = brewermap([],'PuBuGn');
        end
        
        if j+2*(i-1) > 4
            h.XLabel = 'AVIRIS-NG equiv. noise';
           
        end
        
        h.ColorbarVisible = 'off';
        h.CellLabelFormat = '%.2f';
    end
end



