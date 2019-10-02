% Make training dataset for Neural Network
% mix calculated radiances from different aerosol types and apply AVIRIS-NG equivalent noise
%
% input:    (interpolated, normalized for sun-earth distance and SZA) radiances for three aerosol types 
%            from radiative transfer calculations
%           
% output:   Neural Network inputs and outputs for training
%
%
% Steffen Mauceri, Jan 2018

clear
rng(12345)

%load calculated radiances for three aerosol types
load('Preprocessing/2_alexander_tropcarbon_interp_surf.mat')
carbon = Modeledradiance; 
load('Preprocessing/2_stratsulf_indoex_interp_surf.mat')
sulfate = Modeledradiance;
load('Preprocessing/2_bahrain_sand_interp_surf.mat')
sand = Modeledradiance;

%% mix aerosols randomly
rounds = 3; % how often do we go through our calculated radiances
samples=99000; %how many samples do we generate per round

%initialize some variables
mix = zeros(samples*rounds,length(wl));
GroundTruth = repmat(GroundTruth,rounds , 1);
GroundTruth(:,5:7) = zeros(samples*rounds,3);

for j = 0:rounds-1
    for i=1:samples
        weight = (rand(3,1));     % amount of aerosol Type in final radiance
        mix_i = [carbon(i,:);sand(i,:);sulfate(i,:)].*weight; 
        mix(i+j*samples,:) = sum(mix_i,1)/sum(weight); % generate mixed radiance
        %update ground truth values and account for 0.006 AOT of sulfate in
        % the Stratosphere
        if GroundTruth(i+j*samples,1) >= 0.05
            GroundTruth(i+j*samples,5:7) = ((weight/sum(weight))*GroundTruth(i+j*samples,1)) + [0; 0; 0.006];
        else
            GroundTruth(i+j*samples,5:7) = ((weight/sum(weight))*GroundTruth(i+j*samples,1)) + [0; 0; 0.006*GroundTruth(i+j*samples,1)/0.05];
        end
    end
end

% apply AIVIRS-NG instrument noise
[mix] = AVIRIS_noise(mix', wl, 1)';

%% make input and output for Neural Network
input = [mix GroundTruth(:,2) GroundTruth(:,3:4)];
output = GroundTruth(:,[1 5 6 7]);

%% remove samples for testing for final performance analysis
rng(12345)%use a seed to reproduce our random process
id = randi(length(output), [10000, 1]);%get 10000 samples
input_test = input(id,:);
output_test = output(id,:);


%remove test samples from datset
input(id,:)=[]; 
output(id, :)=[];


%save everything
save('Preprocessing/test_set.mat', 'output_test','input_test', 'wl')
dlmwrite('Preprocessing/output.csv', output, 'delimiter', ',', 'precision', 9)
dlmwrite('Preprocessing/input.csv', input, 'delimiter', ',', 'precision', 9)
