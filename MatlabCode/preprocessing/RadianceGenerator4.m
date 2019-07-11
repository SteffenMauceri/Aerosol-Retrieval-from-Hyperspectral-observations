% Use "3-albedo method" to add different surface spectra/types to our
% radiative transfer calculations
% 
% Input: (interpolated, normalized for sun-earth distance and SZA) radiances for three aerosol types 
%        from radiative transfer calculations for constant surface albedos
% 
% Output: radiances for three aerosol types from radiative transfer
% calculations for different surface spectra.
%
%
% Steffen Mauceri, Jan 2018

clear 
% load radiances for the three sets of aerosols
aerosol = {'2_bahrain_sand_', '2_alexander_tropcarbon_', '2_stratsulf_indoex_'};
for m=aerosol
    load(strcat('/Users/stma4117/Studium/LASP/Hyper/Sensitivity/ProcessedData/',string(m),'interp.mat'))

    wl_aerosol = wl(1:end-1);
    radiance(end, :) = [];

    %load surface spectra
    load('/Users/stma4117/Studium/LASP/Hyper/NN/GeneratedData/SurfaceGenerated5_05')
    wl_bands = size(surface,2);
    
    %bring on same wavelength grid
    radiance_ = zeros(wl_bands,9000);
    for i=1:9000
        radiance_(:,i) = interp1(wl_aerosol, radiance(:,i), wl);
    end
    clear radiance

    %use three albedo Method to calculate spherical albedo, two way
    %transmittance and path radiance
    [sphericalalbd, trans2, path] = albedo3_4(radiance_);

    GroundTruth = zeros(99000,4);
    Modeledradiance = zeros(99000, wl_bands);
    Surface_Type = type(1:99000,:);
    values_ = values([1 3 4 5], 1:3:end);

    i=0;
    for k=1:round(100000/3000,0)
        for j=1:3000
            i=i+1;
            GroundTruth(i,:) = values_(:,j)';
            % add surface spectra to radiance calculations
            Modeledradiance(i,:) = path(:,j)' + (trans2(:,j)'.*surface(i,:))./(1-surface(i,:).*sphericalalbd(:,j)'); 
        end
    end

    save(strcat('/Users/stma4117/Studium/LASP/Hyper/NN/GeneratedData/RadianceGenerated5_05',string(m),'interp.mat'),'GroundTruth', 'Modeledradiance', 'wl', 'Surface_Type')
end

% execute input_output_mix to make final training dataset
make_input_output_mix
