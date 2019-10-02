% Use "3-albedo method" to add different surface spectra/types to our
% radiative transfer calculations
% 
% Input: (interpolated, normalized for sun-earth distance and SZA) radiances for three aerosol types 
%        from radiative transfer calculations for constant surface albedo
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
    load(strcat('Preprocessing/',string(m),'interp.mat'))
    % radiance = radiance [wavelength, calculation_i]
    % wl = wavelength in nm
    % values = [[AOT; surface albedo; SZA; ground-sensor distance; ground elevation],caclculation_i]
    
    wl_aerosol = wl(1:end-1);
    radiance(end, :) = [];

    %load surface spectra
    load('data/SurfaceReflectance')
    wl_bands = size(surface,2);
    
    %bring on same wavelength grid
    radiance_ = zeros(wl_bands,9000);
    for i=1:9000
        radiance_(:,i) = interp1(wl_aerosol, radiance(:,i), wl);
    end
    clear radiance

    %use three albedo Method to calculate spherical albedo, two way
    %transmittance and path radiance
    [sphericalalbd, trans2, path] = albedo3(radiance_);

    GroundTruth = zeros(99000,4);
    Modeledradiance = zeros(99000, wl_bands);
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

    save(strcat('Preprocessing/',string(m),'interp_surf.mat'),'GroundTruth', 'Modeledradiance', 'wl')
end

% execute input_output_mix to make final training dataset
make_training_set