% Interpolate between radiative transfer calculations
% 
% Input: calculated radiances for constant surface albedo(0, 0.5, 1.0), SZA, ground-elevation, ground-sensor distance 
% 
% Output: interpolated and radiances and constant surface albedo, SZA, ground elevation, ground-sensor distance 
%
% note: 
% -radiance is convolved to AVIRIS-NG spectral resolution
%
% Steffen Mauceri, Jan 2018

clear
 
aerosol = {'2_bahrain_sand_', '2_stratsulf_tropcarbon_', '2_stratsulf_indoex_', '2_alexander_tropcarbon_'};

rng(12345)%use a seed to reproduce
randNo = rand(100000,2);

for m=aerosol
    load(strcat('/Users/stma4117/Studium/LASP/Hyper/Sensitivity/ProcessedData/',string(m),'.mat'))
    % radiance = radiance [wavelength, calculation_i]
    % wl = wavelength in nm
    % values = [[AOT; surface albedo; SZA; ground-sensor distance; ground elevation],caclculation_i]
    m
    
    %normalize for SZA
    radiance = radiance./cosd(values(3,:));
    %convert SZA to cos
    values(3,:) = cosd(values(3,:));
    %normalize for Sun-Earth distance
    day = 1; %day of year of radiative transfer calculations
    d = 1 - 0.01672*cosd(0.9856*(day - 4));
    radiance = radiance.*d^2;
    
    
    %% Start Interpolate
    j=1;
    k=1;
    p=1;
    radiance_ = zeros(21331,9000);  %[wavlength, samples]
    values_= zeros(5,9000);         %[values(SZA,...), samples]
    
    while k<=3000
        %pick a random sample
        id1 = round(randNo(p,1)*(length(values)-4),0)+4;
        while ~mod(id1,3)==0 && id1>=0
            id1 =id1-1;
        end

        %find a second sample for interpolation
        id2 = round(randNo(p,2)*(length(values)-4),0)+4;
        while ~mod(id2,3)==0 && id2>=0
            id2 =id2-1;
        end
        
        % check that the two found samples are neighbours. E.g. only
        % interpolate between radiance calculation where the difference in
        % cos(SZA) is less than 0.16
        if abs(values(:,id1)-values(:,id2)) <= [0.08;0;0.16;1;1] & values(1,id1)< 0.18 & values(1,id2)< 0.18 | ...
                abs(values(:,id1)-values(:,id2)) <= [0.18;0;0.16;1;1] & values(1,id1)> 0.08 & values(1,id2)> 0.08 | ... %last value was for water
                abs(values(:,id1)-values(:,id2)) <= [0.3;0;0.16;1;1] & values(1,id1)>0.49 & values(1,id2)>0.49
            
            % interpolate radiance and values between two radiative transfer calculations 
            radiance_(:,j:j+2) = (radiance(:,id1-2:id1)*randNo(k,1)+radiance(:,id2-2:id2)*randNo(k,2))/sum(randNo(k,:));
            values_(:,j:j+2) = (values(:,id1-2:id1)*randNo(k,1)+values(:,id2-2:id2)*randNo(k,2))/sum(randNo(k,:));

            k=k+1;
            j = j+3;
        end
        p=p+1;
    end
    values = values_;
    radiance = radiance_;
    
    save(strcat('/Users/stma4117/Studium/LASP/Hyper/Sensitivity/ProcessedData/',string(m),'interp.mat'), 'values', 'radiance', 'wl' )
end
%execute RadianceGenarator4 that will add different surface spectra to calculated radiances
RadianceGenerator4 %
