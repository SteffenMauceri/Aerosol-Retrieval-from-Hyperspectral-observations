% 3-albedo method that allows to calculate radiances for different surface
% types
%
% input: calculated radiance for a constant surface albedo of 0, 0.5 and 1.0 
%
% output: spherical albedo, two way transmittance and path radiance
%
% For details "Verhoef, W. and Bach, H.: Simulation of hyperspectral and 
% directional radiance images using coupled biophysical and atmospheric radiative
% transfer models, Remote Sens. Environ., 87(1), 23?41, 2003."
%
% Steffen Mauceri, Jan 2018


function [sphericalalbd, trans2, path] = albedo3_4(radiance)
    
rad0av = radiance(:,1:3:end); %path radiance
rad5av = radiance(:,2:3:end);
rad1av = radiance(:,3:3:end);

f=(rad1av-rad0av)./(2.*(rad5av-rad0av));

sphericalalbd = 2./(1./(f-1.)+2.);

trans2 = (rad1av-rad0av).*(1.-sphericalalbd);

path = rad0av;
