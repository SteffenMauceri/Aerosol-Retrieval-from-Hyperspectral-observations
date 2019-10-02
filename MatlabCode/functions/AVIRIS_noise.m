% AVIRIS-NG instrument noise model
%
% input:    radiance, wavelength, how much noise (AVIRIS-NG == 1)
%
% output:   radiance peturbed with AVIRIS-NG equivalent noise
%
% From: "Optimal estimation for imaging spectormeter atmospheric
% correction" https://doi.org/10.1016/j.rse.2018.07.003

function [noisy_rad] = AVIRIS_noise(rad, wl, multiplier)

load('data/AVIRIS-NG_noise.mat', 'AVIRISNGnoise')

%calc coefficents
a = interp1(AVIRISNGnoise(:,1),AVIRISNGnoise(:,2), wl);
b = interp1(AVIRISNGnoise(:,1),AVIRISNGnoise(:,3), wl);
c = interp1(AVIRISNGnoise(:,1),AVIRISNGnoise(:,4), wl);

%calc standard deviation of noie equivalent
noise_std = a .* (b .* rad).^0.5 + c;

%apply noise to radiance
noisy_rad = rad + randn(size(noise_std)) .* noise_std*multiplier;