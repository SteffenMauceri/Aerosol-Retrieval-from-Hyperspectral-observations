% Compare MODIS aerosol optical thickness (AOT) to AVIRIS-NG retrieved AOT
%
% input:    MODIS AOT for combined AOT
%           AVIRIS AOT prediction from neural network
% 
% output:   RMSE, Correlation and Plot of combined AOT
%
% Steffen Mauceri, Jan 2019

clear
%% load MODIS retrieved AOT
load('/Users/stma4117/Studium/LASP/Hyper/climatology/MODIS/MODIS.mat')

% remove values without AOT measurements and with quality flag 1,2
MODIS(MODIS(:,2) == -9999 | MODIS(:,4) ~= 3, :) = [];

% convert time to decimal year
start_time = datetime(1993, 01, 01,00,00,00);
MODIS(:,7) = decyear(start_time + MODIS(:,7)/86400);

%% load AVIRIS eval
%load date from AVIRIS flights
file = {'20160101.nc','20160102.nc','20160105.nc', '20160107.nc', '20160110.nc',...
    '20160126.nc','20160127.nc', '20160128.nc','20160129.nc','20160203.nc',...
    '20160204.nc','20160205.nc','20160208.nc','20160208.nc','20160210.nc',...
    '20160211.nc', '20160213.nc','20160221.nc','20160223.nc', ...
    '20160224.nc','20160303.nc'};
variable = {'iyear', 'imonth', 'idayofmonth'};

%concat year, month and day to date
for j=1:length(file)
    date(j,1:3) = [double(ncread(string(file(j)), char(variable(1)))) double(ncread(string(file(j)), ...
        char(variable(2)))) double(ncread(string(file(j)), char(variable(3))))];
end
%convert to decyear
DecYearAviris = decyear(datetime(date,'Format', 'yyyyMMdd'));

%load location of AVIRIS flights
load('/Users/stma4117/Studium/LASP/Hyper/climatology/AvirisLatLon_fine.mat')

%% find closest MODIS measurements in time and space 
for i=1:21 %number of flights
    %cacluate distance in space
    distA = ((LAT(i,:) - MODIS(:,5)).^2 + (LON(i,:) - MODIS(:,6)).^2).^(0.5);
    dist(i,:) = min(distA,[],2)';
    %calculate distace in time
    timedistA(i,:) = abs(DecYearAviris(i) - MODIS(:,7));
end

% define what the maximum distance in time and space is we compare AOT
% retrievals
time_cutoff = 0.0027*1.5; %(1 days = 0.0027)
space_cutoff = 0.2; %(110km = 1deg)

% calculate MODIS averag AOT and standard deviation for AIVIRS flights that
% are within the set distance and time
for i=1:21
    nel(i) = numel(MODIS(dist(i,:)<space_cutoff & timedistA(i,:)<time_cutoff,2));
    MODIS_AOT(i) = mean(MODIS(dist(i,:)<space_cutoff & timedistA(i,:)<time_cutoff,2))/1000;
    MODIS_AOT_STD(i) = std(MODIS(dist(i,:)<space_cutoff & timedistA(i,:)<time_cutoff,2))/1000;
end
mean(nel)   %check that we have enough MODIS measurements to compare to. Otherwise increase cutoff values
min(nel)    %check that we have enough MODIS measurements to compare to
%
%% load AVIRIS eval
Aviris_eval %program that will load and preprocess AVIRIS derived AOT

%% Plot
figure
errorbar(MED(:,1), MODIS_AOT, STD(:,1),'.k', 'horizontal')
hold on
errorbar(MED(:,1), MODIS_AOT, MODIS_AOT_STD,'.k', 'vertical')
plot(MED(:,1), MODIS_AOT,'*b', 'MarkerSize', 6)
x = [0 1];
plot(x,x, '--')
ylim([0 1])
xlim([0 1])
hold off
legend('1\sigma AVIRIS-NG','1\sigma MODIS','MODIS Aqua/Terra', 'x=y') 
xlabel('AVIRIS-NG AOT [\tau_{aer}]')
ylabel('MODIS Terra/Aqua AOT [\tau_{aer}]')

%% Calculate some statistics: Correlation, P-value, RMSE
[MODIS_AOT_all, pval] = corr(MED(:,1), MODIS_AOT')
immse(MED(:,1), MODIS_AOT')^0.5 %RMSE