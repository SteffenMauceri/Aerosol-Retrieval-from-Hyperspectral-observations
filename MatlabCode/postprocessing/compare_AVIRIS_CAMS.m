% Compare CAMS aerosol model aerosol optical thickness (AOT) to AVIRIS-NG
% retrieved AOT
%
% input:    CAMS AOT reanalysis for various aerosol types as netcdf
%           AVIRIS AOT prediction from neural network
% 
% output:   RMSE for every aerosol type and plot
%
% Steffen Mauceri, July 2019


%
clear

%% load CAMS aerosol model AOT
path = "/Volumes/T5/AVIRISraw/";
file = ["netcdfJan2016.nc","netcdfFeb2016.nc","netcdfMar2016.nc"];

lon = ncread(path+file(1), 'longitude');
lat = ncread(path+file(1), 'latitude');
time = [];
for i=1:length(file) 
    l = length(double(ncread(path+file(i), 'time')));
    
    dust_550(:,:,length(time)+1:length(time)+l) = ncread(path+file(i), 'duaod550'); % dust AOT at 550 nm
    carbon_550(:,:,length(time)+1:length(time)+l) = ncread(path+file(i), 'omaod550');% brown carbon AOT at 550 nm
    sulfate_550(:,:,length(time)+1:length(time)+l) = ncread(path+file(i), 'suaod550');% sulfate AOT at 550 nm
    
    time(length(time)+1:length(time)+l) = double(ncread(path+file(i), 'time')); %hours since 1900-01-01 00:00:00.0'
end
imagesc(carbon_550(:,:,2)); %sanity check

% convert time to decimal year
start_time = datetime(1900, 01, 01,00,00,00);
time = decyear(start_time + time'/24);

% remove times without AOT
carbon_550(:,:,mean(mean(dust_550))==0)=[];
sulfate_550(:,:,mean(mean(dust_550))==0)=[];
time(mean(mean(dust_550))==0)=[];
dust_550(:,:,mean(mean(dust_550))==0)=[];

%% load AVIRIS eval time, lat and lon
%get time
file = {'20160101.nc','20160102.nc','20160105.nc', '20160107.nc', '20160110.nc',...
    '20160126.nc','20160127.nc', '20160128.nc','20160129.nc','20160203.nc',...
    '20160204.nc','20160205.nc','20160208.nc','20160208.nc','20160210.nc',...
    '20160211.nc', '20160213.nc','20160221.nc','20160223.nc', ...
    '20160224.nc','20160303.nc'};
variable = {'lataviris', 'lonaviris', 'iyear', 'imonth', 'idayofmonth'};
%get flight date
for j=1:length(file)
    date(j,1:3) = [double(ncread(string(file(j)), char(variable(3)))) double(ncread(string(file(j)), ...
        char(variable(4)))) double(ncread(string(file(j)), char(variable(5))))];
end
%add UTC hour of flights to date
date(:,4) = [7,7,5,7,8,6,9,5,6,7,8,6,7,7,6,6,6,7,6,6,6]';
date(:,5:6) = zeros(21,2);

DecYearAviris = decyear(datetime(date,'Format', 'yyyyMMddH,M,S'));

load('/Users/stma4117/Studium/LASP/Hyper/climatology/AvirisLatLon_fine.mat')
LAT = mean(LAT,2);
LON = mean(LON,2);

%% find closest CAMS measurements in time and space
for i=1:21
    [~,a] = min(abs(LON(i) - lon)); %find closest longitude
    [~,b] = min(abs(LAT(i) - lat)); %find closest latitude
    [~,c] = min(abs(DecYearAviris(i) - time)); %find closest time
    
    %save carbon, dust, sulfate AOT for closest match
    carbon(i) = carbon_550(a,b,c); 
    dust(i) = dust_550(a,b,c);
    sulfate(i) = sulfate_550(a,b,c);
    
    %calculate standard deviation for neigbouring points in space and time
    carbon_std(i) = std(reshape(carbon_550(a-1:a+1,b:b+2,c:c+2),27,1));
    dust_std(i) = std(reshape(dust_550(a-1:a+1,b:b+2,c:c+2),27,1));
    sulfate_std(i) = std(reshape(sulfate_550(a-1:a+1,b:b+2,c:c+2),27,1));
end

%% load AVIRIS eval
Aviris_eval %MED(:,2:4) = AOT carbon, AOT dust, AOT sulfate

CAMS = [carbon;dust;sulfate];
CAMS_std = [carbon_std;dust_std;sulfate_std];

title_i = ["Carbon","Dust","Sulfate"];
x = [0 1];

%% plot comparison
% color = brewermap(21,'RdYlGn');
figure('position', [10, 10, 750, 200])
for i=1:3
    subplot(1,3,i)
    errorbar(MED(:,1+i), CAMS(i,:), STD(:,1+i),'.k', 'horizontal')
    hold on
    errorbar(MED(:,1+i), CAMS(i,:), CAMS_std(i,:),'.k', 'vertical')
    for j=1:21
        if j==5
            plot(MED(j,1+i), CAMS(i,j),'sr','MarkerSize', 6)
        elseif j==11
            plot(MED(j,1+i), CAMS(i,j),'dg','MarkerSize', 6)
        else
            plot(MED(j,1+i), CAMS(i,j),'*b','MarkerSize', 6)
        end
    end
    plot(x,x, '--')
    hold off
    ylim([0 0.5]); xlim([0 0.5])
    title = title_i(i);
    xlabel('AVIRIS-NG AOT [\tau_{aer}]')
    ylabel('CAMS AOT [\tau_{aer}]')
    
    %calculate standard deviation of difference between CAMS and AVIRIS for every aerosol type
    std(abs(MED(:,1+i) - CAMS(i,:)')) 
end
legend('1\sigma AVIRIS-NG','1\sigma CAMS','CAMS', 'x=y')
