% clear all;close all;clc
% load nwe_D
% DataExtractAndPlot
dataNames = {'Healthy_no_exo','Healthy','D','C','B'};

for jj=1:5
    load(dataNames{jj})
figure()
HSR_time = [];
TOR_time = [];

for i=2:size(grf.calcn_r,1)
    if -grf.foot_r(i-1)==0 && -grf.foot_r(i)>.0
    HSR_time = [HSR_time; i];
    end
    
    if -grf.foot_r(i-1)>0 && -grf.foot_r(i)==0
    TOR_time = [TOR_time; i];
    end
end

HSL_time = [];
TOL_time = [];
for i=2:size(grf.calcn_l,1)
    if -grf.foot_l(i-1)==0 && -grf.foot_l(i)>0
    HSL_time = [HSL_time; i];
    end
    
    if -grf.foot_l(i-1)>0 && -grf.foot_l(i)==0
    TOL_time = [TOL_time; i];
    end
end
HSL_time = HSL_time*0.01;
HSR_time = HSR_time*0.01;

TOL_time = TOL_time*0.01;
TOR_time = TOR_time*0.01;

% TOR_time(1)=[];

DR = (TOR_time(1:end-1)-HSR_time(1:end-1))./diff(HSR_time);
DL = (TOL_time-HSL_time(1:end-1))./diff(HSL_time);

str_right = diff(HSR_time);
str_left = diff(HSL_time);

ax = size(str_right); bx = size(str_left);
cx = min(ax,bx);

stride_time.(dataNames{jj}) = [str_right(1:cx,1) str_left(1:cx,1)];

scatter(1:size(diff(HSL_time)),diff(HSL_time),'fill'); 
hold on;
scatter(1:size(diff(HSR_time)),diff(HSR_time),'fill')
xlabel('Step#')
ylabel('Stride time [s]')
legend('Left', 'Right')
figure()

scatter(1:size(DR),DR*100,'fill'); 
hold on;
scatter(1:size(DL),DL*100,'fill')

xlabel('Step#')
ylabel('Stance percentage [%]')
legend('Reft', 'Light')
end