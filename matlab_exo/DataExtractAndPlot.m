close all
load ('C:\Users\smshusht\OneDrive - University of Waterloo\ExoSim\OpenSim-Based\Projects\RAL_Traj\RAL_v04_sims\DataBase\FF_new');
load ('C:\Users\smshusht\OneDrive - University of Waterloo\ExoSim\OpenSim-Based\Projects\RAL_Traj\RAL_v04_sims\DataBase\grfref_new');
t = sub.outPut(:,1);
x = sub.outPut(:,5:11);
f_i = sub.outPut(:,end-7:end);
r =zeros(size(t,1),7);
f_e=zeros(size(t,1),7);
f_f =  zeros(size(t,1),7);
f_l =  zeros(size(t,1),7);
f_l(:,[3,6]) = sub.outPut(:,end-8:end-7);
f_exo = zeros(size(t,1),7);

r_exo = zeros(size(t,1),7);

% for m=1:7
%     r(:,m) = HC.(motorSet{m}).signals(:,2);
%     f_e(:,m) = HC.(motorSet{m}).signals(:,5);
%     f_f(:,m) = FF.(motorSet{m})((t+phase));
%     f_exo(:,m) = EC.(motorSet{m}).signals(:,4);
%     r_exo(:,m) = EC.(motorSet{m}).signals(:,2);
%     e_exo(:,m) = EC.(motorSet{m}).signals(:,3);
% end



p_m = FF.m(t+phase);
p_x = FF.x(t+phase);
p_y = FF.y(t+phase);

g_xr = grfref.xr(t+phase);
g_yr = grfref.yr(t+phase);
g_xl = grfref.xl(t+phase);
g_yl = grfref.yl(t+phase);



grf.foot_r=sub.outPut(:,20);
grf.calcn_r=sub.outPut(:,22);
grf.toes_r=sub.outPut(:,24);

grf.foot_l=sub.outPut(:,26);
grf.calcn_l=sub.outPut(:,28);
grf.toes_l=sub.outPut(:,30);

grf.foot_ry=sub.outPut(:,32);
grf.calcn_ry=sub.outPut(:,33);
grf.toes_ry=sub.outPut(:,34);

grf.foot_ly=sub.outPut(:,35);
grf.calcn_ly=sub.outPut(:,36);
grf.toes_ly=sub.outPut(:,37);

s = sub.outPut(:,4);


pos.toes_r.x  = sub.outPut(:,12);
pos.toes_r.y  = sub.outPut(:,13);
pos.calcn_r.x = sub.outPut(:,14);
pos.calcn_r.y = sub.outPut(:,15);
pos.toes_l.x  = sub.outPut(:,16);
pos.toes_l.y  = sub.outPut(:,17);
pos.calcn_l.x = sub.outPut(:,18);
pos.calcn_l.y = sub.outPut(:,19);

% Graph2