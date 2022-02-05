clear all;clc;close all;
load href_new;

%% Constants and Params:
motorSet={'hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l','back'};
muscleSet={'hamstrings_r','bifemsh_r','glut_max_r','iliopsoas_r','rect_fem_r','vasti_r','gastroc_r','soleus_r','tib_ant_r','hamstrings_l','bifemsh_l','glut_max_l','iliopsoas_l','rect_fem_l','vasti_l','gastroc_l','soleus_l','tib_ant_l'};
d2r = pi/180;
%% simulation parameters
for i=1:1 
    dt_0=0.002;     % controller update referesh rate
    updT = 0.01;    % save and GUI referesh rate
    phase = 0;      % initial phase of the motion. phase = 0 starts from the heel strike of the right leg
    phi = 0.61595;  % right-left phase difference (it has to be always equal to 1.2319/2)
    
    exo_enable = 1; % add exo to the simulation
    mus_enable = 0; % add muscles to the simulations
    T=60;           % simulation period
end
%% Controller and Force parameters

for i=1:1 %human parameters
    pelvis_stiffness = .0*[500 0 0]; pelvis_damping = .0*[50 0 0];  % tilt,x,y
    init_speed_gain =1;                                             % healthy human only
    p_gain = 3*[[10 2 5] 1*[10 2 5] 15]*60;                         % internal controller P gain
    d_gain = 2*[.3  0.1 0.1 .3 0.1 0.1 1]*60;                       % internal controller D gains
    sat = 1*[500 500 500 500 500 500 500];                          % internal controller saturation level
end

%% initial states setting
ref = href; % get the reference trajectory 
for m=1:6
    if ismember(m,[4,5,6])  % for the left leg we have phi instead of phase 
        % because we are using the same trajectory for the left leg as we
        % are using for the right leg. We only shift it by phi.

        x0.(motorSet{m}) = d2r*ref.(motorSet{m-3})(0+phase+phi);                    
        v0.(motorSet{m}) = d2r*differentiate(ref.(motorSet{m-3}),0+phase+phi);
        v0.(motorSet{m}) = init_speed_gain*v0.(motorSet{m});   % initial speed gain is for having a stable transient form standing to walking
        
    else    % for the right leg
        x0.(motorSet{m}) = d2r*ref.(motorSet{m})(phase);
        v0.(motorSet{m}) = d2r*differentiate(ref.(motorSet{m}),phase);
        v0.(motorSet{m}) = init_speed_gain*v0.(motorSet{m});
    end
end
x0.back=-0.42; % initial torso joint angle
v0.back = 0;   % initial torso joint velocity
%% initiating the simulation
% get instance form Human_Exo class and call it sub
sub = Human_Exo_v02(x0,v0,updT,p_gain,d_gain,sat,pelvis_stiffness,pelvis_damping,exo_enable, mus_enable);
% set the muscle activation and the external forces equal to zero
for m=1:7
        f_exo.(motorSet{m}) = 0;
end
for m=1:18
        activation.(muscleSet{m}) = 0;
end
%% Simulation loop 
   
for i=1:(T/dt_0)+1
    % comute simulation time
    t = (i-1)*dt_0; 
    % compute each leg's gait cycle and 
    [s_l,s_r,r_human] = GaitGen(t,href,motorSet,phase,phi); 
    % update the modle for one step by passing the reference trajectory to
    % the system. We pass the reference trajectory to the system because
    % the iternal controllers of the system needs that. It is actually sets the
    % rest length for each joints spring. 
    sub.step(t+dt_0,f_exo,activation,r_human);
    
end

%% Functions
function [s_r,s_l,r] = GaitGen(t,href,motorSet,phase,phi)
    T=1.2319;
    t_l = t+phase+phi;
    t_r = t+phase;
    s_r = mod(t_r,T)/T;
    s_l = mod(t_l,T)/T;
    r=zeros(7,1);
    for m=1:7
        if ismember(m,[4,5,6])
            r(m,1)=deg2rad(href.(motorSet{m-3})(s_l*T));
        else
            r(m,1)=deg2rad(href.(motorSet{m})(s_r*T));
        end
    end
    r(7,1) = -0.4;
end
