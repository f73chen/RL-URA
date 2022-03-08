%% Higher-level interaction with the OpenSim model
    % Similar to OsimModel
    % Implements functions to get and set muscles/joints etc.
classdef Exo_Model < handle
    properties
        motorSet
        muscleSet
        d2r

        dt_0
        updT
        phase
        phi

        exo_enable
        mus_enable
        T

        pelvis_stiffness
        pelvis_damping
        init_speed_gain
        p_gain
        d_gain
        sat

        ref
        x0
        v0

        sub
        f_exo
        activation
    end
    methods
        % Constructor
        function env = Exo_Model()
            load href_new;

            % Constants and params:
            env.motorSet = {'hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l','back'};
            env.muscleSet = {'hamstrings_r','bifemsh_r','glut_max_r','iliopsoas_r','rect_fem_r','vasti_r','gastroc_r','soleus_r','tib_ant_r','hamstrings_l','bifemsh_l','glut_max_l','iliopsoas_l','rect_fem_l','vasti_l','gastroc_l','soleus_l','tib_ant_l'};
            env.d2r = pi/180;
    
            % Simulation parameters
            env.dt_0 = 0.002;   % controller update referesh rate
            env.updT = 0.01;    % save and GUI referesh rate
            env.phase = 0;      % initial phase of the motion. phase = 0 starts from the heel strike of the right leg
            env.phi = 0.61595;  % right-left phase difference (it has to always be equal to 1.2319/2 seconds)
            
            env.exo_enable = 1; % add exo to the simulation
            env.mus_enable = 0; % add muscles to the simulations
            env.T = 60;           % simulation period
    
            % Controller and force parameters
            env.pelvis_stiffness = .0*[500 0 0]; env.pelvis_damping = .0*[50 0 0];  % tilt,x,y
            env.init_speed_gain = 1;                                                % healthy human only
            env.p_gain = 3*[[10 2 5] 1*[10 2 5] 15]*60;                             % internal controller P gain
            env.d_gain = 2*[.3  0.1 0.1 .3 0.1 0.1 1]*60;                           % internal controller D gain
            env.sat = 1*[500 500 500 500 500 500 500];                              % internal controller saturation level
    
            % Set initial states
            env.ref = href; % get the reference trajectory 
            for m = 1:6
                if ismember(m,[4,5,6])  % for the left leg we have phi instead of phase 
                    % because we are using the same trajectory for the left leg as we
                    % are using for the right leg. We only shift it by phi.
            
                    env.x0.(env.motorSet{m}) = env.d2r*env.ref.(env.motorSet{m-3})(0+env.phase+env.phi);                    
                    env.v0.(env.motorSet{m}) = env.d2r*differentiate(env.ref.(env.motorSet{m-3}),0+env.phase+env.phi);
                    env.v0.(env.motorSet{m}) = env.init_speed_gain*env.v0.(env.motorSet{m});   % initial speed gain is for having a stable transient form standing to walking
                else    % for the right leg
                    env.x0.(env.motorSet{m}) = env.d2r*env.ref.(env.motorSet{m})(env.phase);
                    env.v0.(env.motorSet{m}) = env.d2r*differentiate(env.ref.(env.motorSet{m}),env.phase);
                    env.v0.(env.motorSet{m}) = env.init_speed_gain*env.v0.(env.motorSet{m});
                end
            end
    
            env.x0.back = -0.42; % initial torso joint angle
            env.v0.back = 0;   % initial torso joint velocity

            % get instance form Human_Exo class and call it sub
            env.sub = Human_Exo_v02(env.x0,env.v0,env.updT,env.p_gain,env.d_gain,env.sat,env.pelvis_stiffness,env.pelvis_damping,env.exo_enable, env.mus_enable);
    
            % set the muscle activation and the external forces equal to zero
            for m = 1:7
                    env.f_exo.(env.motorSet{m}) = 0;
            end
            for m = 1:18
                    env.activation.(env.muscleSet{m}) = 0;
            end
        end

        function [s_r,s_l,r] = GaitGen(env,t,href,motorSet,phase,phi)
            env.T = 1.2319;
            t_l = t+phase+phi;
            t_r = t+phase;
            s_r = mod(t_r,env.T)/env.T;
            s_l = mod(t_l,env.T)/env.T;
            r = zeros(7,1);
            for m = 1:7
                if ismember(m,[4,5,6])
                    r(m,1)=deg2rad(href.(motorSet{m-3})(s_l*env.T));
                else
                    r(m,1)=deg2rad(href.(motorSet{m})(s_r*env.T));
                end
            end
            r(7,1) = -0.4;
        end

        function simulate(env)
            for i = 1:(env.T/env.dt_0)+1
                % compute simulation time
                t = (i-1)*env.dt_0; 
                % compute each leg's gait cycle and 
                [s_l,s_r,r_human] = GaitGen(env,t,env.ref,env.motorSet,env.phase,env.phi); 
                % update the model for one step by passing the reference trajectory to
                % the system. We pass the reference trajectory to the system because
                % the iternal controllers of the system needs that. It actually sets the
                % rest length for each joints spring. 
                env.sub.step(t+env.dt_0,env.f_exo,env.activation,r_human);
            end
        end
    end
end
