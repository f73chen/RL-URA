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
        function model = Exo_Model(ref)
            % Constants and params:
            model.motorSet = {'hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l','back'};
            model.muscleSet = {'hamstrings_r','bifemsh_r','glut_max_r','iliopsoas_r','rect_fem_r','vasti_r','gastroc_r','soleus_r','tib_ant_r','hamstrings_l','bifemsh_l','glut_max_l','iliopsoas_l','rect_fem_l','vasti_l','gastroc_l','soleus_l','tib_ant_l'};
            model.d2r = pi/180;
    
            % Simulation parameters
            model.dt_0 = 0.002;   % controller update referesh rate
            model.updT = 0.01;    % save and GUI referesh rate
            model.phase = 0;      % initial phase of the motion. phase = 0 starts from the heel strike of the right leg
            model.phi = 0.61595;  % right-left phase difference (it has to always be equal to 1.2319/2 seconds)
            
            model.exo_enable = 1; % add exo to the simulation
            model.mus_enable = 0; % add muscles to the simulations
    
            % Controller and force parameters
            model.pelvis_stiffness = .0*[500 0 0]; model.pelvis_damping = .0*[50 0 0];  % tilt,x,y
            model.init_speed_gain = 1;                                                % healthy human only
            model.p_gain = 3*[[10 2 5] 1*[10 2 5] 15]*60;                             % internal controller P gain
            model.d_gain = 2*[.3  0.1 0.1 .3 0.1 0.1 1]*60;                           % internal controller D gain
            model.sat = 1*[500 500 500 500 500 500 500];                              % internal controller saturation level
    
            % Set initial states
            model.ref = ref; % get the reference trajectory 
            for m = 1:6
                if ismember(m,[4,5,6])  % for the left leg we have phi instead of phase 
                    % because we are using the same trajectory for the left leg as we
                    % are using for the right leg. We only shift it by phi.
            
                    model.x0.(model.motorSet{m}) = model.d2r*model.ref.(model.motorSet{m-3})(0+model.phase+model.phi);                    
                    model.v0.(model.motorSet{m}) = model.d2r*differentiate(model.ref.(model.motorSet{m-3}),0+model.phase+model.phi);
                    model.v0.(model.motorSet{m}) = model.init_speed_gain*model.v0.(model.motorSet{m});   % initial speed gain is for having a stable transient form standing to walking
                else    % for the right leg
                    model.x0.(model.motorSet{m}) = model.d2r*model.ref.(model.motorSet{m})(model.phase);
                    model.v0.(model.motorSet{m}) = model.d2r*differentiate(model.ref.(model.motorSet{m}),model.phase);
                    model.v0.(model.motorSet{m}) = model.init_speed_gain*model.v0.(model.motorSet{m});
                end
            end
    
            model.x0.back = -0.42; % initial torso joint angle
            model.v0.back = 0;   % initial torso joint velocity
    
            % set the muscle activation and the external forces equal to zero
            for m = 1:7
                model.f_exo.(model.motorSet{m}) = 0;
            end
            for m = 1:18
                model.activation.(model.muscleSet{m}) = 0;
            end

            % Get a fresh OpenSim model
            model.Reset();
        end

        % Resets the OpenSim model's position
        % Reset every variable changed during simulation
        function Reset(model)
            model.T = 60;           % simulation period
            % get instance form Human_Exo class and call it sub
            model.sub = Human_Exo_v02(model.x0,model.v0,model.updT,model.p_gain,model.d_gain,model.sat,model.pelvis_stiffness,model.pelvis_damping,model.exo_enable, model.mus_enable);
        end

        % Generate the next set of limb positions
        function [s_r,s_l,r] = GaitGen(model,t,href,motorSet,phase,phi)
            model.T = 1.2319;
            t_l = t+phase+phi;
            t_r = t+phase;
            s_r = mod(t_r,model.T)/model.T;
            s_l = mod(t_l,model.T)/model.T;
            r = zeros(7,1);
            for m = 1:7
                if ismember(m,[4,5,6])
                    r(m,1)=deg2rad(href.(motorSet{m-3})(s_l*model.T));
                else
                    r(m,1)=deg2rad(href.(motorSet{m})(s_r*model.T));
                end
            end
            r(7,1) = -0.4;
        end

        % Run and open the OpenSim walking simulation
        function Simulate(model)
            for i = 1:(model.T/model.dt_0)+1
                % compute simulation time
                t = (i-1) * model.dt_0; 
                % compute each leg's gait cycle and 
                [s_l,s_r,r_human] = GaitGen(model, t, model.ref, model.motorSet, model.phase, model.phi); 
                % update the model for one step by passing the reference trajectory to
                % the system. We pass the reference trajectory to the system because
                % the iternal controllers of the system needs that. It actually sets the
                % rest length for each joints spring. 
                model.sub.step(t+model.dt_0, model.f_exo, model.activation, r_human);
            end
        end

        % Return model state variables
        function desc = GetStateDesc(model)
            desc = model.sub.getOutPut();
        end
    end
end
