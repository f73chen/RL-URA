%% Closer to the RL side
    % Similar to OsimEnv / OsimEnvRSI / L2RunEnvRSI / L2RunEnvMod
    % Defines step function
    % Defines reward and stopping condition
    % Defines an observation as a function of the state
classdef Exo_Env < handle
    properties
        model
        ref

        dt_0
        T
    end
    methods
        % Constructor
        function env = Exo_Env()
            load href_new;
            env.ref = href;
            env.T = 60;
            env.dt_0 = 0.002;

            env.LoadModel(env.ref, env.T, env.dt_0);
        end

        % Reset model parameters and load the model into the env
        function LoadModel(env, href, T, dt_0)
            env.model = Exo_Model(href, T, dt_0);
        end

        % Calculate the overall reward weight as a function of time
        % It's more important to follow the trajectory at some parts
        % Ex. stance phase >> mid swing
        % @@@ TODO: find a function to describe the weight
        function timing_factor = GetTimingFactor(env, time)
            timing_factor = 1;
        end

        % Get the current reward for the state
        % Start with mimic reward based on trajectory (location of back)
        % @@@ TODO: add individual weights for each joint
        function reward = GetReward(env, time)
            obs = env.GetObservation();
            ref_obs = env.ref.back(time);
            timing_factor = env.GetTimingFactor(time);
            reward = timing_factor * (ref_obs - obs).^2;
        end
        
        % Whether to end the simulation
        % Stop if pelvis < 0.6m --> pos.ground_pelvis.y < 0.6
        function done = IsDone(env)
            desc = env.GetStateDesc();
            done = desc(3) < 0.6;
        end

        % Pass up the state description from the model
        function desc = GetStateDesc(env)
            desc = env.model.GetStateDesc();
        end

        % Convert the state description to something usable by RL agent
        % Depends on what variables the controller cares about
        function obs = GetObservation(env)
            desc = env.GetStateDesc();
            obs = desc(2:19);   % Only observe joint locations
        end

        % Reset the environment
        function obs = Reset(env)
            env.model.Reset();
            obs = env.GetObservation();
        end

        % Calculate the next state based on actions predicted by RL
        % @@@ TODO: propagate action back to controller
            % Update model based on action (exo input)
        function [obs, reward, done] = Step(env, t, action)
            % compute simulation time
            env.model.Simulate(t);
            obs = env.GetObservation();
            reward = env.GetReward(t);
            done = env.IsDone();
        end

        % Step through the whole sequence
        function Simulate(env)
            for i = 1:(env.T/env.dt_0)+1
                t = (i-1) * env.dt_0; 
                [obs, reward, done] = env.Step(t, [0 0 0]);
                disp(reward);
            end
        end
    end
end