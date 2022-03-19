%% Closer to the RL side
    % Similar to OsimEnv / OsimEnvRSI / L2RunEnvRSI / L2RunEnvMod
    % Defines step function
    % Defines reward and stopping condition
    % Defines an observation as a function of the state
classdef Exo_Env < handle
    properties
        model
        visualize
        ref
    end
    methods
        % Constructor
        function env = Exo_Env(visualize)
            load href_new;
            env.visualize = visualize;
            env.ref = href;
            env.LoadModel();
        end

        % Reset model parameters and load the model into the env
        function LoadModel(env)
            env.model = Exo_Model(env.ref);
        end

        % Get the current reward for the state
        % Start with mimic reward based on trajectory (location of back)
        function reward = GetReward(env, time)
            obs = env.GetObservation();
            ref_obs = href.back(time);
            reward = (ref_obs - obs)^2;
        end
        
        % Whether to end the simulation
        % Stop if pelvis < 0.6m --> pos.ground_pelvis.x < 0.6
        function done = IsDone(env)
            desc = env.GetStateDesc();
            done = desc(2) < 0.6;
        end

        % Pass up the state description from the model
        function desc = GetStateDesc(env)
            desc = env.model.GetStateDesc();
        end

        % Convert the state description to something usable by RL agent
        % Depends on what variables the controller cares about
        function obs = GetObservation(env)
            desc = env.GetStateDesc();
            obs = desc(2:19);   % Only all joint locations
        end

        % Reset the environment
        function obs = Reset(env)
            env.model.Reset();
            obs = env.GetObservation();
        end

        % Calculate the next state based on actions predicted by RL
        function [obs, reward, done] = Step(env, action)
            % @@@ Actuate muscles & use model.sub.step()
            obs = env.GetObservation();
            reward = env.GetReward();
            done = env.IsDone();
        end
    end
end