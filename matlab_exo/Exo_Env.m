%% Closer to the RL side
    % Similar to OsimEnv / OsimEnvRSI / L2RunEnvRSI / L2RunEnvMod
    % Defines step function
    % Defines reward and stopping condition
    % Defines an observation as a function of the state
classdef Exo_Env < handle
    properties
        model
        visualize
    end
    methods
        % Constructor
        function env = Exo_Env(visualize)
            env.visualize = visualize;
            env.LoadModel();
        end

        % Reset model parameters and load the model into the env
        function LoadModel(env)
            env.model = Exo_Model();
        end

        % Get the current reward for the state
        % Start with mimic reward based on trajectory
        function reward = GetReward()

        end
        
        % Whether to end the simulation
        % Stop if pelvis < 0.6m
        function done = IsDone()
            
        end

        % Convert state to agent-usable observations
        function obs = GetObservation(env)

        end

        % Reset the environment
        function obs = Reset(env)
            env.model.reset();
            obs = env.GetObservation();
        end

        % Calculate the next state based on actions
        function [obs, reward, done] = Step(env, action)
            % @@@ Actuate muscles & use model.sub.step()
            obs = env.GetObservation();
            reward = env.GetReward();
            done = env.IsDone();
        end
    end
end