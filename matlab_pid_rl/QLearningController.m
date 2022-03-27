classdef QLearningController < handle
    properties
        gait_split          % How many components to split each gait cycle into
        kp                  % P-stiffness
        kd                  % D-stiffness

        q_table             % Matrix of states and actions
        local_rew_factor    % Weight of reward in one phase
        mean_rew_factor     % Weight of the episode's mean/cumulative error
        alpha               % Learning rate
        gamma               % Discount factor: long-term if close to 1
    end
    methods
        % Object constructor
        function controller = QLearningController()
            controller.gait_split = 20;
            controller.kp = zeros([gait_split, 1]);
            controller.kd = zeros([gait_split, 1]);

            controller.q_table = zeros([gait_split, 6]);
            controller.local_rew_factor = 1;
            controller.mean_rew_factor = 2;
            controller.alpha = 1e-3;
            controller.gamma = 0.8;
        end

        % Optional: calculate error from state and reference trajectory
        function reward = calculate_reward(controller, state, ref)
            % Parameters:
            %   state: knee angle etc.
            %   ref: reference knee angle
            % Output:
            %   abs_error: absolute tracking error for each phase

            reward = -abs(state - ref);
            reward = controller.local_rew_factor .* reward + ...
                     controller.mean_rew_factor   * mean(reward);
        end

        % Update kp and kd based on reward
        % Iterate until kp and kd converge (small abs_error)
        function update(controller, reward)
            % Read and write kp & kd in-place
            % Parameters:
            %   reward: negative tracking error for the last gait cycle


        
        end

        % Get kp and kd values
        function [kp, kd] = get_gains(controller)
            kp = controller.kp;
            kd = controller.kd;
        end
    end
end