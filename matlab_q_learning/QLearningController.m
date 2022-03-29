% Next step:
    % Dynamic learning rate proportional to step number
    % Dynamic exploration chance

classdef QLearningController < handle
    properties
        gait_split          % How many components to split each gait cycle into
        actions             % Previous actions from the last episode
        kp                  % P-stiffness
        kd                  % D-stiffness

        q_table             % Matrix of states and actions
        local_rew_factor    % Weight of reward in one phase
        mean_rew_factor     % Weight of the episode's mean/cumulative error
        alpha               % Learning rate
        exploration         % Chance to try sub-optimal actions
    end
    methods
        % Object constructor
        function q = QLearningController()
            q.gait_split = 20;
            q.actions = zeros([q.gait_split, 1]) - 1;
            q.kp = zeros([q.gait_split, 1]) + 1;
            q.kd = zeros([q.gait_split, 1]) + 1;

            q.q_table = randn(q.gait_split, 5);
            q.local_rew_factor = 1;
            q.mean_rew_factor = 2;
            q.alpha = 0.5;
            q.exploration = 0.1;
        end

        % Optional: calculate error from state and reference trajectory
        function reward = calculate_reward(q, state, ref)
            % Parameters:
            %   state: knee angle etc.
            %   ref: reference knee angle
            % Output:
            %   abs_error: absolute tracking error for each phase

            reward = -abs(state - ref);
            reward = q.local_rew_factor .* reward + ...
                     q.mean_rew_factor   * mean(reward);
        end

        % Update Q-table based on reward and previous actions
        % Update internal kp and kd values
        function [kp, kd] = update(q, state, ref)
            % Parameters:
            %   reward: negative tracking error for the last gait cycle
            
            % Calculate state reward
            reward = q.calculate_reward(state, ref);

            % Update Q-table if choices were made during the last episode
            if q.actions(1) > -1
                for i = 1:q.gait_split
                    % Bellman equation without discount term
                    q.q_table(i, q.actions(i)) = (1 - q.alpha) * q.q_table(i, q.actions(i)) + q.alpha * reward(i);
                end
            end

            % Choose the next round of choices and remember decisions
            for j = 1:q.gait_split
                % Exploration vs. exploitation
                if rand() > q.exploration
                    [val, q.actions(j)] = max(q.q_table(j, :));
                else
                    q.actions(j) = randi(5);
                end
                switch q.actions(j)
                    case 1      % Increase kp by 5%
                        q.kp(j) = q.kp(j) * 1.05;
                    case 2      % Decrease kp by 5%
                        q.kp(j) = q.kp(j) * 0.95;
                    case 3      % Increase kd by 5%
                        q.kd(j) = q.kd(j) * 1.05;
                    case 4      % Decrease kd by 5%
                        q.kd(j) = q.kd(j) * 0.95;
                    % Otherwise no change
                end
            end
            kp = q.kp;
            kd = q.kd;
        end
    end
end