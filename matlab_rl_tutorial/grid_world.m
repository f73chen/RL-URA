env = rlPredefinedEnv("BasicGridWorld");    % Create the grid environment
env.ResetFcn = @() 2;   % Set to position [2, 1] (aka state 2)
rng(0)                  % Fix random seed

% Create Q table using observations and actions from the grid world
qTable = rlTable(getObservationInfo(env), getActionInfo(env));
qRep = rlQValueRepresentation(qTable, getObservationInfo(env), getActionInfo(env));
qRep.Options.LearnRate = 1;

% Create Q-learning agent using the table
agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = .04;
qAgent = rlQAgent(qRep, agentOpts);

% Train the agent
trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 50;
trainOpts.MaxEpisodes = 200;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 11;
trainOpts.ScoreAveragingWindowLength = 30;

% Run training iterations
trainingStats = train(qAgent, env, trainOpts);

% Visualize training results
plot(env)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;
sim(qAgent, env)