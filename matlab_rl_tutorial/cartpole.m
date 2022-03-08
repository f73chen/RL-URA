%% Set up the simulation environment
env = rlPredefinedEnv("CartPole-Discrete");
env.PenaltyForFalling = -10;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
rng(0)
plot(env)

%% Create Actor-Critic Agent
% Create the critic
criticNetwork = [
    featureInputLayer(4,'Normalization','none','Name','state')  % 4 observations
    fullyConnectedLayer(1,'Name','CriticFC')];
criticOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);
critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

% Create the actor
actorNetwork = [
    featureInputLayer(4,'Normalization','none','Name','state')
    fullyConnectedLayer(2,'Name','fc')  % 2 possible actions: +10 or -10
    softmaxLayer('Name','actionProb')];
actorOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);
actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);

% Merge the actor-critic agent
agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32, ...
    'DiscountFactor',0.99);
agent = rlACAgent(actor,critic,agentOpts);

%% Train the Agent
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',500,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480,...
    'ScoreAveragingWindowLength',10); 
trainingStats = train(agent,env,trainOpts);

%% Validate performance via simulation
simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)