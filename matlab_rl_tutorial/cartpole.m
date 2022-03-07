% Set up the simulation environment
env = rlPredefinedEnv("CartPole-Discrete");
env.PenaltyForFalling = -10;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
rng(0)
plot(env)
