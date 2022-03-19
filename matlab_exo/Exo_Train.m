%% Trains an agent based on Exo_Env
    % Defines the model/network type and parameters
    % Optional callbacks
    % Train and save the model

clear all;clc;close all;
load href_new;

%% Test: simulate the environment after turning it into a class
model = Exo_Model(href);
model.Simulate();