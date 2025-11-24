# main.py

import gym
from agents.dqn_agent import DQNAgent
from envs.simple_env import SimpleEnv
from rl_pipeline.train import train
from rl_pipeline.evaluate import evaluate

def main():
    # Initialize the environment
    env = SimpleEnv()

    # Initialize the DQN agent
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    # Train the agent
    train(agent, env, episodes=1000)

    # Evaluate the agent
    evaluate(agent, env, episodes=100)

if __name__ == "__main__":
    main()