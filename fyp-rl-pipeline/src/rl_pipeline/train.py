import numpy as np
import gym
from agents.dqn_agent import DQNAgent
from envs.simple_env import SimpleEnv
from utils.replay_buffer import ReplayBuffer

def train_agent(episodes, max_steps, agent, env, replay_buffer):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            agent.train(replay_buffer)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    env = SimpleEnv()
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    replay_buffer = ReplayBuffer(buffer_size=10000)
    
    train_agent(episodes=1000, max_steps=200, agent=agent, env=env, replay_buffer=replay_buffer)