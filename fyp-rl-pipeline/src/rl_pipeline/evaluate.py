def evaluate_agent(agent, env, num_episodes=100):
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        
        total_rewards.append(total_reward)
    
    average_reward = sum(total_rewards) / num_episodes
    return average_reward

if __name__ == "__main__":
    from src.envs.simple_env import SimpleEnv
    from src.agents.dqn_agent import DQNAgent
    
    env = SimpleEnv()
    agent = DQNAgent()
    
    average_reward = evaluate_agent(agent, env)
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")