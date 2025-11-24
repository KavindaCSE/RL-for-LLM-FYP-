def generate_dummy_data(num_samples=100):
    import numpy as np
    import pandas as pd

    # Generate random observations and rewards
    observations = np.random.rand(num_samples, 4)  # Assuming 4 features for the observation
    rewards = np.random.rand(num_samples)  # Random rewards

    # Create a DataFrame
    dummy_data = pd.DataFrame({
        'observation_1': observations[:, 0],
        'observation_2': observations[:, 1],
        'observation_3': observations[:, 2],
        'observation_4': observations[:, 3],
        'reward': rewards
    })

    return dummy_data

if __name__ == "__main__":
    # Generate and save dummy data to a CSV file
    dummy_data = generate_dummy_data()
    dummy_data.to_csv('data/dummy_data.csv', index=False)