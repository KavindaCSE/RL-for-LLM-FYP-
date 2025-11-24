# Reinforcement Learning Pipeline

This project implements a reinforcement learning (RL) pipeline using a Deep Q-Network (DQN) agent. The goal is to provide a structured approach to training and evaluating RL agents in a simple environment.

## Project Structure

```
fyp-rl-pipeline
├── src
│   ├── main.py                # Entry point of the application
│   ├── envs
│   │   └── simple_env.py      # Defines a simple RL environment
│   ├── agents
│   │   └── dqn_agent.py       # Implements the DQN agent
│   ├── models
│   │   └── network.py         # Defines the neural network architecture
│   ├── data
│   │   └── dummy_dataset.py    # Generates dummy data for testing
│   ├── rl_pipeline
│   │   ├── train.py           # Contains the training loop
│   │   ├── evaluate.py        # Evaluates the trained agent
│   │   └── utils.py           # Utility functions for training and evaluation
│   └── utils
│       └── replay_buffer.py    # Implements a replay buffer for experience storage
├── experiments
│   └── first_approach
│       ├── config.yaml        # Configuration settings for the experiment
│       └── run.sh             # Shell script to run the training and evaluation
├── notebooks
│   └── exploration.ipynb      # Jupyter notebook for exploratory data analysis
├── data
│   └── dummy_data.csv         # Dummy data in CSV format for testing
├── tests
│   └── test_pipeline.py       # Unit tests for the RL pipeline
├── requirements.txt           # Lists project dependencies
└── README.md                  # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd fyp-rl-pipeline
pip install -r requirements.txt
```

## Usage

To run the training and evaluation pipeline, navigate to the `experiments/first_approach` directory and execute the `run.sh` script:

```bash
cd experiments/first_approach
bash run.sh
```

## Components

- **Environment**: A simple RL environment defined in `src/envs/simple_env.py`.
- **Agent**: The DQN agent implemented in `src/agents/dqn_agent.py`.
- **Model**: The neural network architecture used by the agent, defined in `src/models/network.py`.
- **Training**: The training loop is handled in `src/rl_pipeline/train.py`.
- **Evaluation**: Performance evaluation of the agent is done in `src/rl_pipeline/evaluate.py`.
- **Utilities**: Various utility functions are provided in `src/rl_pipeline/utils.py` and `src/utils/replay_buffer.py`.

## Experiments

The `experiments/first_approach/config.yaml` file contains hyperparameters and settings for the first approach experiment. Modify this file to adjust the training configuration.

## Testing

Unit tests for the pipeline can be found in `tests/test_pipeline.py`. Run the tests to ensure that all components are functioning as expected.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.