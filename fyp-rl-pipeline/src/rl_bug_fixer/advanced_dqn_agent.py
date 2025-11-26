"""
Advanced DQN Agent with Dueling Architecture, Prioritized Experience Replay, and Noisy Networks
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Register buffers for noise
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.sample_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def sample_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
    
    def forward(self, x):
        if self.training:
            self.sample_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return torch.nn.functional.linear(x, weight, bias)


class DuelingQNetwork(nn.Module):
    """Dueling Architecture: separates value and advantage streams"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_noisy=True):
        super(DuelingQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy
        
        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        if use_noisy:
            self.value_head = NoisyLinear(hidden_dim // 2, 1)
        else:
            self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        if use_noisy:
            self.advantage_head = NoisyLinear(hidden_dim // 2, action_dim)
        else:
            self.advantage_head = nn.Linear(hidden_dim // 2, action_dim)
    
    def forward(self, state):
        features = self.features(state)
        
        # Value stream
        value = self.value_stream(features)
        value = self.value_head(value)
        
        # Advantage stream
        advantage = self.advantage_stream(features)
        advantage = self.advantage_head(advantage)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with Sum Tree"""
    def __init__(self, capacity=1000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling weight
        self.beta_increment = 0.001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # New experience gets max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha / (priorities ** self.alpha).sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on temporal difference errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class AdvancedDQNAgent:
    """Advanced DQN Agent with multiple improvements"""
    
    PROMPT_TEMPLATES = [
        "Identify the root cause of the error: {error}",
        "Fix the bug that causes: {error}",
        "Explain why this error occurs: {error} and provide a solution",
        "Debug this code. Error: {error}",
        "What causes {error}? Provide a fix with error handling",
        "Resolve the following error: {error} with proper validation",
        "Analyze the error {error} and suggest improvements",
        "Troubleshoot {error} with detailed explanation",
    ]
    
    def __init__(self, state_dim=100, action_dim=8, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, use_dueling=True, use_noisy=True, use_per=True):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.use_noisy = use_noisy
        self.use_per = use_per
        
        # Q-Networks with Dueling Architecture
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim=256, use_noisy=use_noisy)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim=256, use_noisy=use_noisy)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for robustness
        
        # Prioritized Replay Buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=2000, alpha=0.6, beta=0.4)
        else:
            self.replay_buffer = self._create_simple_buffer()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        print(f"🤖 Advanced DQN Agent initialized on {self.device}")
        print(f"   - Dueling Architecture: {use_dueling}")
        print(f"   - Noisy Networks: {use_noisy}")
        print(f"   - Prioritized Replay: {use_per}")
    
    def _create_simple_buffer(self):
        """Fallback to simple replay buffer"""
        class SimpleBuffer:
            def __init__(self):
                self.buffer = deque(maxlen=2000)
            
            def push(self, state, action, reward, next_state, done):
                self.buffer.append((state, action, reward, next_state, done))
            
            def sample(self, batch_size):
                batch = random.sample(self.buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                weights = torch.ones(batch_size)
                indices = list(range(batch_size))
                return (
                    torch.tensor(np.array(states), dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.long),
                    torch.tensor(rewards, dtype=torch.float32),
                    torch.tensor(np.array(next_states), dtype=torch.float32),
                    torch.tensor(dones, dtype=torch.float32),
                    weights,
                    indices
                )
            
            def update_priorities(self, indices, td_errors):
                pass
            
            def __len__(self):
                return len(self.buffer)
        
        return SimpleBuffer()
    
    def encode_error(self, error_text):
        """Convert error text to state vector"""
        np.random.seed(hash(error_text) % (2**32))
        state = np.random.randn(100).astype(np.float32)
        return state
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if not self.use_noisy and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action
    
    def generate_prompt(self, error):
        """Generate prompt using learned policy"""
        state = self.encode_error(error)
        action = self.select_action(state)
        prompt = self.PROMPT_TEMPLATES[action].format(error=error)
        return prompt, action, state
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        """Train on batch with advanced techniques"""
        if len(self.replay_buffer) < batch_size:
            return 0
        
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Double DQN: use current network to select, target to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute weighted loss
        td_errors = (q_values - q_targets).detach().cpu().numpy()
        loss = (weights * self.loss_fn(q_values, q_targets)).mean()
        
        # Update priorities (if using PER)
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.001
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_statistics(self):
        """Get agent statistics"""
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device),
            'use_noisy': self.use_noisy,
            'use_per': self.use_per
        }