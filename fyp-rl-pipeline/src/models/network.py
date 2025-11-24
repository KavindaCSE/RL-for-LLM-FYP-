class DQNNetwork:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        
        return model

    def forward(self, x):
        import torch
        x = torch.FloatTensor(x)
        return self.model(x)

    def get_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values.argmax().item()