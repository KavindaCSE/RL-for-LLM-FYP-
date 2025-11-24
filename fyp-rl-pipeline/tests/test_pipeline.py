import unittest
from src.envs.simple_env import SimpleEnv
from src.agents.dqn_agent import DQNAgent
from src.models.network import DQNNetwork
from src.utils.replay_buffer import ReplayBuffer

class TestRLPipeline(unittest.TestCase):

    def setUp(self):
        self.env = SimpleEnv()
        self.agent = DQNAgent(state_size=self.env.state_size, action_size=self.env.action_size)
        self.replay_buffer = ReplayBuffer(buffer_size=1000)

    def test_environment_reset(self):
        state = self.env.reset()
        self.assertEqual(len(state), self.env.state_size)

    def test_agent_action_selection(self):
        state = self.env.reset()
        action = self.agent.select_action(state)
        self.assertIn(action, range(self.agent.action_size))

    def test_replay_buffer(self):
        self.replay_buffer.add_experience((1, 2, 3, 4, 5))
        sample = self.replay_buffer.sample_batch(1)
        self.assertEqual(len(sample), 1)

    def test_network_forward_pass(self):
        model = DQNNetwork(state_size=self.env.state_size, action_size=self.env.action_size)
        state = self.env.reset()
        q_values = model.forward(state)
        self.assertEqual(len(q_values), self.env.action_size)

if __name__ == '__main__':
    unittest.main()