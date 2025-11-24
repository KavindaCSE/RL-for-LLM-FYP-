class SimpleEnv:
    def __init__(self):
        self.state = None
        self.done = False

    def reset(self):
        self.state = self._initialize_state()
        self.done = False
        return self.state

    def step(self, action):
        reward = self._take_action(action)
        self.done = self._check_done()
        self.state = self._get_next_state()
        return self.state, reward, self.done

    def render(self):
        print(f"Current State: {self.state}")

    def _initialize_state(self):
        return 0  # Example initial state

    def _take_action(self, action):
        return 1  # Example reward for taking an action

    def _check_done(self):
        return False  # Example condition for episode termination

    def _get_next_state(self):
        return self.state + 1  # Example transition to the next state