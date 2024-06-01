import random
import numpy as np
from agent import DQNAgent

class ReinforcementLearning:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train(self, episodes=1000, batch_size=32):
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.agent.state_size])
            for time in range(500):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.agent.state_size])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"Episode: {e}/{episodes}, Score: {reward}, Epsilon: {self.agent.epsilon}")
                    break
                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)
            if e % 100 == 0:
                self.agent.epsilon *= self.agent.epsilon_decay
