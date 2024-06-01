from reinforcement_learning import ReinforcementLearning

class MetaLearning:
    def __init__(self, rl_agent):
        self.rl_agent = rl_agent

    def meta_learn(self, meta_epochs=10, episodes_per_epoch=100):
        for meta_epoch in range(meta_epochs):
            for _ in range(episodes_per_epoch):
                self.rl_agent.train(episodes=1)
            self.rl_agent.agent.model.save(f'meta_epoch_{meta_epoch}.h5')
