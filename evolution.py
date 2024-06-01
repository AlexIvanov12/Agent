import random
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, losses, metrics
from tensorflow import keras
from agent import DQNAgent

class AlgorithmEvolution:
    def __init__(self, env, state_size, action_size, strategy):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.strategy = strategy

    def create_model(self, layer_sizes):
        model = models.Sequential()
        model.add(layers.Dense(layer_sizes[0], input_dim=self.state_size, activation='relu'))
        for size in layer_sizes[1:]:
            model.add(layers.Dense(size, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def evaluate_model(self, model, episodes=10):
        total_reward = 0
        for _ in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            while not done:
                action = np.argmax(model.predict(state)[0])
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                total_reward += reward
                state = next_state
        return total_reward / episodes

    def genetic_algorithm(self, population_size=10, generations=10):
        population = [random.choices(range(10, 101, 10), k=random.randint(1, 5)) for _ in range(population_size)]
        
        for generation in range(generations):
            scores = []
            for individual in population:
                model = self.create_model(individual)
                score = self.evaluate_model(model)
                scores.append((score, individual))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            best_individuals = scores[:population_size // 2]
            
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(best_individuals, 2)
                child = self.crossover(parent1[1], parent2[1])
                self.mutate(child)
                new_population.append(child)
            
            population = new_population
            print(f"Generation {generation + 1}/{generations}, Best Score: {scores[0][0]}")
        
        best_individual = scores[0][1]
        return self.create_model(best_individual)

    def crossover(self, parent1, parent2):
        child = []
        for i in range(min(len(parent1), len(parent2))):
            child.append(random.choice([parent1[i], parent2[i]]))
        return child

    def mutate(self, individual):
        mutation_rate = 0.1
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] += random.choice(range(-10, 11, 10))

    def evolve_strategy_params(self):
        params = self.strategy.strategy_params
        params = self.mutate_strategy_params(params)
        self.strategy.update_strategy_params(params)
        return params
    

    def mutate_strategy_params(self, params):
        mutation_rate = 0.1
        for param in params:
            if random.random() < mutation_rate:
                params[param] += random.uniform(-0.1, 0.1) * params[param]
        return params
