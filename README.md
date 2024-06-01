Project name: Automated trading system based on agent DQN
The purpose of the project
The project is designed to develop and implement an automated trading system on the stock market using machine learning methods, in particular DQN (Deep Q-Network) agent. The system is able to automatically make trading decisions, improve its algorithms and adapt to market conditions without user intervention.

Basic functions
Data collection and preparation:

The system receives historical trade data from Interactive Brokers (IB) using an API.
The data is cleaned and prepared for further use.
DQN agent:

The DQN (Deep Q-Network) agent is used for learning and making trading decisions.
The agent model includes a neural network that learns from past trading data.
Trading strategy:

The system uses various trading strategies and technical indicators to make decisions about buying or selling stocks.
Evolution of algorithms:

The use of genetic algorithms to improve the architecture and parameters of the neural network.
The system automatically adjusts the parameters of the trading strategy based on the training results.
Reinforcement learning:

Using reinforcement learning methods to improve agent performance in real market conditions.
Target engagement:

Rapid adaptation of an agent's learning strategy to improve performance.
Monitoring and auditing:

The system continuously monitors the performance of the model and conducts audits to identify and correct problems.

Main components
IBDataLoader: A class for loading and preparing data from Interactive Brokers.
DQNAgent: A class that implements a DQN agent.
TradingStrategy: A class that defines a trading strategy based on technical indicators.
AlgorithmEvolution: A class for the evolution of neural network architecture and parameters.
ReinforcementLearning: A class for reinforcement learning.
MetaLearning: A class for rapidly adapting an agent's learning strategy.
MonitoringAudit: A class for monitoring performance and auditing the model.
TradingEnv: A class that defines the reinforcement learning environment.
