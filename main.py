import threading
import tkinter as tk
from tkinter import messagebox
from ib_data_loader import IBDataLoader
from trading_strategy import TradingStrategy
from agent import DQNAgent
from evolution import AlgorithmEvolution
from reinforcement_learning import ReinforcementLearning
from meta_learning import MetaLearning
from monitoring_audit import MonitoringAudit
from trading_env import TradingEnv

def main():
    symbol = 'AAPL'
    exchange = 'SMART'
    end_date = ''  # Поточна дата та час за замовчуванням
    state_size = 4  # Відповідає кількості індикаторів у TradingEnv
    action_size = 2
    
    # Завантаження і підготовка даних з Interactive Brokers
    ib_data_loader = IBDataLoader()
    prepared_data = ib_data_loader.get_prepared_data(symbol, exchange, end_date)
    
    # Створення агента
    agent = DQNAgent(state_size, action_size)
    
    # Торгова стратегія
    strategy = TradingStrategy(agent.model)
    
    # Навчання з підкріпленням
    env = TradingEnv(prepared_data)
    rl_agent = ReinforcementLearning(agent, env)
    rl_agent.train()
    
    # Мета-залучення
    meta_learning = MetaLearning(rl_agent)
    meta_learning.meta_learn()
    
    # Еволюція алгоритмів
    evolution = AlgorithmEvolution(env, state_size, action_size, strategy)
    agent.model = evolution.genetic_algorithm()
    evolved_params = evolution.evolve_strategy_params()
    strategy.update_strategy_params(evolved_params)
    
    # Моніторинг і аудит
    monitoring_audit = MonitoringAudit(rl_agent, evolution)
    monitoring_audit.monitor_performance()
    monitoring_audit.audit_model(threshold=0.8)

if __name__ == "__main__":
    main()
