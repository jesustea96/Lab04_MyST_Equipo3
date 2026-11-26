"""Entrenamiento con Stable-Baselines3 por fold, configuracion y semilla."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from code.env import BTCEnv

ALGOS = {"PPO": PPO, "DQN": DQN}


def hiperparametros(algo: str, gamma: float) -> dict:
    """gamma = 0 es la eleccion principal: en cada punto de decision el agente esta plano y el mercado es exogeno,
    asi que la accion no cambia la distribucion del siguiente estado y el problema de Bellman se reduce a
    max_a E[C | S, a]. Con gamma > 0 el descuento seria por decision y no por hora (sesgo semi-Markov)."""
    if algo == "PPO":
        return dict(learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10, gamma=gamma,
                    gae_lambda=0.95, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]))
    return dict(learning_rate=5e-4, buffer_size=50_000, learning_starts=2_000, batch_size=128, gamma=gamma,
                train_freq=4, target_update_interval=1_000, exploration_fraction=0.3,
                exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64]))


def entrenar(algo: str, datos_train, cols, semilla: int, pasos: int, env_kw: dict, gamma: float = 0.0):
    """Entrena un modelo y devuelve (modelo, curva) donde curva = (pasos acumulados, recompensa por episodio)."""
    set_random_seed(semilla)
    env = Monitor(BTCEnv(datos_train, cols, **env_kw))
    env.reset(seed=semilla)
    env.action_space.seed(semilla)
    modelo = ALGOS[algo]("MlpPolicy", env, seed=semilla, verbose=0, device="cpu", **hiperparametros(algo, gamma))
    modelo.learn(total_timesteps=pasos)
    curva = (np.cumsum(env.get_episode_lengths()), np.array(env.get_episode_rewards()))
    return modelo, curva
