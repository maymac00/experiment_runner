from typing import Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO
from experiment_runner.ExperimentManager import ExperimentManager


class CartpoleHPT(ExperimentManager):

    def build_model(self, env, model_args: Dict[str, Any]) -> Any:
        model_args["batch_size"] = model_args["n_steps"] * 5
        model = PPO("MlpPolicy", env, device="cpu", **model_args)
        return model

    def build_env(self, args: Dict[str, Any]) -> Any:
        return gym.make("CartPole-v1")

    def evaluate(self, model, env, exp_args: Dict[str, Any]) -> float:
        obs, _ = env.reset()
        rew = 0
        for ep in range(exp_args["n_eval_episodes"]):
            for step in range(500):
                action, _ = model.predict(obs)
                obs, reward, tr, tm, _ = env.step(action)
                rew += reward
                if tr or tm:
                    obs, _ = env.reset()
                    break
        return rew / exp_args["n_eval_episodes"]


if __name__ == '__main__':
    # em = CartpoleHPT("test_study", "cartpole_example", hp_path="hp_search.yaml")
    em = CartpoleHPT("test", "cartpole_example", hp_path="hp_search.yaml", tb_log=True, normalize_reward=True)
    em.optimize(10)
