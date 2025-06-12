import os
from functools import partial
from typing import List, Dict, AnyStr, Any

from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, MoVecEnv, MoDummyVecEnv

from stable_baselines3.common.monitor import Monitor, MoMonitor
from stable_baselines3.common.vec_env import VecMonitor, MoVecMonitor

from experiment_runner.HyperparameterManager import HyperparameterManager
import optuna
import abc
import os

class ExperimentManager(abc.ABC):
    """
    Main class that uses all the other classes. Loads hyperparameters, runs the experiment, and saves the results in a structured way using optuna.
    :param name: Name of the experiment
    :param save_dir: Path to the directory where the results are saved by stable-baselines
    :param storage: Path to the database where the results are stored
    :param hp_path: Path to the YAML file with the hyperparameters
    """
    def __init__(self, name: str, save_dir : str, storage: str = None, hp_path: str = None, prune: bool = False, save_models: bool = True, tb_log: bool = False, normalize_reward : bool = False, n_objectives: int = 1, reload=False):
        self.name = name
        self.save_dir = save_dir
        self.prune = prune
        self.save_models = save_models
        self.tb_log = tb_log
        self.normalize_reward = normalize_reward
        self.n_objectives = n_objectives
        if storage is None:
            path = os.path.join(os.getcwd(), save_dir)
            print(f"Using sqlite storage in sqlite:///{path}/{name}.db")
            # create folder if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)
            storage = f"sqlite:///{path}/{name}.db"
        self.study = optuna.create_study(study_name=name + "_study", storage=storage, load_if_exists=True, direction="maximize")
        self.hp_manager = HyperparameterManager(hp_path, reload=reload)

    def get_callbacks(self, args : Dict[str, Dict[str, Any]]) -> List[MaybeCallback]:
        """
        Get the callbacks to be used in the experiment
        """
        return []

    @abc.abstractmethod
    def build_model(self, env, model_args: Dict[str, Any]) -> Any:
        """
        Build the model to be used in the experiment
        :param env: Environment to be used
        :param model_args: model hyperparameters
        :param policy_args: Dictionary with the policy hyperparameters
        """
        raise NotImplementedError("You need to implement the model building function")

    @abc.abstractmethod
    def build_env(self, args: Dict[str, Any]) -> Any:
        """
        Build the environment to be used in the experiment
        :param monitor: Add monitor for SB3
        :param args: Dictionary with the hyperparameters
        """
        raise NotImplementedError("You need to implement the environment building function")

    @abc.abstractmethod
    def evaluate(self, model, env, exp_args: Dict[str, Any]) -> float:
        """
        Evaluate the model in the environment
        :param model: Model to be evaluated
        :param env: Environment to be used
        :param exp_args: Dictionary with the hyperparameters
        """
        raise NotImplementedError("You need to implement the evaluation function")

    def objective(self, trial: optuna.trial.Trial, args: Dict[str, Dict[str, Any]]) -> float:
        """
        Objective function to be optimized
        :param trial: Optuna trial
        :param args: Dictionary with the hyperparameters
        """
        root_path = os.getcwd()
        trial_path = f"{self.save_dir}/{self.name}_{trial.number}"
        args["experiment"]["experiment_path"] = trial_path
        # cd trial path
        os.makedirs(trial_path, exist_ok=True)
        os.chdir(trial_path)

        def make_env():
            def _init():
                env = self.build_env(args["env"])
                env = Monitor(env)
                return env
            return _init

        def mo_make_env():
            def _init():
                env = self.build_env(args["env"])
                env = MoMonitor(env)
                return env
            return _init



        if self.n_objectives>1:
            if "n_envs" in args["experiment"] and  args["experiment"]["n_envs"] > 1:
                env = MoVecEnv([mo_make_env() for _ in range(args["experiment"]["n_envs"])])
            else:
                env = MoDummyVecEnv([mo_make_env()])
                env = MoVecMonitor(env)
        else:
            if "n_envs" in args["experiment"] and args["experiment"]["n_envs"] > 1:
                env = SubprocVecEnv([make_env() for _ in range(args["experiment"]["n_envs"])])
            else:
                env = DummyVecEnv([make_env()])
                env = VecMonitor(env)
        args["model"]["policy_kwargs"] = args["policy"]

        if "log_interval" not in args["experiment"].keys():
            args["experiment"]["log_interval"] = 50
        if self.tb_log:
            args["model"]["tensorboard_log"] = f"{os.curdir}"
            trial.set_user_attr(f"log_dir", trial_path)


        model = self.build_model(env, args["model"])

        if self.normalize_reward:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=getattr(model, "gamma"))

        callbacks = self.get_callbacks(args)

        if self.prune:
            #TODO: callbacks.append(optuna.integration.OptunaPruningCallback(trial, "reward"))
            pass
        if "n_timesteps" not in args["experiment"]:
            raise ValueError("Please provide n_timesteps as experiment hyperparameter")

        # Set parameters as trial's user attributes
        for arg in args.keys():
            for key, value in args[arg].items():
                try:
                    trial.set_user_attr(f"{arg}_{key}", value)
                except Exception as e:
                    continue

        model.learn(total_timesteps=args["experiment"]["n_timesteps"],
                    log_interval=args["experiment"]["log_interval"],
                    callback=callbacks,
                    tb_log_name=f"{self.name}_trial{trial.number}"
                    )

        if self.save_models:
            model.save("final_model")
            trial.set_user_attr(f"model_dir", trial_path)
        try:
            env.close()
        except Exception:
            pass
        value = self.evaluate(model, self.build_env(args["env"]), args["experiment"])

        os.chdir(root_path)
        return value

    def optimize(self, n_trials: int):
        """
        Run the optimization loop.
        :param n_trials: Number of trials to run
        """
        for i in range(n_trials):
            trial = self.study.ask()
            args = self.hp_manager.define_search_space(trial)

            try:
                value = self.objective(trial, args)
            except optuna.TrialPruned as e:
                self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            self.study.tell(trial, value)
            print(f"Run {i} completed with value {value}. Best value so far: {self.study.best_value} of trial {self.study.best_trial.number}")

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found by the optimizer
        """
        return self.study.best_params

    def get_best_value(self) -> float:
        """
        Get the best value found by the optimizer
        """
        return self.study.best_value


