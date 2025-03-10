from typing import Dict, Any

import yaml
import optuna
class HyperparameterManager:
    """
    Class to manage hyperparameters. Loads YAML specifications of search spaces.
    Maps parameter types to Optuna distributions (e.g., suggest_float)
    """
    def __init__(self, yaml_path: str):
        self.data = self.load_conifg(yaml_path)

    def load_conifg(self, yaml_path: str) -> dict:
        with open(yaml_path, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc
        return data

    def define_search_space(self, trial : optuna.trial ) -> Dict[str, Dict[str, Any]]:
        args = {"model": {}, "policy": {}, "env": {}, "experiment": {}} # These are dictionaries that will be filled with the hyperparameters
        for arg in args.keys():
            if arg not in self.data.keys():
                continue
            if self.data[arg] is None:
                continue
            for key, value in self.data[arg].items():
                # adress fixed values
                if not isinstance(value, dict) or "type" not in value.keys():
                    args[arg][key] = value
                    continue
                step = value["step"] if "step" in value.keys() else None
                if value['type'] == 'float':
                    args[arg][key] = trial.suggest_float(key, float(value['low']), float(value['high']), step=step)
                elif value['type'] == 'int':
                    args[arg][key] = trial.suggest_int(key, float(value['low']), float(value['high']), step=step)
                elif value['type'] == 'categorical':
                    args[arg][key] = trial.suggest_categorical(key, value['choices'])
                else:
                    raise ValueError(f"Unknown hp type {value['type']}")
        return args


