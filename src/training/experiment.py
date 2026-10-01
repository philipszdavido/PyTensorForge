import json
import os
import platform
import sys
import time
import uuid

import numpy as np


def hardware_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "device": "cpu",
    }


class Experiment:

    def __init__(self, directory, config_dict, tokenizer_identity, dataset_info, run_id=None):
        self.directory = directory
        self.path = os.path.join(directory, "experiment.json")

        self.record = {
            "run_id": run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "start_time": time.time(),
            "config": config_dict,
            "tokenizer": tokenizer_identity,
            "dataset": dataset_info,
            "hardware": hardware_info(),
            "seed": config_dict.get("training", {}).get("seed"),
            "resumes": [],
            "final_metrics": None,
        }

    @classmethod
    def load_or_create(cls, directory, config_dict, tokenizer_identity, dataset_info):
        path = os.path.join(directory, "experiment.json")

        exp = cls(directory, config_dict, tokenizer_identity, dataset_info)

        if os.path.isfile(path):
            with open(path) as f:
                exp.record = json.load(f)

            exp.record["resumes"].append({"time": time.time(), "hardware": hardware_info()})

        exp.save()

        return exp

    @property
    def run_id(self):
        return self.record["run_id"]

    def update_metrics(self, metrics):
        self.record["final_metrics"] = metrics
        self.save()

    def save(self):
        os.makedirs(self.directory, exist_ok=True)
        tmp = self.path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(self.record, f, indent=2, default=str)

        os.replace(tmp, self.path)
