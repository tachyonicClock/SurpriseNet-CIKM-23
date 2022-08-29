import pandas as pd
import os
import re
import typing as t
from tqdm import tqdm
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "loss": "Loss_MB/train_phase/train_stream/Task000",
    "final_accuracy": "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000",
}

def final_metrics(ea: EventAccumulator) -> dict:
    """
    Returns the final metrics from the event file.
    """
    result = {}
    for tag_name, tag in TAGS.items():
        if tag in ea.Tags()['scalars']:
            result[tag_name] = ea.Scalars(tag)[-1].value
    return result


def match_listdir(directory, pattern) -> t.List[str]:
    return list([x for x in os.listdir(os.path.join(directory)) if re.match(pattern, x)])



def load_events_to_df(pattern: str) -> pd.DataFrame:
    experiment_logs = "experiment_logs"

    records = []

    # Loop over each directory in a directory
    for experiment in tqdm(match_listdir(os.path.join(experiment_logs), pattern)):
        experiment_code = experiment.split("_")
        i, host, repo_hash, experiment_category, dataset, arch, strategy = experiment_code

        record = {
            "experiment_code": experiment_code,
            "dataset": dataset,
            "architecture": arch,
            "strategy": strategy,
            "experiment_category": experiment_category,
            "id": i,
            "repo_hash": repo_hash
        }

        try:
            experiment_path = os.path.join(experiment_logs, experiment)
            ea = EventAccumulator(experiment_path)
            ea.Reload()
            record.update(final_metrics(ea))
        except KeyError as e:
            print(f"Could not load required metric {e} in '{experiment}'")

        try:
            with open(os.path.join(experiment_logs, experiment, "config.json"), "r") as f:
                config = json.load(f)
                record.update(config)
        except FileNotFoundError as e:
            print(f"Could not load config file '{experiment}/config.json'")

        records.append(record)

    return pd.DataFrame.from_dict(records)


df = load_events_to_df(".*")
df.to_csv("results/all_experiments.csv", index=False)
