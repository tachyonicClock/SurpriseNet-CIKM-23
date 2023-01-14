from multiprocessing import Event
import pandas as pd
import os
import re
import typing as t
from tqdm import tqdm
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from multiprocessing import Pool
from itertools import repeat


VALUE_TAGS = {
    "loss": "Loss_MB/train_phase/train_stream/Task000",
    "final_accuracy": "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000",
}

SERIES_TAGS = {
    "accuracy": "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000",
    "Task Identification Accuracy": "Conditional/P(correct_task_id)",
    "Accuracy given Bad Task Identification": "Conditional/P(correct|!correct_task_id)",
    "Accuracy given Good Task Identification": "Conditional/P(correct|correct_task_id)",
}

def extract_values(ea: EventAccumulator) -> dict:
    """
    Returns the final metrics from the event file.
    """
    result = {}
    for tag_name, tag in VALUE_TAGS.items():
        if tag in ea.Tags()['scalars']:
            result[tag_name] = ea.Scalars(tag)[-1].value
    return result

def extract_series(ea: EventAccumulator) -> dict:
    result = {}
    for tag_name, tag in SERIES_TAGS.items():
        try:
            series = ea.Scalars(tag)
            series = list(map(lambda x: x.value, series))
            result[tag_name] = series
        except KeyError as e:
            pass
            # print(f"Failed to extract {e}")

    return result


def match_listdir(directory, pattern) -> t.List[str]:
    return list([x for x in os.listdir(os.path.join(directory)) if re.match(pattern, x)])


def process_experiment(args) -> dict:
    experiment, log_dir = args
    experiment_code = experiment.split("_")
    i, host, repo_hash, experiment_category, dataset, arch, strategy = experiment_code

    record = {
        "experiment_code": experiment,
        "dataset": dataset,
        "architecture": arch,
        "strategy": strategy,
        "experiment_category": experiment_category,
        "id": i,
        "repo_hash": repo_hash
    }

    experiment_path = os.path.join(log_dir, experiment)
    ea = EventAccumulator(experiment_path)
    ea.Reload()

    try:
        record.update(extract_values(ea))
        record.update(extract_series(ea))
        record["completed_tasks"] = len(record["accuracy"])
    except KeyError as e:
        print(f"Could not load required metric {e} in '{experiment}'")

    try:
        with open(os.path.join(log_dir, experiment, "config.json"), "r") as f:
            config = json.load(f)
            record.update(config)
    except FileNotFoundError as e:
        print(f"Could not load config file '{experiment}/config.json'")
    
    return record


def load_events_to_df(experiment_logs: str, pattern: str) -> pd.DataFrame:
    records = []

    # Loop over each directory in a directory
    experiments = match_listdir(os.path.join(experiment_logs), pattern)

    with Pool(8) as p:
        for record in tqdm(p.imap_unordered(process_experiment, zip(experiments, repeat(experiment_logs))), total=len(experiments)):
            records.append(record)

    return pd.DataFrame.from_dict(records)


df = load_events_to_df("experiment_logs", ".*")
df.to_csv("results/data/ExperimentLogs.csv")
