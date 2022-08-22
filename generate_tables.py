import itertools
import pandas as pd
import os
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import texttable
import latextable
import json
import re
import typing as t

CITE_KEYS = {
    "si": "Zenke_Poole_Ganguli_2017",
    "lwf": "Li_Hoiem_2017",
    "genReplay": "Shin_Lee_Kim_Kim_2017",
    "replay": "",
}

ALL_DATASETS = ["splitFMNIST", "splitCIFAR10", "splitCIFAR100", "splitCORe50", "splitEmbeddedCIFAR100", "splitEmbeddedCORe50"]


tags = {
    "loss": "Loss_MB/train_phase/train_stream/Task000",
    "final_accuracy": "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000",
}

def final_metrics(ea: EventAccumulator) -> dict:
    """
    Returns the final metrics from the event file.
    """
    result = {}
    for tag_name, tag in tags.items():
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


def table_equal_prune():
    df = load_events_to_df(".*(6d14d70a).*EP.*")
    df.to_csv("results/experiments_equal_prune.csv")
    tbl = texttable.Texttable()
    tbl.header(["", ""] + ALL_DATASETS)

    for strategy in ["taskOracle", "taskInference"]:
        for arch in ["AE", "VAE"]:
            row = [strategy, arch]
            for dataset in ALL_DATASETS:
                try:
                    result = df.loc[
                        (df['dataset'] == dataset) &
                        (df['strategy'] == strategy) &
                        (df['architecture'] == arch)
                    ]
                    accuracy = result.sort_values(by='id').iloc[-1]["final_accuracy"]
                    row.append(f"{accuracy*100:0.1f}\%")
                except:
                    row.append(f"NA")
            tbl.add_row(row)

    print(tbl.draw())
    print(latextable.draw_latex(
        tbl,
        caption="Equal Prune Experiment",
        label="tab:equal_prune_experiment",
        use_booktabs=True
    ))


def table_other_strategies():
    df = load_events_to_df(".*(6d14d70a|05b6f96cD|05b6f96c)_OS.*")
    tbl = texttable.Texttable()
    df.to_csv("results/experiments_other_strategies.csv")
    # df = pd.read_csv("results/experiments_other_strategies.csv")

    tbl = texttable.Texttable()
    tbl.header([""] + ALL_DATASETS)


    for strategy in ["SI", "LwF", "replay", "genReplay"]:
        row = [strategy] 
        for dataset in ALL_DATASETS:
            try:
                df_row = df.loc[
                    (df['dataset'] == dataset) &
                    (df['strategy'] == strategy)
                ]
                accuracy = df_row.sort_values(by='id').iloc[-1]["final_accuracy"]
                row.append(f"{accuracy*100:0.1f}\%")
            except:
                row.append(f"NA")
        tbl.add_row(row)
        
    print(tbl.draw())
    print(latextable.draw_latex(
        tbl,
        caption="Equal Prune Experiment",
        label="tab:equal_prune_experiment",
        use_booktabs=True
    ))


# table_other_strategies()

all = load_events_to_df(".*")
all.to_csv("results/all_experiments.csv")