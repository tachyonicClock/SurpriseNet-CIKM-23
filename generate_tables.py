from cgitb import text
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

ALL_DATASETS = ["splitFMNIST", "splitCIFAR10", "splitCIFAR100",
                "splitCORe50", "splitEmbeddedCIFAR100", "splitEmbeddedCORe50"]

DATASET_NAME_MAP = {
    "splitFMNIST": "S-FMNIST",
    "splitCIFAR10": "S-CIFAR10",
    "splitCIFAR100": "S-CIFAR100",
    "splitCORe50": "S-CORe50",
    "splitEmbeddedCIFAR100": "SE-CIFAR100",
    "splitEmbeddedCORe50": "SE-CORe50",
}

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


class TableGenerator():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_table(self) -> texttable.Texttable:
        self.tt = texttable.Texttable(max_width=180)
        tt = self.tt

        # Create Table Header
        dataset_headers = map(DATASET_NAME_MAP.get, ALL_DATASETS)
        tt.header(["CL Strategy", "AE/VAE", "", *dataset_headers])

        self._add_baselines()
        self._add_50_prune()
        self._add_prune_levels()
        self._add_equal_prune()
        return tt

    def _add_row(self, strategy: str, arch: str, hp: str, row_data: pd.DataFrame) -> t.List[str]:
        df = row_data
        row = [strategy, arch, hp]
        for dataset in ALL_DATASETS:
            try:
                result = df[df["dataset"] == dataset]
                accuracy = result.sort_values(by='id').iloc[-1]["final_accuracy"]
                row.append(f"{accuracy*100:0.1f}\%")
            except:
                row.append("x")
        self.tt.add_row(row)

    def relevant_experiments(self, repo_hash, experiment_category) -> pd.DataFrame:
        return self.df[
            (self.df["repo_hash"].str.match(repo_hash)) & 
            (self.df["experiment_category"] == experiment_category)]
    

    def _add_baselines(self):

        # Add Cumulative and finetuning baselines
        df = self.relevant_experiments("dd88ddf", "N")
        for strategy in ["cumulative", "taskOracle", "finetuning"]:
            for arch in ["AE", "VAE"]:
                row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]
                self._add_row(strategy, arch, "", row_df)

        # Add other baselines
        df = self.relevant_experiments("6d14d70a", "OS")

        row_data = df[(df["strategy"] == "SI")]
        self._add_row("SI", "AE", f"$\\lambda$={row_data['si_lambda'].values[0]}", row_data)
        row_data = df[(df["strategy"] == "LwF")]
        self._add_row("LwF", "AE", f"$\\alpha$={row_data['lwf_alpha'].values[0]}", row_data)
        row_data = df[(df["strategy"] == "replay")]
        self._add_row("Replay", "AE", f"mem={row_data['replay_buffer'].values[0]}", row_data)
        row_data = df[(df["strategy"] == "genReplay")]
        self._add_row("genReplay", "VAE", f"", row_data)

    def _add_50_prune(self):
        df = self.relevant_experiments("dd88ddf", "N")
        for strategy in ["taskInference"]:
            for arch in ["AE", "VAE"]:
                row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]
                self._add_row(strategy, arch, "$\\lambda$=0.5", row_df)

    def _add_prune_levels(self):
        df = self.relevant_experiments("(7ee898bd|55dff8e1)", "PL")
        for prune_level in ["0.2", "0.4", "0.6", "0.8"]:
            row_df = df[
                (df["strategy"] == "taskInference") & 
                (df["prune_proportion"] == prune_level)]
            self._add_row("taskInference", "AE", f"$\\lambda$={prune_level}", row_df)

    def _add_equal_prune(self):
        df = self.relevant_experiments("(6d14d70a)", "EP")
        for strategy in ["taskInference"]:
            for arch in ["AE", "VAE"]:
                row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]
                self._add_row(strategy, arch, "EP", row_df)
        


df = pd.read_csv("results/all_experiments.csv")
tt = TableGenerator(df).generate_table()
tt.draw()

print(latextable.draw_latex(
        tt,
        caption="Experimental Results",
        label="tab:experiment_results",
        use_booktabs=True,
        caption_above=True
))

# df = load_events_to_df(".*")
# df.to_csv("results/all_experiments.csv", index=False)