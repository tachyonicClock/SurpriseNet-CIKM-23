from cgitb import text
import pandas as pd
from pandas.io.formats.style import Styler
import os
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import re
import typing as t
from matplotlib import pyplot as plt
import seaborn as sns


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

class TableGenerator():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_table(self) -> pd.DataFrame:

        # Create Table Header
        dataset_headers = map(DATASET_NAME_MAP.get, ALL_DATASETS)
        self.columns = list(["CL Strategy", "AE/VAE", "HP", *dataset_headers])
        self.rows = []

        self._add_baselines()
        self._add_50_prune()
        self._add_prune_levels()
        self._add_equal_prune()
        table = pd.DataFrame(self.rows, columns=self.columns)

        return table

    def _add_row(self, strategy: str, arch: str, hp: str, row_data: t.Union[pd.DataFrame, pd.Series]):
        if isinstance(row_data, pd.Series):
            row_data = row_data.to_frame().T
        df = row_data
        row = [strategy, arch, hp]
        for dataset in ALL_DATASETS:
            try:
                result = df[df["dataset"] == dataset]
                accuracy_mean = result["final_accuracy"].mean()
                accuracy_std = result["final_accuracy"].std()
                    # accuracy_std = 0.0

                n = len(result)
                row.append((accuracy_mean, accuracy_std, n))
            except Exception as err:
                print(f"Could not add {strategy} {arch} {hp} {dataset}")
                print(f"{type(err)}:{err}")
                exit(1)
                return
        self.rows.append(row)

    def relevant_experiments(self, repo_hash, experiment_category) -> pd.DataFrame:
        return self.df[
            (self.df["repo_hash"].str.match(repo_hash)) & 
            (self.df["experiment_category"] == experiment_category)]
    

    def _add_baselines(self):

        # Add Cumulative and finetuning baselines
        df = self.relevant_experiments("(54f30668|dd88ddf)", "N")
        for strategy in ["cumulative", "taskOracle", "finetuning"]:
            for arch in ["AE", "VAE"]:
                row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]
                hp = "$\lambda=0.5$" if strategy == "taskOracle" else ""

                self._add_row(strategy, arch, hp, row_df)

        # Add other baselines
        df = self.relevant_experiments("(6d14d70a|ad06b17a)", "OS")

        strategy = df[(df["strategy"] == "SI")]
        si_lambdas = strategy["si_lambda"].unique()
        si_lambdas.sort()
        for si_lambda in si_lambdas:
            row = strategy[strategy["si_lambda"] == si_lambda]
            self._add_row("SI", "AE", f"$\\lambda$={si_lambda}", row)

        strategy = df[(df["strategy"] == "LwF")]
        lwf_alphas = strategy["lwf_alpha"].unique()
        lwf_alphas.sort()
        for lwf_alpha in lwf_alphas:
            row = strategy[strategy["lwf_alpha"] == lwf_alpha]
            self._add_row("LwF", "AE", f"$\\alpha$={lwf_alpha}", row)

        row_data = df[(df["strategy"] == "replay")]
        self._add_row("Replay", "AE", f"mem={row_data['replay_buffer'].values[0]}", row_data)
        row_data = df[(df["strategy"] == "genReplay")]
        self._add_row("genReplay", "VAE", f"", row_data)

    def _add_50_prune(self):
        df = self.relevant_experiments("dd88ddf", "N")
        # for strategy in ["taskInference"]:
        #     for arch in ["AE", "VAE"]:
        #         row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]
        #         self._add_row(strategy, arch, "$\\lambda$=0.5", row_df)

    def _add_prune_levels(self):
        df = self.relevant_experiments("(44424907)", "PL")
        for prune_level in ["0.2", "0.4", "0.5", "0.6", "0.8"]:
            for arch in ["AE", "VAE"]:
                row_df = df[
                    (df["architecture"] == arch) &
                    (df["strategy"] == "taskInference") & 
                    (df["prune_proportion"] == prune_level)]
                self._add_row("taskInference", arch, f"$\\lambda$={prune_level}", row_df)

    def _add_equal_prune(self):
        df = self.relevant_experiments("(6d14d70a|e2133f95)", "EP")
        for strategy in ["taskInference", "taskOracle"]:
            for arch in ["AE", "VAE"]:
                row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]
                self._add_row(strategy, arch, "EP", row_df)

def bold_column(ignore_rows: t.List[int]):
    def _bold_column(col: pd.Series):
        col = col.copy()
        accuracy = col.map(lambda x: x[0])
        col.values[ignore_rows] = accuracy.min()
        return [ "font-weight: bold;" if v == accuracy.max() else "" for i, v in enumerate(accuracy)]
    return _bold_column

def create_styler(df: pd.DataFrame):
    styler = df.style
    # styler.apply(bold_column(list(range(0, 4))))
    styler: Styler = styler.format({col: lambda x : f"{x[0]*100:.1f}$\\pm${x[1]*100:.0f}\% {{\\tiny ({x[2]})}}" for col in DATASET_NAME_MAP.values()})
    styler.caption = "Experimental Results"
    return styler


df = pd.read_csv("results/all_experiments.csv")
table = TableGenerator(df).generate_table()

table = table.set_index(["CL Strategy", "AE/VAE", "HP"])

# cols = sns.color_palette("cubehelix", 7)
# plt.figure(figsize=(15, 10))
# figs, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=True, figsize=(15, 10))
# table.plot.bar(y=[0, 1], ax=ax0, color=[cols[5], cols[0]])
# table.plot.bar(y=[2, 4], ax=ax1, color=[cols[4], cols[1]])
# table.plot.bar(y=[3, 5], ax=ax2, color=[cols[3], cols[2]])
# ax0.grid(axis="y")
# ax1.grid(axis="y")
# ax2.grid(axis="y")
# ax0.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.tight_layout(pad=2)
# plt.savefig("tmp.png")
# plt.savefig("experiment_results.pdf")

style = create_styler(table)
print(table)
print(style.to_latex(
    hrules=True, 
    convert_css=True, 
    position_float="centering",
    multirow_align="t"

))
