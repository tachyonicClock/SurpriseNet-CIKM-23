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

ALL_DATASETS = ["S-FMNIST", "S-CIFAR10", "S-CIFAR100",
                "S-CORe50", "SE-CIFAR100", "SE-CORe50"]

DATASET_NAME_MAP = {
    "S-FMNIST": "S-FMNIST",
    "S-CIFAR10": "S-CIFAR10",
    "S-CIFAR100": "S-CIFAR100",
    "S-CORe50": "S-CORe50",
    "SE-CIFAR100": "SE-CIFAR100",
    "SE-CORe50": "SE-CORe50",
}

DATASET_TASKS ={
    "S-FMNIST": 5,
    "S-CIFAR10": 5,
    "S-CIFAR100": 10,
    "S-CORe50": 10,
    "SE-CIFAR100": 10,
    "SE-CORe50": 10,
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

        # Cumulative
        # self.add_rows("(fce838ce|75988428|692a9d04)", "BL", ["AE", "VAE"], ["cumulative"], check_task_count=False)
        # # Task Oracle
        # self.add_rows("(20bcad46|75988428|692a9d04)", "TO", ["AE"], ["taskOracle"], hp_label="$\\lambda$=0.5")
        # # # Replay
        # self.add_rows("(d11b4e3f|692a9d04)", "OS", ["AE"], ["replay"], ("replay_buffer", [100, 1000, 10000]))
        # self._add_csv_row("results/SnB.csv", "S\\&B", "FF", "mem=1000")
        # # # Naive Strategy
        # self.add_rows("(fce838ce|75988428|692a9d04)", "BL", ["AE"], ["finetuning"])

        # self._add_csv_row("results/GR.csv", "GR", "VAE", "")


        # # TODO Add comparable strategies

        self.add_rows("(54dcf601)", "PL", ["AE", "VAE"], ["taskInference"], ("prune_proportion", [0.2, 0.4, 0.5, 0.6, 0.8]))

        # self.add_rows("(fce838ce|75988428|692a9d04)", "EP", ["VAE", "AE"], ["taskInference"], hp_label="EP")


        table = pd.DataFrame(self.rows, columns=self.columns)

        return table

    def _add_csv_row(self, csv: str, strategy, arch, hp):
        df = pd.read_csv(csv)
        row = [strategy, arch, hp]

        for dataset in ALL_DATASETS:
            # get column as list
            if dataset in df.columns:
                result = df[dataset]
                accuracy_mean = result.mean()
                accuracy_std = result.std()
                n = len(result)
                row.append((accuracy_mean, accuracy_std, n))
            else:
                row.append(None)

        self.rows.append(row)

    def _add_row(self, strategy: str, arch: str, hp: str, row_data: t.Union[pd.DataFrame, pd.Series], check_task_count=True):
        if isinstance(row_data, pd.Series):
            row_data = row_data.to_frame().T
        df = row_data
        row = [strategy, arch, hp]

        for dataset in ALL_DATASETS:
            try:
                result = df[df["dataset"] == dataset]
                if check_task_count:
                    result = result[result["completed_tasks"] == DATASET_TASKS[dataset]]
                
                # result = DATASET_TASKS
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
    
    def add_rows(self, pattern: str, experiment_code: str, archs: t.List[str], strategies: t.List[str], hp: t.Tuple[str, t.List[any]] = None, hp_label = "", check_task_count=True):
        df = self.relevant_experiments(pattern, experiment_code)
        

        if hp is None:
            hp_values = [None]
        else:
            hp_name, hp_values = hp

        for hp_value in hp_values:
            for arch in archs:
                for strategy in strategies:
                    row_df = df[(df["architecture"] == arch) & (df["strategy"] == strategy)]



                    if hp is None:
                        self._add_row(strategy, arch, f"{hp_label}", row_df, check_task_count=check_task_count)
                    else:
                        hp_df = row_df[row_df[hp_name] == hp_value]
                        # print(row_df[hp_name] == float(hp_value))

                        self._add_row(strategy, arch, f"{hp_label}{hp_name}={hp_value}", hp_df, check_task_count=True)

def replace_hp(table: pd.DataFrame):
    table = table.copy()
    table["HP"] = table["HP"].str.replace("prune_proportion", "$\\\lambda$")
    table["HP"] = table["HP"].str.replace("replay_buffer", "mem")
    
    return table


def bold_column(ignore_rows: t.List[int]):
    def _get_accuracy(x):
        if x:
            return x[0]
        else:
            return 0.0 
    def _bold_column(col: pd.Series):
        col = col.copy()
        accuracy = col.map(_get_accuracy)
        accuracy[ignore_rows] = 0.0
        return [ "font-weight: bold;" if v == accuracy.max() else "" for i, v in enumerate(accuracy)]
    return _bold_column

def create_styler(df: pd.DataFrame):
    
    def _format(x):
        if x:
            return f"{x[0]*100:.1f}$\\pm${x[1]*100:.0f}\% {{\\tiny ({x[2]})}}"
        else:
            return "N/A"

    styler = df.style
    styler.apply(bold_column(list(range(0, 3))))
    styler: Styler = styler.format({col: _format  for col in DATASET_NAME_MAP.values()})
    styler.caption = "Experimental Results"
    return styler


df = pd.read_csv("results/all_experiments.csv")
table = TableGenerator(df).generate_table()

table = replace_hp(table)
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

def style_console_table(table: pd.DataFrame):
    table = table.copy()

    def _format_cell(x):
        if x:
            return  f"{x[0]*100:.1f}±{x[1]*100:.2f}% ({x[2]})"
        else:
            return "N/A"

    for col in table.columns:
        table[col] = table[col].map(_format_cell)
    return table

print(style_console_table(table))

style = create_styler(table)
print(style.to_latex(
    hrules=True, 
    convert_css=True, 
    position_float="centering",
    multirow_align="t"

))
