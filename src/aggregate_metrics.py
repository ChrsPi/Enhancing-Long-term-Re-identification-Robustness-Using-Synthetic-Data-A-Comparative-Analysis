import os

import pandas as pd

BASE_EXP_DIR = "../logs/t10/"


for experiment_dir in os.listdir(BASE_EXP_DIR):
    EXP_DIR = BASE_EXP_DIR + experiment_dir + "/"
    rank_columns = [f"Rank {i} Acc." for i in range(1, 11)]

    columns = ["Test", "Mean Average Precision"] + rank_columns


    metrics_pd = pd.DataFrame()
    for col in columns:
        metrics_pd[col] = None

    metrics_pd.set_index("Test", inplace=True)

    for test_dir in sorted(os.listdir(EXP_DIR)):
        if not os.path.isdir(EXP_DIR + test_dir):
            continue
        for filepath in os.listdir(EXP_DIR + test_dir):
            if filepath != "metrics.txt":
                continue
            test_nr = test_dir.split("_")[1]
            if len(test_nr) == 1:
                test_nr = "0" + test_nr
            with open(EXP_DIR + test_dir + "/" + filepath) as f:
                lines = f.readlines()
                assert len(lines) == 2

                # Parse the mean average precision
                mAP = float(lines[1].split(": ")[1])
                metrics_pd.loc[test_nr, "Mean Average Precision"] = mAP

                # Parse the rank accuracies
                rank_accuracies = {}
                for rank_accuracy in lines[0].split("{")[1].split("}")[0].split(", "):
                    rank, accuracy = rank_accuracy.split(": ")
                    rank_accuracies[int(rank)] = float(accuracy)
                    metrics_pd.loc[test_nr, f"Rank {rank} Acc."] = float(accuracy)

    metrics_pd.sort_index(inplace=True)

    metrics_pd.loc["max"] = metrics_pd.max()
    metrics_pd.loc["min"] = metrics_pd.min()
    metrics_pd.loc["mean"] = metrics_pd.mean()
    metrics_pd.loc["std"] = metrics_pd.std()



    print(metrics_pd)


    print("Saving the aggregated metrics to excel: ", EXP_DIR + "metrics_aggregated.xlsx")
    metrics_pd.to_excel(EXP_DIR + "metrics_aggregated.xlsx")
