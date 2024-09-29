import os

import pandas as pd

exp_base_path = "../logs/"

overall_df = pd.DataFrame()

# create metric columns
base_columns = ["Heading", "mAP", "Rank 1", "Rank 3", "Rank 5", "Rank 10"]

overall_df["Test"] = None
overall_df["Model"] = None
# create mean, std, min, max columns for each metric
for col in base_columns[1:]:
    overall_df[col + " Mean"] = None
    overall_df[col + " Std"] = None
    overall_df[col + " Min"] = None
    overall_df[col + " Max"] = None

for exp_name in sorted(os.listdir(exp_base_path)):
    exp_path = os.path.join(exp_base_path, exp_name)
    if not os.path.isdir(exp_path):
        continue
    
    for model_name in sorted(os.listdir(exp_path)):
        model_path = os.path.join(exp_path, model_name)
        if not os.path.isdir(model_path):
            continue
        
        metrics_path = os.path.join(model_path, "metrics_aggregated.xlsx")
        if not os.path.exists(metrics_path):
            print(f"File not found at {metrics_path}")
            continue
        
        df = pd.read_excel(metrics_path)
        row = {
            "Heading": f"{exp_name} - {model_name}",
            "Test": f"{exp_name}",
            "Model": f"{model_name}",
            "mAP Mean": df.loc[df["Test"] == "mean", "Mean Average Precision"].values[0],
            "mAP Std": df.loc[df["Test"] == "std", "Mean Average Precision"].values[0],
            "mAP Min": df.loc[df["Test"] == "min", "Mean Average Precision"].values[0],
            "mAP Max": df.loc[df["Test"] == "max", "Mean Average Precision"].values[0],
            "Rank 1 Mean": df.loc[df["Test"] == "mean", "Rank 1 Acc."].values[0],
            "Rank 1 Std": df.loc[df["Test"] == "std", "Rank 1 Acc."].values[0],
            "Rank 1 Min": df.loc[df["Test"] == "min", "Rank 1 Acc."].values[0],
            "Rank 1 Max": df.loc[df["Test"] == "max", "Rank 1 Acc."].values[0],
            "Rank 2 Mean": df.loc[df["Test"] == "mean", "Rank 2 Acc."].values[0],
            "Rank 2 Std": df.loc[df["Test"] == "std", "Rank 2 Acc."].values[0],
            "Rank 2 Min": df.loc[df["Test"] == "min", "Rank 2 Acc."].values[0],
            "Rank 2 Max": df.loc[df["Test"] == "max", "Rank 2 Acc."].values[0],
            "Rank 3 Mean": df.loc[df["Test"] == "mean", "Rank 3 Acc."].values[0],
            "Rank 3 Std": df.loc[df["Test"] == "std", "Rank 3 Acc."].values[0],
            "Rank 3 Min": df.loc[df["Test"] == "min", "Rank 3 Acc."].values[0],
            "Rank 3 Max": df.loc[df["Test"] == "max", "Rank 3 Acc."].values[0],
            "Rank 5 Mean": df.loc[df["Test"] == "mean", "Rank 5 Acc."].values[0],
            "Rank 5 Std": df.loc[df["Test"] == "std", "Rank 5 Acc."].values[0],
            "Rank 5 Min": df.loc[df["Test"] == "min", "Rank 5 Acc."].values[0],
            "Rank 5 Max": df.loc[df["Test"] == "max", "Rank 5 Acc."].values[0],
            "Rank 10 Mean": df.loc[df["Test"] == "mean", "Rank 10 Acc."].values[0],
            "Rank 10 Std": df.loc[df["Test"] == "std", "Rank 10 Acc."].values[0],
            "Rank 10 Min": df.loc[df["Test"] == "min", "Rank 10 Acc."].values[0],
            "Rank 10 Max": df.loc[df["Test"] == "max", "Rank 10 Acc."].values[0],
        }

        overall_df.loc[row["Heading"]] = row

print(overall_df.head())

# save df
overall_df.to_excel("../logs/overall_metrics.xlsx", index=False)







