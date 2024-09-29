import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from metrics import (
    calculate_distance_matrix,
    calculate_rank_20_accuracies,
    calculate_reid_mean_average_precision,
    calculate_summarized_confusion_matrix,
)
from pallet_block_dataset import PalletBlockImage


class ReidExperiment:

    def __init__(self, exp_name: str, model_name: str, exp_type: str,
                 gal_pbs: list[PalletBlockImage] | None = None,
                 query_pbs: list[PalletBlockImage] | None = None) -> None:
        self.exp_name = exp_name
        self.model_name = model_name
        self.exp_type = exp_type
        if gal_pbs:
            self.gallery_pbs = gal_pbs
        if query_pbs:
            self.query_pbs = query_pbs
    
        self.path = f"../logs/{self.exp_name}/{self.model_name}/{self.exp_type}/"

        if self.gallery_pbs is not None and self.query_pbs is not None:
            self.calculate_metrics()

    def register_gallery(self, pbs: list[PalletBlockImage]) -> None:
        self.gallery_pbs = pbs


    def register_query(self, pbs: list[PalletBlockImage]) -> None:
        self.query_pbs = pbs


    def log_experiment(self) -> None:
        assert self.gallery_pbs, "Gallery pallet blocks not registered"
        assert self.query_pbs, "Query pallet blocks not registered"
        assert self.distance_matrix is not None, "Metrics not calculated"
        print(f"Logging pallet blocks to {self.path}")
        os.makedirs(self.path, exist_ok=True)
        with open(self.path + 'gallery_pbs.txt', 'w') as f:
            for pb in self.gallery_pbs:
                f.write(f"{pb.img_file_path}\n")

        with open(self.path + 'query_pbs.txt', 'w') as f:
            for pb in self.query_pbs:
                f.write(f"{pb.img_file_path}\n")

        dates_gal = list(set([pb.date_of_capture for pb in self.gallery_pbs]))
        dates_query = list(set([pb.date_of_capture for pb in self.query_pbs]))

        with open(self.path + 'dates.txt', 'w') as f:
            f.write("Gallery Dates:\n")
            for date in dates_gal:
                f.write(f"{date}\n")
            f.write("Query Dates:\n")
            for date in dates_query:
                f.write(f"{date}\n")


        np.save(self.path + 'distance_matrix.npy', self.distance_matrix)

        with open(self.path + 'metrics.txt', 'w') as f:
            f.write(f"Rank k Accuracies: {self.rank_k_accuracies}\n")
            f.write(f"Mean Average Precision: {self.map}\n")

        self.save_plots()
        self.save_experiment()

    def save_experiment(self) -> None:
        assert self.gallery_pbs, "Gallery pallet blocks not registered"
        assert self.query_pbs, "Query pallet blocks not registered"
        assert self.distance_matrix is not None, "Metrics not calculated"

        for pb in self.gallery_pbs:
            if hasattr(pb, 'image') and pb.image is not None:
                del pb.image
        for pb in self.query_pbs:
            if hasattr(pb, 'image') and pb.image is not None:
                del pb.image

        print(f"Saving experiment to {self.path}")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + 'experiment.pkl', 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_experiment(cls, path: str) -> "ReidExperiment":
        print(f"Loading experiment from {path}")
        with open(path, 'rb') as f:
            reid_exp = pickle.load(f)
            for pb in reid_exp.gallery_pbs:
                pb.load_image()
            for pb in reid_exp.query_pbs:
                pb.load_image()
            return reid_exp


    def calculate_metrics(self) -> None:
        assert self.gallery_pbs, "Gallery pallet blocks not registered"
        assert self.query_pbs, "Query pallet blocks not registered"

        print("Calculating metrics...")
        gallery_ids = [pb.pf_id for pb in self.gallery_pbs]
        query_ids = [pb.pf_id for pb in self.query_pbs]
        self.distance_matrix = calculate_distance_matrix(self.gallery_pbs, self.query_pbs)
        self.cm, self.u_ids = calculate_summarized_confusion_matrix(self.distance_matrix, gallery_ids, query_ids)
        self.rank_k_accuracies = calculate_rank_20_accuracies(self.distance_matrix, gallery_ids, query_ids)

        self.map = calculate_reid_mean_average_precision(self.distance_matrix, gallery_ids, query_ids)
        print(f"Mean Average Precision: {self.map}")
        print(f"Rank k Accuracies: {self.rank_k_accuracies}")


    def save_plots(self) -> None:
        assert self.distance_matrix is not None, "Metrics not calculated"
        self.plot_confusion_matrix(show=False, save=True)
        self.plot_rank_k_accuracy_curve(show=False, save=True)
        plt.close('all')

    def show_plots(self) -> None:
        assert self.distance_matrix is not None, "Metrics not calculated"

        print(f"Mean Average Precision: {self.map}")
        print(f"Rank k Accuracies: {self.rank_k_accuracies}")
        plt.close('all')
        self.plot_confusion_matrix(show=False)
        self.plot_rank_k_accuracy_curve(show=True)

    def plot_rank_k_accuracy_curve(self, show=True, save=False) -> None:
        k_values = list(self.rank_k_accuracies.keys())
        accuracy_values = list(self.rank_k_accuracies.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracy_values, marker='o')
        
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.title('Rank K Accuracy Curve')
        plt.xticks(ticks=range(min(k_values), max(k_values) + 1))
        plt.grid()

        if save:
            plt.savefig(self.path + 'rank_k_accuracy_curve.png')
        if show:
            plt.show()

    def plot_confusion_matrix(self, show=True, save=False) -> None:
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.u_ids, yticklabels=self.u_ids)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if save:
            plt.savefig(self.path + 'confusion_matrix.png')
        if show:
            plt.show()
