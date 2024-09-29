from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pallet_block_dataset import PalletBlockImage


def calculate_distance_matrix(pbs_gallery: list[PalletBlockImage],
                              pbs_query: list[PalletBlockImage]) -> np.ndarray:
    gallery_signatures = np.array([pb.signature for pb in pbs_gallery])
    query_signatures = np.array([pb.signature for pb in pbs_query])
    gallery_signatures = gallery_signatures.reshape(gallery_signatures.shape[0], -1)
    query_signatures = query_signatures.reshape(query_signatures.shape[0], -1)
    
    return np.linalg.norm(gallery_signatures[:, np.newaxis] - query_signatures, axis=2)



def show_distance_heatmap(distance_matrix: np.ndarray, pbs_gallery: list[PalletBlockImage], 
                 pbs_query: list[PalletBlockImage]) -> None:
    # Extract the IDs
    gallery_ids = [pb.pf_id for pb in pbs_gallery]
    query_ids = [pb.pf_id for pb in pbs_query]


    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, xticklabels=query_ids, yticklabels=gallery_ids, cmap="mako", annot=True, fmt=".2f")
    plt.xlabel('Query Pallet Block Image ID')
    plt.ylabel('Gallery Pallet Block Image ID')
    plt.title('Distance Matrix Heatmap')
    plt.show()


def calculate_nearest_ids(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str]) -> list[str]:
    nearest_ids = []
    for i in range(len(query_ids)):
        min_idx = np.argmin(distance_matrix[:, i])
        nearest_ids.append(gallery_ids[min_idx])
    return nearest_ids

def calculate_accuracy(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str]) -> float:
    nearest_ids = calculate_nearest_ids(distance_matrix, gallery_ids, query_ids)
    correct_matches = 0
    
    for nearest_id, query_id in zip(nearest_ids, query_ids):
        if nearest_id == query_id:
            correct_matches += 1
    
    accuracy = correct_matches / len(query_ids)
    return accuracy

def calculate_accuracy_per_id(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str]) -> dict:
    nearest_ids = calculate_nearest_ids(distance_matrix, gallery_ids, query_ids)
    id_accuracy = defaultdict(list)
    
    for nearest_id, query_id in zip(nearest_ids, query_ids):
        if nearest_id == query_id:
            id_accuracy[query_id].append(1)
        else:
            id_accuracy[query_id].append(0)
    
    accuracy_per_id = {key: np.mean(value) for key, value in id_accuracy.items()}
    return accuracy_per_id

def plot_accuracy_heatmap(accuracy_per_id: dict):
    ids = list(accuracy_per_id.keys())
    accuracies = list(accuracy_per_id.values())
    
    data = np.array(accuracies).reshape(1, -1)
    
    plt.figure(figsize=(10, 1))
    sns.heatmap(data, annot=True, xticklabels=ids, yticklabels=["Accuracy"], cmap="viridis", cbar=False)
    plt.title("Accuracy per ID")
    plt.show()


def plot_confusion_matrix(cm, ids, show=True) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ids, yticklabels=ids)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if show:
        plt.show()

def calculate_summarized_confusion_matrix(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str]):
    nearest_ids = calculate_nearest_ids(distance_matrix, gallery_ids, query_ids)
    
    # Create a dictionary to aggregate the counts
    count_dict = defaultdict(int)
    for true_id, pred_id in zip(query_ids, nearest_ids):
        count_dict[(true_id, pred_id)] += 1
    
    # Create a list of unique IDs
    unique_ids = sorted(set(query_ids + gallery_ids))
    
    # Initialize the summarized confusion matrix
    summarized_cm = np.zeros((len(unique_ids), len(unique_ids)), dtype=int)
    
    # Fill the summarized confusion matrix
    id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
    for (true_id, pred_id), count in count_dict.items():
        summarized_cm[id_to_index[true_id], id_to_index[pred_id]] = count
    
    return summarized_cm, unique_ids

def calculate_nearest_k_ids(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str], k: int) -> list[list[str]]:
    nearest_k_ids = []
    for i in range(len(query_ids)):
        nearest_k_idx = np.argsort(distance_matrix[:, i])[:k]
        nearest_k_ids.append([gallery_ids[idx] for idx in nearest_k_idx])
    return nearest_k_ids

def calculate_rank_k_accuracy(nearest_k_ids: list[list[str]], query_ids: list[str]) -> float:
    correct_matches = 0
    for nearest_ids, query_id in zip(nearest_k_ids, query_ids):
        if query_id in nearest_ids:
            correct_matches += 1
    accuracy = correct_matches / len(query_ids)
    return accuracy

def calculate_rank_20_accuracies(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str]) -> dict[int, float]:
    accuracies = {}
    for k in range(1, 21):
        nearest_k_ids = calculate_nearest_k_ids(distance_matrix, gallery_ids, query_ids, k=k)
        accuracy = calculate_rank_k_accuracy(nearest_k_ids, query_ids)
        accuracies[k] = accuracy
    return accuracies


def calculate_reid_mean_average_precision(distance_matrix: np.ndarray, gallery_ids: list[str], query_ids: list[str]) -> float:
    nearest_ids = calculate_nearest_k_ids(distance_matrix, gallery_ids, query_ids, k=len(gallery_ids))
    average_precisions = []

    for query_id in list(set(query_ids)):
        relevant_indices = [i for i, x in enumerate(query_ids) if x == query_id]
        avg_precisions = []
        # print(f"Relevant Indices: {relevant_indices}")
        for query_idx in relevant_indices:
            current_precisions = []
            retrieved_relevant = 0
            
            # print(f"Query ID: {query_id}, Nearest: {nearest_ids[query_idx]}")
            for rank, nearest_id in enumerate(nearest_ids[query_idx]):
                if nearest_id == query_id:
                    retrieved_relevant += 1
                    current_precisions.append(retrieved_relevant / (rank + 1))
                
            # print(f"Current Precisions: {current_precisions}")
            if current_precisions:
                avg_precisions.append(np.mean(current_precisions))

        if avg_precisions:
            average_precisions.append(np.mean(avg_precisions))

    # print(f"Average Precisions: {average_precisions}")
    return float(np.mean(average_precisions))   


def plot_rank_k_accuracy_curve(rank_k_accuracies: dict[int, float], show=True) -> None:
    k_values = list(rank_k_accuracies.keys())
    accuracy_values = list(rank_k_accuracies.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_values, marker='o')
    
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Rank K Accuracy Curve')
    plt.xticks(ticks=range(min(k_values), max(k_values) + 1))
    plt.grid()
    if show:
        plt.show()


if __name__ == '__main__':
    distance_matrix = np.array([[0.1, 0.3, 0.2],
                                [0.4, 0.1, 0.3],
                                [0.2, 0.4, 0.1],
                                [0.3, 0.2, 0.5]])
    gallery_ids = ['id1', 'id2', 'id3', 'id1']
    query_ids = ['id1', 'id2', 'id3', 'id1']

    map_value = calculate_reid_mean_average_precision(distance_matrix, gallery_ids, query_ids)
    print(f"Mean Average Precision: {map_value:.4f}")
