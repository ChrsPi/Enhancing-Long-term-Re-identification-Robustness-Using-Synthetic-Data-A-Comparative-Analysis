import os
import pickle

import torch
from pallet_block_dataset import PalletBlockDataset, PalletBlockImage
from reid_experiment import ReidExperiment
from torchreid.utils import FeatureExtractor
from tqdm import tqdm

ML_MODEL_NAME = "pcb_p4-model.pth.tar-50"
MODEL_DIR_PATH = "../models/re-id/real80/"


model_path = MODEL_DIR_PATH + ML_MODEL_NAME
extractor = FeatureExtractor(
    model_name='pcb_p4',
    model_path=model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

DATA_DIR = '../data/boxes-raw/'
TEMP_DIR = '../temp/'
RELOAD_IMAGES = False
RELOAD_VECTORS = True

def load_pallet_blocks(reload=False) -> list[PalletBlockImage]:
    if os.path.exists(TEMP_DIR + 'pallet_blocks.pkl') and not reload:
        print("Loading pallet blocks from temp file...")
        with open(TEMP_DIR + 'pallet_blocks.pkl', 'rb') as f:
            pallet_blocks = pickle.load(f)
        return pallet_blocks

    pallet_blocks = []
    sorted_files = sorted(os.listdir(DATA_DIR))

    print("Loading pallet blocks...")
    for filename in tqdm(sorted_files):
        pallet_block = PalletBlockImage(os.path.join(DATA_DIR, filename))
        pallet_block.load_image()
        pallet_blocks.append(pallet_block)

    with open(TEMP_DIR + 'pallet_blocks.pkl', 'wb') as f:
        os.makedirs(TEMP_DIR, exist_ok=True)
        print("Saving pallet blocks to temp file...")
        pickle.dump(pallet_blocks, f)

    return pallet_blocks    

pallet_blocks = load_pallet_blocks(reload=RELOAD_IMAGES)


# load cached dataset
model_name = ML_MODEL_NAME.split('-model')[0]
dataset_path = TEMP_DIR + f'dataset-{len(pallet_blocks)}-{model_name}.pkl'

def load_dataset(dataset_path: str, reload=False) -> PalletBlockDataset:
    if os.path.exists(dataset_path) and not reload:
        print(f"Loading dataset from temp file: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = PalletBlockDataset(pallet_blocks, extractor)
        with open(dataset_path, 'wb') as f:
            os.makedirs(TEMP_DIR, exist_ok=True)
            print(f"Saving dataset to temp file: {dataset_path}")
            pickle.dump(dataset, f)
    return dataset

dataset = load_dataset(dataset_path, reload=RELOAD_VECTORS)
dates = dataset.get_valid_dates()

id_strs = sorted(list(set([pb.pf_id for pb in dataset.pallet_blocks])))



full_model_name = f"{model_name}-real80"

for i in range(2, len(dates) + 1):
    gal_pbs = dataset.get_pbs_by_dates(dates[:i-1])
    query_pbs = dataset.get_pbs_by_date(dates[i-1])
    if i == len(dates):
        query_pbs = dataset.get_undamaged_pbs(query_pbs)
    test = ReidExperiment(exp_name="t01",
                        model_name=full_model_name,
                        exp_type=f"test_{i}",
                        gal_pbs=gal_pbs,
                        query_pbs=query_pbs)

    test.log_experiment()
    del test
    print(f"Test {i} done")

gal_pbs = dataset.get_undamaged_pbs()
damaged_pbs = dataset.get_damaged_pbs()

test_damaged = ReidExperiment(exp_name="t01",
                        model_name=full_model_name,
                        exp_type="test_damaged",
                        gal_pbs=gal_pbs,
                        query_pbs=damaged_pbs)
test_damaged.log_experiment()



