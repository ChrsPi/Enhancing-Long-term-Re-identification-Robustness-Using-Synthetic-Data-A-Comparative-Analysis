import os
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
import torch
from tqdm import tqdm

DATA_DIR = '../data/boxes-raw/'
ref_img = cv2.imread(DATA_DIR + '00_20240619_r_t.jpg')

class CamAngle(Enum):
    FRONT = 'f'
    LEFT = 'l'
    RIGHT = 'r'


class PalletBlockImage:
    def __init__(self, img_file_path: str):
        self.img_file_path = img_file_path
        filename = os.path.basename(img_file_path)
        split = filename.split('.')[0].split('_')
        self.pf_id = split[0]
        self.date_of_capture = split[1]
        self.angle_of_capture = split[2] 
        
        assert self.angle_of_capture in [CamAngle.FRONT.value, CamAngle.LEFT.value, CamAngle.RIGHT.value], "angle of capture should be one of f, l, r"
        
        self.is_damaged = split[3] == 'd'

    def load_image(self, size=(768, 384)) -> np.ndarray:
        np_img = cv2.imread(self.img_file_path)

        if size is not None:
            np_img = cv2.resize(np_img, size)
        self.image = np_img
        return np_img

    def save_signature(self, signature: np.ndarray):
        self.signature = signature

    def as_dict(self):
        return {"pf_id": self.pf_id, "date_of_capture": self.date_of_capture, "angle_of_capture": self.angle_of_capture, "is_damaged": self.is_damaged, "signature": self.signature}

    def __repr__(self):
        return repr(f"PalletBlock: {self.pf_id=}, {self.date_of_capture=}, {self.angle_of_capture=}, {self.is_damaged=}") 



class PalletBlockDataset:
    def __init__(self, pallet_blocks: list[PalletBlockImage], extractor):
        self.pallet_blocks = pallet_blocks
        if extractor is not None:
            self.extractor = extractor
            self._extract_vectors()


    def _extract_vectors(self):
        print("Extracting feature vectors...")
        for pb in tqdm(self.pallet_blocks):
            
            pb.signature = self.extractor(pb.image)
            if torch.cuda.is_available():
                pb.signature = pb.signature.cpu().detach().numpy()

            pb.signature /= np.linalg.norm(pb.signature)
    
    def get_pbs_by_date(self, date: str, pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.date_of_capture == date]

    def get_pbs_by_angle(self, angle: str, pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.angle_of_capture == angle]

    def get_pbs_by_id(self, pf_id: str, pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.pf_id == pf_id]

    def get_pbs_by_dates(self, dates: list[str], pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.date_of_capture in dates]

    def get_pbs_by_angles(self, angles: list[str], pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.angle_of_capture in angles]

    def get_pbs_by_ids(self, pf_ids: list[str], pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.pf_id in pf_ids]

    def get_train_test_split(self, train_percentage=0.8, pbs=None, permutation=None) -> tuple[list[PalletBlockImage], list[PalletBlockImage]]:
        if pbs is None:
            pbs = self.pallet_blocks
        id_list = list(set([pb.pf_id for pb in pbs]))
        if permutation is None:
            permutation = np.random.permutation(id_list)
        train_ids = permutation[:int(len(id_list) * train_percentage)]
        test_ids = permutation[int(len(id_list) * train_percentage):]
        
        assert len(set(train_ids).intersection(test_ids)) == 0, "train and test ids should not overlap"
        
        train_pbs = [pb for pb in pbs if pb.pf_id in train_ids]
        test_pbs = [pb for pb in pbs if pb.pf_id in test_ids]
        
        return train_pbs, test_pbs

    def get_valid_dates(self) -> list[str]:
        date_strings = [pb.date_of_capture for pb in self.pallet_blocks]
        unique_dates = list(set(date_strings))
        dates = [datetime.strptime(date, "%Y%m%d") for date in unique_dates]
        sorted_dates = sorted(dates)
        sorted_date_strings = [date.strftime("%Y%m%d") for date in sorted_dates]
        return sorted_date_strings

    def get_damaged_pbs(self, pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if pb.is_damaged]

    def get_undamaged_pbs(self, pbs=None) -> list[PalletBlockImage]:
        if pbs is None:
            pbs = self.pallet_blocks
        return [pb for pb in pbs if not pb.is_damaged]



