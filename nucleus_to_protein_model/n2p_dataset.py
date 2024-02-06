from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class Nuc2ProtDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        trim: Optional[int] = None,
        split_protein: Optional[str] = None,
        split_images: Optional[str] = None, 
        unique_protein=False,
        negative_images=True,
        shuffle=None,
        downsample=None,
        transform: Optional[Sequence] = (
            transforms.RandomApply(
                [
                    lambda x: transforms.functional.rotate(x, 0),
                    lambda x: transforms.functional.rotate(x, 90),
                    lambda x: transforms.functional.rotate(x, 180),
                    lambda x: transforms.functional.rotate(x, 270),
                ]
            ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ),
    ) -> None:
        super(Dataset, self).__init__()

        self.split_protein = split_protein
        self.split_images = split_images
        if self.split_protein is not None and self.split_images is not None:
            self.labels = labels[
                (labels["split_protein"] == self.split_protein)
                & (labels["split_images_fov"] == self.split_images)
            ]
        elif self.split_protein is not None:
            self.labels = labels[(labels["split_protein"] == self.split_protein)]
        elif self.split_images is not None:
            self.labels = labels[(labels["split_images_fov"] == self.split_images)]
        else:
            self.labels = labels

        if unique_protein:
            self.labels = self.labels.drop_duplicates(subset="ensg")

        if downsample is not None:
            self.labels = self.labels[::downsample]

        self.num_label_class = len(self.labels["label"].unique())
        self.negative_images = negative_images

        self.images = images
        self.trim = trim
        self.transform = transforms.Compose(
            [torch.from_numpy, lambda x: torch.permute(x, [2, 0, 1])]
            + ([] if transform is None else list(transform))
        )

        if shuffle is not None:
            self.shuffle = np.random.RandomState(seed=shuffle).permutation(
                len(self.labels)
            )
        else:
            self.shuffle = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int, str]]:
        if self.shuffle is not None:
            idx = self.shuffle[idx]

        row = self.labels.iloc[idx]
        index = row.name
        images = self.images[index, :, :, :3] #[target  protein, nucleus, nuclear distance, nuclear segmentation]
        if self.trim is not None:
            images = images[self.trim : -self.trim, self.trim : -self.trim]
        images = self.transform(images)

        item = dict()
        item["index"] = index
        item["ensg"] = row.ensg
        item["name"] = row["name"]
        item["loc_grade1"] = row.loc_grade1
        item["loc_grade2"] = row.loc_grade2
        item["loc_grade3"] = row.loc_grade3
        item["protein_id"] = row.protein_id
        item["peptide"] = row["Peptide"].replace("*", "")
        item["ensp"] = row["Protein stable ID"]
        item["FOV_id"] = row.FOV_id
        item["seq_embedding_index"] = row["seq_embedding_index"]
        item["truncation"] = row["truncation"]
        item["label"] = row.label
        item["image"] = images[:2].float() # TODO # why :2??
        item["protein"] = images[0].float() # TODO
        item["nucleus"] = images[1].float() # TODO
        item["nuclei_distance"] = images[2].float() # TODO
        item["localization"] = row["localization"]
        item["complex"] = row["complex"]
        item["complex_fig"] = row["complex_fig"]

        if self.negative_images:
            negative_index = -1
            while negative_index == -1:
                negative_idx = np.random.randint(len(self))
                negative_row = self.labels.iloc[negative_idx]
                if negative_row.ensg != row.ensg:
                    negative_index = negative_row.name
            negative_images = self.images[negative_index, :, :, :3]
            if self.trim is not None:
                negative_images = negative_images[self.trim : -self.trim, self.trim : -self.trim]
            negative_images = self.transform(negative_images)
            item["image_negative"] = negative_images[:2].float()
        
        return item