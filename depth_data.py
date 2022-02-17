from operator import index
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import cv2


class DepthDataset(torch.utils.data.Dataset):

    def __init__(
                self, 
                dataset_location: Path, 
                split='all',
                val_split=0.1, 
                test_split=0.1,
                randomize_train=True,
                res=(112, 112),
                return_values = ['image', 'label']
            ) -> None:
        super().__init__()

        self.split = split
        if res != (112, 112) and res != (640, 480):
            raise NotImplementedError(
                'Resizing images/labels not implemented. Resolution must be (112, 112) or (640, 480)'
            )
        self.res = res
        self.dataset_location = dataset_location
        self.csv_location = self.dataset_location / 'index.csv'
        self.labels_location = self.dataset_location / ('Labels')
        self.images_location = self.dataset_location / ('Images')
        if self.res != (112, 112):
            self.csv_location = self.csv_location.parent / (self.csv_location.name.replace('.', '_fullres.'))
            self.labels_location = self.labels_location.parent / (self.labels_location.name + 'FullRes')
            self.images_location = self.images_location.parent / (self.images_location.name + 'FullRes')
        self.randomize_train = randomize_train
        self.return_values = return_values

        if not self.csv_location.exists():
            print(self.csv_location, 'not found. Dataset being indexed.')
            img_files = {p.name.replace('.png', '') for p in self.images_location.iterdir()}
            label_files = {p.name.replace('.npy', '') for p in self.labels_location.iterdir()}
            cases = sorted(img_files & label_files)
            files, frames = np.array([c.split('_') for c in cases]).T
            files_unique = np.unique(files)
            splits_unique = [['train', 'val', 'test'][i] for i in np.random.choice([0, 1, 2], len(files_unique), 
                       p=[1 - val_split - test_split, val_split, test_split])]
            split_map = {f: s for f, s in zip(files_unique, splits_unique)}
            splits = [split_map[k] for k in files]
            df = pd.DataFrame({
                'filename': files,
                'frame': frames,
                'split': splits
            })
            df.to_csv(self.csv_location, index=False)
            print('New index saved to ', self.csv_location)
        
        self.data_index = pd.read_csv(self.csv_location)

        if self.split != 'all':
            self.data_index = self.data_index[self.data_index['split'] == self.split]
        
        if self.randomize_train and self.split == 'train':
            self.data_index = self.data_index.sample(frac=1)

    def __getitem__(self, idx):
        filename, frame, split = self.data_index.iloc[idx]
        label = np.load(self.labels_location / f'{filename}_{frame}.npy')
        image = cv2.imread(str(self.images_location / f'{filename}_{frame}.png'))
        return_dict = {
            'image': image,
            'label': label,
            'filename': filename,
            'frame': frame,
            'split': split
        }
        return [return_dict[k] for k in self.return_values]

    def __len__(self):
        return len(self.data_index)


class DepthDatasetFullRes(torch.utils.data.Dataset):

    def __init__(
                self,
                split='all',
                dataset_location=Path('/workspace') / 'data' / 'drives' / 'Local_SSD' / 'sdc' / 'Depthmap', 
                randomize_train=True,
                return_values = ['image', 'label']
            ) -> None:
        super().__init__()

        self.split = split
        self.dataset_location = dataset_location
        self.csv_location = self.dataset_location / 'index_fullres.csv'
        self.labels_location = self.dataset_location / 'LabelsFullRes'
        self.images_location = self.dataset_location / 'ImagesFullRes'
        self.randomize_train = randomize_train
        self.return_values = return_values

        if not self.csv_location.exists():
            print(self.csv_location, 'not found. Dataset being indexed.')



    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass



if __name__ == '__main__':
    ds = DepthDataset(res=(480, 480))
    x, y_true = ds[0]
    print(x.shape, y_true.shape)
