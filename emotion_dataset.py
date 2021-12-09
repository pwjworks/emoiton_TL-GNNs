from torch_geometric.data import Data

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


class EmotionDataset(InMemoryDataset):
    def __init__(self, config, stage, root, sub_idx, pos=None, X=None, Y=None, edge_index=None,
                 transform=None, pre_transform=None):
        self.stage = stage  # Train or test
        #  train setting
        self.subjects = config['subjects']
        self.feature = config["feature"]
        self.dataset = config['dataset']
        self.sub_idx = sub_idx
        # train data
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        self.pos = pos  # position of EEG nodes
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @ property
    def raw_file_names(self):
        return []

    @ property
    def processed_file_names(self):
        return ['./V_{:s}_{:s}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                self.dataset, self.feature, self.stage, self.subjects, self.sub_idx)]

    def download(self):
        pass

    def process(self):
        data_list = []
        # process by samples
        num_samples = np.shape(self.Y)[0]
        for sample_id in tqdm(range(num_samples)):
            x = self.X[sample_id, :, :]
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(self.Y[sample_id, :])
            data = Data(x=x, y=y, pos=self.pos)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
