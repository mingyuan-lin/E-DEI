import os
import numpy as np
import h5py
import hdf5plugin

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utilities.event_process import events_to_voxel_grid, filter_events_spatial, filter_events_temporal


class Dataloader_train(Dataset):
    def __init__(self, args):
        super(Dataloader_train, self).__init__()
        self.args = args
        self.file_names = self.readFilePaths(suffix='.h5')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """ -------------------- load all data -------------------- """
        file_name = self.file_names[idx]
        timestamps, short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, events, prefix = load_data_single(file_name)

        short_0 = np.transpose(short_0, (2,0,1))  # [h,w,3] -> [3,h,w]
        short_1 = np.transpose(short_1, (2,0,1))
        long_0 = np.transpose(long_0, (2,0,1))
        long_1 = np.transpose(long_1, (2,0,1))
        relong_0 = np.transpose(relong_0, (2,0,1))
        relong_1 = np.transpose(relong_1, (2,0,1))
        true_0 = np.transpose(true_0, (2,0,1))
        true_1 = np.transpose(true_1, (2,0,1))
        
        event_t = events['t'][:]
        event_x = events['x'][:]
        event_y = events['y'][:]
        event_p = events['p'][:]

        """ -------------------- get voxel grid -------------------- """
        _,h,w = long_0.shape
        timestamp_target = timestamps[3]
        timestamp_start = timestamps[6]
        timestamp_end = timestamps[12]
        timestamp_target_start = timestamps[3]-5000
        timestamp_target_end = timestamps[3]+5000

        """ -------------------- random crop -------------------- """
        if long_0.shape[1] > self.args.crop_height and long_0.shape[2] > self.args.crop_width:
            y = np.random.randint(low=1, high=(long_0.shape[1] - self.args.crop_height + 1))
            x = np.random.randint(low=1, high=(long_0.shape[2] - self.args.crop_width + 1))

            short_0 = short_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            short_1 = short_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_0 = long_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_1 = long_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            relong_0 = relong_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            relong_1 = relong_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_0 = true_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_1 = true_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            
            events = np.stack((event_t, event_x, event_y, event_p), axis=1).astype(np.float64)
            events = filter_events_spatial(events, y, x, self.args.crop_height, self.args.crop_width)
            events_0 = filter_events_temporal(events, timestamp_target, timestamp_start)
            events_1 = filter_events_temporal(events, timestamp_target, timestamp_end)
            evg_0 = events_to_voxel_grid(events_0, num_bins=6, width=self.args.crop_width, height=self.args.crop_height)
            evg_1 = events_to_voxel_grid(events_1, num_bins=6, width=self.args.crop_width, height=self.args.crop_height)
            
            events_2 = filter_events_temporal(events, timestamp_target_start, timestamp_target_end)
            evg_2 = events_to_voxel_grid(events_2, num_bins=6, width=self.args.crop_width, height=self.args.crop_height)

        """ -------------------- to tensor -------------------- """
        short_0 = torch.from_numpy(short_0).float() / 255.
        short_1 = torch.from_numpy(short_1).float() / 255.
        long_0 = torch.from_numpy(long_0).float() / 255.
        long_1 = torch.from_numpy(long_1).float() / 255.
        relong_0 = torch.from_numpy(relong_0).float() / 255.
        relong_1 = torch.from_numpy(relong_1).float() / 255.
        true_0 = torch.from_numpy(true_0).float() / 255.
        true_1 = torch.from_numpy(true_1).float() / 255.

        evg_0 = torch.from_numpy(evg_0).float()
        evg_1 = torch.from_numpy(evg_1).float()
        evg_2 = torch.from_numpy(evg_2).float()

        return short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, evg_0, evg_1, evg_2, prefix

    def readFilePaths(self, suffix='.h5'):
        file_names = []
        path = os.path.join(self.args.dataset_path, self.args.dataset_name, 'train', 'train_h5')
        for seq in sorted(os.listdir(path)):
            for bag in sorted(os.listdir(os.path.join(path, seq))):
                if os.path.splitext(bag)[-1] == suffix:
                    file_names.append(os.path.join(path, seq, bag))

        return file_names


class Dataloader_test(Dataset):
    def __init__(self, args):
        super(Dataloader_test, self).__init__()
        self.args = args
        self.file_names = self.readFilePaths(suffix='.h5')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """ -------------------- load all data -------------------- """
        file_name = self.file_names[idx]
        timestamps, short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, events, prefix = load_data_single(file_name)

        short_0 = np.transpose(short_0, (2,0,1))  # [h,w,3] -> [3,h,w]
        short_1 = np.transpose(short_1, (2,0,1))
        long_0 = np.transpose(long_0, (2,0,1))
        long_1 = np.transpose(long_1, (2,0,1))
        relong_0 = np.transpose(relong_0, (2,0,1))
        relong_1 = np.transpose(relong_1, (2,0,1))
        true_0 = np.transpose(true_0, (2,0,1))
        true_1 = np.transpose(true_1, (2,0,1))
        
        event_t = events['t'][:]
        event_x = events['x'][:]
        event_y = events['y'][:]
        event_p = events['p'][:]

        """ -------------------- get voxel grid -------------------- """
        _,h,w = long_0.shape
        timestamp_target = timestamps[3]
        timestamp_start = timestamps[6]
        timestamp_end = timestamps[12]
        timestamp_target_start = timestamps[3]-5000
        timestamp_target_end = timestamps[3]+5000
        
        eidx = np.logical_and(event_t>=timestamp_target, event_t<=timestamp_start)
        events_0 = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float64)
        evg_0 = events_to_voxel_grid(events_0, num_bins=6, width=w, height=h)

        eidx = np.logical_and(event_t>=timestamp_target, event_t<=timestamp_end)
        events_1 = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float64)
        evg_1 = events_to_voxel_grid(events_1, num_bins=6, width=w, height=h)

        eidx = np.logical_and(event_t>=timestamp_target_start, event_t<=timestamp_target_end)
        events_2 = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float64)
        evg_2 = events_to_voxel_grid(events_2, num_bins=6, width=w, height=h)

        """ -------------------- random crop --------------------
        if long_0.shape[1] > self.args.crop_height and long_0.shape[2] > self.args.crop_width:
            y = np.random.randint(low=1, high=(long_0.shape[1] - self.args.crop_height + 1))
            x = np.random.randint(low=1, high=(long_0.shape[2] - self.args.crop_width + 1))

            short_0 = short_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            short_1 = short_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_0 = long_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_1 = long_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            relong_0 = relong_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            relong_1 = relong_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_0 = true_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_1 = true_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            evg_0 = evg_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            evg_1 = evg_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width] """

        """ -------------------- to tensor -------------------- """
        short_0 = torch.from_numpy(short_0).float() / 255.
        short_1 = torch.from_numpy(short_1).float() / 255.
        long_0 = torch.from_numpy(long_0).float() / 255.
        long_1 = torch.from_numpy(long_1).float() / 255.
        relong_0 = torch.from_numpy(relong_0).float() / 255.
        relong_1 = torch.from_numpy(relong_1).float() / 255.
        true_0 = torch.from_numpy(true_0).float() / 255.
        true_1 = torch.from_numpy(true_1).float() / 255.

        evg_0 = torch.from_numpy(evg_0).float()
        evg_1 = torch.from_numpy(evg_1).float()
        evg_2 = torch.from_numpy(evg_2).float()

        return short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, evg_0, evg_1, evg_2, prefix

    def readFilePaths(self, suffix='.h5'):
        file_names = []
        path = os.path.join(self.args.dataset_path, self.args.dataset_name, 'val', 'val_h5')
        for seq in sorted(os.listdir(path)):
            for bag in sorted(os.listdir(os.path.join(path, seq))):
                if os.path.splitext(bag)[-1] == suffix:
                    file_names.append(os.path.join(path, seq, bag))

        return file_names


def load_data_single(file_name):
    # load data
    h5_file = os.path.join(file_name)
    h5 = h5py.File(h5_file, "r")

    timestamps = h5['timestamps'][:]
    short_0 = h5['short_0'][:]
    short_1 = h5['short_1'][:]
    long_0 = h5['long_0'][:]
    long_1 = h5['long_1'][:]
    relong_0 = h5['relong_0'][:]
    relong_1 = h5['relong_1'][:]
    true_0 = h5['true_0'][:]
    true_1 = h5['true_1'][:]
    events = h5['events']

    prefix = file_name
    return timestamps, short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, events, prefix

