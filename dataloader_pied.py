import os
import cv2
import numpy as np
import h5py
import hdf5plugin

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utilities.event_process import events_to_voxel_grid, filter_events_spatial, filter_events_temporal

device = torch.device("cpu")

class Dataloader_train(Dataset):
    def __init__(self, args):
        super(Dataloader_train, self).__init__()
        self.args = args
        self.split = 'train_3'
        self.path = os.path.join(self.args.dataset_path, self.args.dataset_name)
        self.file_names = self.readFilePaths(suffix='.h5')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """ -------------------- load all data -------------------- """
        file_name = self.file_names[idx]
        timestamps, short_0, short_1, long_0, long_1, true_0, true_1, events, prefix = load_data_single(self.path, file_name, self.split)

        short_0 = np.transpose(short_0, (2,0,1))  # [h,w,3] -> [3,h,w]
        short_1 = np.transpose(short_1, (2,0,1))
        long_0 = np.transpose(long_0, (2,0,1))
        long_1 = np.transpose(long_1, (2,0,1))
        true_0 = np.transpose(true_0, (2,0,1))
        true_1 = np.transpose(true_1, (2,0,1))
        
        event_t = events['t'][:]
        event_x = events['x'][:]
        event_y = events['y'][:]
        event_p = events['p'][:]

        """ -------------------- get voxel grid -------------------- """
        _,h,w = long_0.shape
        timestamp_target = timestamps[1]
        timestamp_start = timestamps[3]
        timestamp_end = timestamps[5]
        timestamp_target_start = timestamps[1]-1500
        timestamp_target_end = timestamps[1]+1500

        """ -------------------- random crop -------------------- """
        if long_0.shape[1] > self.args.crop_height and long_0.shape[2] > self.args.crop_width:
            y = np.random.randint(low=1, high=(long_0.shape[1] - self.args.crop_height + 1))
            x = np.random.randint(low=1, high=(long_0.shape[2] - self.args.crop_width + 1))

            short_0 = short_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            short_1 = short_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_0 = long_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_1 = long_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
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
        true_0 = torch.from_numpy(true_0).float() / 255.
        true_1 = torch.from_numpy(true_1).float() / 255.

        evg_0 = torch.from_numpy(evg_0).float()
        evg_1 = torch.from_numpy(evg_1).float()
        evg_2 = torch.from_numpy(evg_2).float()

        return short_0, short_1, long_0, long_1, short_0, short_1, true_0, true_1, evg_0, evg_1, evg_2, prefix

    def readFilePaths(self, suffix='.h5'):
        file_names = []
        for seq in sorted(os.listdir(os.path.join(self.path, self.split, 'event_trigger'))):
            for bag in sorted(os.listdir(os.path.join(self.path, self.split, 'event_trigger', seq))):
                if os.path.splitext(bag)[-1] == suffix:
                    file_names.append(os.path.join(seq, os.path.splitext(bag)[0]))

        return file_names


class Dataloader_test(Dataset):
    def __init__(self, args):
        super(Dataloader_test, self).__init__()
        self.args = args
        self.split = 'test_3'
        self.path = os.path.join(self.args.dataset_path, self.args.dataset_name)
        self.file_names = self.readFilePaths(suffix='.h5')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """ -------------------- load all data -------------------- """
        file_name = self.file_names[idx]
        timestamps, short_0, short_1, long_0, long_1, true_0, true_1, events, prefix = load_data_single(self.path, file_name, self.split)

        short_0 = np.transpose(short_0, (2,0,1))  # [h,w,3] -> [3,h,w]
        short_1 = np.transpose(short_1, (2,0,1))
        long_0 = np.transpose(long_0, (2,0,1))
        long_1 = np.transpose(long_1, (2,0,1))
        true_0 = np.transpose(true_0, (2,0,1))
        true_1 = np.transpose(true_1, (2,0,1))
        
        event_t = events['t'][:]
        event_x = events['x'][:]
        event_y = events['y'][:]
        event_p = events['p'][:]

        """ -------------------- get voxel grid -------------------- """
        _,h,w = long_0.shape
        timestamp_target = timestamps[1]
        timestamp_start = timestamps[3]
        timestamp_end = timestamps[5]
        timestamp_target_start = timestamps[1]-1500
        timestamp_target_end = timestamps[1]+1500

        eidx = np.logical_and(event_t>=timestamp_target, event_t<=timestamp_start)
        events_0 = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float64)
        # evg_0 = events_to_polarity_integration(events_0, num_bins=16, width=w, height=h)
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
            true_0 = true_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_1 = true_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            
            events = np.stack((event_t, event_x, event_y, event_p), axis=1).astype(np.float64)
            events = filter_events_spatial(events, y, x, self.args.crop_height, self.args.crop_width)
            events_0 = filter_events_temporal(events, timestamp_target, timestamp_start)
            events_1 = filter_events_temporal(events, timestamp_target, timestamp_end)
            evg_0 = events_to_voxel_grid(events_0, num_bins=6, width=self.args.crop_width, height=self.args.crop_height)
            evg_1 = events_to_voxel_grid(events_1, num_bins=6, width=self.args.crop_width, height=self.args.crop_height) """

        """ -------------------- to tensor -------------------- """
        short_0 = torch.from_numpy(short_0).float() / 255.
        short_1 = torch.from_numpy(short_1).float() / 255.
        long_0 = torch.from_numpy(long_0).float() / 255.
        long_1 = torch.from_numpy(long_1).float() / 255.
        true_0 = torch.from_numpy(true_0).float() / 255.
        true_1 = torch.from_numpy(true_1).float() / 255.

        evg_0 = torch.from_numpy(evg_0).float()
        evg_1 = torch.from_numpy(evg_1).float()

        return short_0, short_1, long_0, long_1, short_0, short_1, true_0, true_1, evg_0, evg_1, evg_2, prefix

    def readFilePaths(self, suffix='.h5'):
        file_names = []
        for seq in sorted(os.listdir(os.path.join(self.path, self.split, 'event_trigger'))):
            for bag in sorted(os.listdir(os.path.join(self.path, self.split, 'event_trigger', seq))):
                if os.path.splitext(bag)[-1] == suffix:
                    file_names.append(os.path.join(seq, os.path.splitext(bag)[0]))

        return file_names


def load_data_single(path, file_name, split):
    # load data
    seq = file_name.split('/')[0]
    idx = int(file_name.split('/')[-1])
    
    short_0 = cv2.imread(os.path.join(path, 'orig', 'image_ll', seq, str(idx).zfill(3)+'.png'))
    short_1 = cv2.imread(os.path.join(path, 'orig', 'image_ll', seq, str(idx+3).zfill(3)+'.png'))
    long_0 = cv2.imread(os.path.join(path, split, 'image_nl_blur', seq, str(idx).zfill(3)+'.png'))
    long_1 = cv2.imread(os.path.join(path, split, 'image_nl_blur', seq, str(idx+3).zfill(3)+'.png'))
    true_0 = cv2.imread(os.path.join(path, 'orig', 'image_nl', seq, str(idx).zfill(3)+'.png'))
    true_1 = cv2.imread(os.path.join(path, 'orig', 'image_nl', seq, str(idx+3).zfill(3)+'.png'))

    event_trigger_file = h5py.File(os.path.join(path, split, 'event_trigger', seq, str(idx).zfill(3)+'.h5'), "r")
    events = event_trigger_file['events']
    triggers = event_trigger_file['triggers']
    trigger_t_all = triggers['t'][:]
    trigger_p_all = triggers['p'][:]
    timestamps = (trigger_t_all[::2]+trigger_t_all[1::2]) / 2

    prefix = file_name
    return timestamps, short_0, short_1, long_0, long_1, true_0, true_1, events, prefix

