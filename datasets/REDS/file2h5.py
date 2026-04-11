import os
import cv2
import numpy as np
import h5py
import hdf5plugin


def ensure_dir(s):
    if not os.path.exists(s):
        os.makedirs(s)

def read_events(path):
    f = h5py.File(path, 'r')
    events = f['events'][:]
    f.close()
    return events


def main(paths):
    for path in paths:
        seqs = sorted(os.listdir(os.path.join(path, path+'_orig')))
        for seq in seqs:
            print(path, seq)

            short_dir = os.path.join(path, path+'_short', seq)
            long_dir = os.path.join(path, path+'_long', seq)
            relong_dir = os.path.join(path, path+'_relong', seq)
            true_dir = os.path.join(path, path+'_orig', seq)

            event_path = os.path.join(path, path+'_event', seq, 'events.h5')
            timestamp_path = os.path.join(path, path+'_event', seq, 'timestamps.txt')

            event = read_events(event_path)
            timestamp = np.loadtxt(timestamp_path, dtype=np.uint32)

            for i in range(len(os.listdir(short_dir))-1):
                idx_0 = i*6
                idx_1 = (i+1)*6
                short_0 = cv2.imread(os.path.join(short_dir, str(idx_0+3).zfill(8)+'.png'))
                short_1 = cv2.imread(os.path.join(short_dir, str(idx_1+3).zfill(8)+'.png'))
                long_0 = cv2.imread(os.path.join(long_dir, str(idx_0+3).zfill(8)+'.png'))
                long_1 = cv2.imread(os.path.join(long_dir, str(idx_1+3).zfill(8)+'.png'))
                relong_0 = cv2.imread(os.path.join(relong_dir, str(idx_0+3).zfill(8)+'.png'))
                relong_1 = cv2.imread(os.path.join(relong_dir, str(idx_1+3).zfill(8)+'.png'))
                true_0 = cv2.imread(os.path.join(true_dir, str(idx_0+3).zfill(8)+'.png'))
                true_1 = cv2.imread(os.path.join(true_dir, str(idx_1+3).zfill(8)+'.png'))

                timestamp_split = timestamp[idx_0:idx_1+6+1]
                eidx = np.logical_and(event[:,0]>=timestamp[idx_0], event[:,0]<=timestamp[idx_1+6])
                event_split = event[eidx,:]

                ensure_dir(os.path.join(path, path+'_h5', seq))
                h5_file = os.path.join(path, path+'_h5', seq, str(idx_0+3).zfill(8)+".h5")
                h5 = h5py.File(h5_file, "w")

                h5.create_dataset('timestamps', data=timestamp_split)
                
                h5.create_dataset('short_0', data=short_0)
                h5.create_dataset('short_1', data=short_1)
                h5.create_dataset('long_0', data=long_0)
                h5.create_dataset('long_1', data=long_1)
                h5.create_dataset('relong_0', data=relong_0)
                h5.create_dataset('relong_1', data=relong_1)
                h5.create_dataset('true_0', data=true_0)
                h5.create_dataset('true_1', data=true_1)

                h5.create_group('events')
                h5['events'].create_dataset('t', data=event_split[:,0].astype(np.uint32))
                h5['events'].create_dataset('x', data=event_split[:,1].astype(np.uint16))
                h5['events'].create_dataset('y', data=event_split[:,2].astype(np.uint16))
                h5['events'].create_dataset('p', data=event_split[:,3].astype(np.uint8))

                h5.close()
    return 0


def test():
    path = 'val'
    pair = '000'
    idx = 6
    h5_file = os.path.join('./'+path, path+'_h5', pair, str(idx*6+3).zfill(8)+".h5")
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

    cv2.imwrite('./short_0.png', short_0)
    cv2.imwrite('./short_1.png', short_1)
    cv2.imwrite('./long_0.png', long_0)
    cv2.imwrite('./long_1.png', long_1)
    cv2.imwrite('./relong_0.png', relong_0)
    cv2.imwrite('./relong_1.png', relong_1)
    cv2.imwrite('./true_0.png', true_0)
    cv2.imwrite('./true_1.png', true_1)
    print(timestamps, events['t'][:])


if __name__ == '__main__':
    paths = ['./train']
    main(paths)
    # test()