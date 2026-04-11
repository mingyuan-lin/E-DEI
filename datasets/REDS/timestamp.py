import os
import numpy as np

paths = ['./train']
splits = ['train_orig']

frame_rate = 120

for path in paths:
    for split in splits:
        for seq in sorted(os.listdir(os.path.join(path, split))):
            txt_file = os.path.join(path, 'train_event', seq, 'timestamps.txt')

            print(os.path.join(path, split, seq), ' ---> ', txt_file)
            img_num = len(os.listdir(os.path.join(path, split, seq)))
            # timestamps = np.linspace(0, (img_num-1)/frame_rate, img_num, dtype=np.float64)
            timestamps = np.linspace(0, (img_num-1)*1000000/frame_rate, img_num, dtype=np.uint32)
            
            with open(txt_file, "w") as f:
                for idx in range(img_num):
                    line = '%012d\n' % timestamps[idx]
                    f.writelines(line)
            

