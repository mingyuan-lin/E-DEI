import os
import cv2
import numpy as np
from natsort import natsorted


def ensure_dir(s):
    if not os.path.exists(s):
        os.makedirs(s)

def darken_random(img, alpha, beta, gamma):
    img = img.astype(np.float32) / 255.
    dark = beta * (alpha * img) ** gamma
    return dark

def lighten_random(dark, alpha, beta, gamma):
    img = ((dark / beta) ** (1 / gamma)) / alpha
    return img

def add_noise(frame):
    sigma_p = np.random.uniform(0.05, 0.1)
    sigma_g = np.random.uniform(0.05, 0.1)
    frame_noisy = np.random.normal(frame, sigma_p*frame+sigma_g**2)
    return frame_noisy

def main():
    split = 'val'
    read_path = './'+split+'/'+split+'_orig'
    short_path = './'+split+'/'+split+'_short'
    relong_path = './'+split+'/'+split+'_relong'
    
    ensure_dir(short_path)
    ensure_dir(relong_path)
    txt = open(os.path.join(short_path, 'darkparams.txt'), 'w')

    seq_list = sorted(os.listdir(read_path))
    for seq in seq_list:
        image_dir = os.path.join(read_path, seq)
        image_path = []
        image_path += [os.path.join(image_dir, npy) for npy in os.listdir(image_dir)]
        image_path = natsorted(image_path)

        path_length = (len(image_path)-1) // 6
        print(image_dir, path_length)

        ensure_dir(os.path.join(short_path, seq))
        ensure_dir(os.path.join(relong_path, seq))

        alpha = np.random.uniform(0.9, 1)
        beta = np.random.uniform(0.5, 1)
        gamma = np.random.uniform(2, 3.5)

        txt.write(split+'_'+seq+'\t'+str(alpha)+'\t'+str(beta)+'\t'+str(gamma)+'\n')

        for i in range(path_length):
            idx = i * 6
            sharp3_path = os.path.join(image_dir, str(idx + 3).rjust(8, '0') + '.png')
            sharp3 = cv2.imread(sharp3_path).astype(np.float32)
            
            image_dark =  darken_random(sharp3, alpha, beta, gamma)
            image_noisy = add_noise(image_dark)
            image_relong = lighten_random(image_noisy, alpha, beta, gamma)

            image_noisy = np.clip(image_noisy*255, 0, 255).astype(np.uint8)
            image_relong = np.clip(image_relong*255, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(short_path, seq, str(idx+3).rjust(8, '0')+'.png'), image_noisy)
            cv2.imwrite(os.path.join(relong_path, seq, str(idx+3).rjust(8, '0')+'.png'), image_relong)
        
    txt.close()


if __name__ == '__main__':
    main()