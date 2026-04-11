# Dual-Exposure Imaging with Events

[Mingyuan Lin](https://mingyuan-lin.github.io/)<sup>1</sup>
·
Hongyi Liu<sup>2</sup>
·
[Chu He](https://scholar.google.com/citations?user=mvo0cL8AAAAJ&hl=en&oi=ao)<sup>1</sup>
·
[Wen Yang](http://eis.whu.edu.cn/index/szdwDetail?rsh=00200590)<sup>1</sup>
·
[Gui-Song Xia](http://www.captain-whu.com/zh/person/xiaguisong.html)<sup>2</sup>
·
[Lei Yu](https://dvs-whu.cn/)<sup>2</sup>

<sup>1</sup> School of Electronic Information, Wuhan University &nbsp; <sup>2</sup>School of Artificial Intelligence, Wuhan University


## [Paper]()

**Abstract:** By combining complementary benefits of short- and long-exposure images, Dual-Exposure Imaging (DEI) enhances image quality in low-light scenarios. However, existing DEI approaches inevitably suffer from producing artifacts due to spatial displacement from scene motion and image feature discrepancies from different exposure times. To tackle this problem, we propose a novel Event-based DEI (E-DEI) algorithm, which reconstructs high-quality images from dual-exposure image pairs and events, leveraging high temporal resolution of event cameras to provide accurate inter-/intra-frame dynamic information. Specifically, we decompose this complex task into an integration of two sub-tasks, i.e., event-based motion deblurring and low-light image enhancement tasks, which guides us to design E-DEI network as a dual-path parallel feature propagation architecture. We propose a Dual-path Feature Alignment and Fusion (DFAF) module to effectively align and fuse features extracted from dual-exposure images with assistance of events. Furthermore, we build a real-world Dataset containing Paired low-/normal-light Images and Events (PIED). Experiments on multiple datasets show the superiority of our method.

## Environment setup
- Python 3.8.19
- Pytorch 1.9.1
- NVIDIA GPU + CUDA 11.1
- numpy, argparse, tqdm, natsort, opencv, h5py, hdf5plugin

## Dataset prepare
### REDS
1. Download the original 120fps [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset and put it into the folder  `./datasets/REDS/`.
2. Generate the timestamps for every images, and simulate events via [v2e](https://github.com/SensorsINI/v2e).
3. Simulate short-exposure noisy images via `./datasets/darken.py`, simulate long-exposure blurry image via [RIFE](https://github.com/hzwer/Practical-RIFE).
4. Make h5 files via `./datasets/file2h5.py`

We provide several sample datas in [Google Drive](https://drive.google.com/drive/folders/1la5weO653jkFWeDWWFU132peoEQDDlmr?usp=sharing).

### PIED
In our paper, we build a real-world dataset PIED which contains paired normal- and low-light images and real-world events. (Coming soon...)

## Quick start
### Test
Download the weights file from [Google Drive](https://drive.google.com/drive/folders/1la5weO653jkFWeDWWFU132peoEQDDlmr?usp=sharing) and put it into the folder `./weights`.
```bash
sh test.sh
```

### Train
```bash
sh train.sh
```

## Citation
If you find our work useful in your research, please cite:

```
@article{lin2026dual,
        title={Dual-Exposure Imaging with Events},
        author={Lin, Mingyuan and Liu, Hongyi and He, Chu and Yang, Wen and Xia, Gui-Song and Yu, Lei},
        journal={arXiv},
        year={2026}
        }
```
