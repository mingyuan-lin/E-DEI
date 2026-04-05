# Dual-Exposure Imaging with Events
## [Paper](https://arxiv.org/abs/2309.09513)

By combining complementary benefits of short- and long-exposure images, Dual-Exposure Imaging (DEI) enhances image quality in low-light scenarios. However, existing DEI approaches inevitably suffer from producing artifacts due to spatial displacement from scene motion and image feature discrepancies from different exposure times. To tackle this problem, we propose a novel Event-based DEI (E-DEI) algorithm, which reconstructs high-quality images from dual-exposure image pairs and events, leveraging high temporal resolution of event cameras to provide accurate inter-/intra-frame dynamic information. Specifically, we decompose this complex task into an integration of two sub-tasks, i.e., event-based motion deblurring and low-light image enhancement tasks, which guides us to design E-DEI network as a dual-path parallel feature propagation architecture. We propose a Dual-path Feature Alignment and Fusion (DFAF) module to effectively align and fuse features extracted from dual-exposure images with assistance of events. Furthermore, we build a real-world Dataset containing Paired low-/normal-light Images and Events (PIED). Experiments on multiple datasets show the superiority of our method.

## Environment setup
- Python 3.8.19
- Pytorch 1.9.1
- NVIDIA GPU + CUDA 11.1
- numpy, argparse, tqdm, natsort, opencv, h5py, hdf5plugin

## Download data
In our paper, we build a real-world dataset **PIED** which contains paired normal- and low-light images and real-world events. (Coming soon.)

## Quick start
### Test
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
