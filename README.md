# VasNet

 
* **2020 Nature Machine Intelligence**, [Augmenting Vascular Disease Diagnosis by Vasculature-aware Unsupervised Domain Transfer Learning](https://www.nature.com/articles/s42256-020-0188-z)
* Authors: Yong Wang#, Mengqi Ji#, Shengwei Jiang#, Xukang Wang, Jiamin Wu, Feng Duan, Jingtao Fan, Laiqiang Huang, Shaohua Ma*, Lu Fang*, Qionghai Dai*
    -  (#): These authors contributed equally and ranked by coin-tossing.
    -  (*): Corresponding authors.


<p align="center"><img width="500" src="figures/Augment_vascular_disease_diagnosis.jpg"></p>
  
**Fig.1**: **Augment vascular disease diagnosis.** We present the docters with explainable images, including both vascular structures and multi-dimensional features under different modalities, instead of treating deep learning as a "black box" for 0 or 1 diagnosis.

<p align="center"><img width="500" src="figures/VasNet.jpg"></p>
  
**Fig.2**: **The augmentation principle of vascular disease diagnosis.** VasNet learns the image-to-image mapping between two unpaired image domains: raw vascular observations corrupted by scattering, aberrations or non-uniform noises and the segmentation of retinal vascular images. It extracts the vascular topology, colour-codes the blood flow dynamics and unveils the spatiotemporal illumination of regions of interest, examines the pathological features and presents suspicions in contrasting colours, and discovers new diagnostic features and suggests the probability of a disease occurrence.


## Pytorch implementation
* Download the trained model and the demo videos from [figshare](https://figshare.com/articles/VasNet-SI/11986962)  or [Baiduyun](https://pan.baidu.com/s/1JckTg8kLgCgrkJM0_XxtMA) (PIN: bj3y) 

* File structure
    - `./input` includes the training data (trainA & trainB) and the demo data (testB), where the domain B is defined as scattering modality. If you want to test your own image, please notice the image size and put it in the folder `./input/testB`. (Note, the testA is not needed.)
    - `./model` includes the model files for testing
        * Download the trained model 00163.pth file, and place in this subfolder.
    - `./output_test/train` the test/train output will save to this folder 

* Install the environment
    - `pip install -r requirements.txt`  # Python 3
    - Copy the `./model/vgg*` files to the path `~/.torch/models/`
    - My hardware is Titan Xp with CUDA10.1.

* To train: `bash qsub_train.sh`
* To test: `bash qsub_test.sh`

## License
VasNet is released under the MIT License (refer to the LICENSE file for details).

## Citing VasNet
If you find VasNet useful in your research, please consider citing:

    Wang, Y., Ji, M., Jiang, S. et al. Augmenting vascular disease diagnosis by vasculature-aware unsupervised learning. Nat Mach Intell 2, 337–346 (2020). https://doi.org/10.1038/s42256-020-0188-z

    @inproceedings{nmi2020vasnet,
    title={Augmenting Vascular Disease Diagnosis by Vasculature-aware Unsupervised Learning},
    author={Wang, Yong and Ji, Mengqi and Jiang, Shengwei and Wang, Xukang and Wu, Jiamin and Duan, Feng and Fan, Jingtao and Huang, Laiqiang and Ma, Shaohua and Fang, Lu and Dai, Qionghai},
    journal={Nature Machine Intelligence},
    year={2020},
    type={Journal Article}
    }

## Acknowledgments
The reference code includes [CycleGAN](https://github.com/junyanz/CycleGAN), [DRIT](https://github.com/HsinYingLee/DRIT) and [Unsupervised Deblur](https://github.com/ustclby/Unsupervised-Domain-Specific-Deblurring).
