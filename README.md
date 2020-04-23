# VasNet

## 2020 Nature Machine Intelligence Paper 
* **"Augmenting Vascular Disease Diagnosis by Vasculature-aware Unsupervised Domain Transfer Learning"**
* Authors: Yong Wang#, Mengqi Ji#, Shengwei Jiang#, Xukang Wang, Jiamin Wu, Feng Duan, Jingtao Fan, Laiqiang Huang, Shaohua Ma*, Lu Fang*, Qionghai Dai*
    -  (#): These authors contributed equally and ranked by coin-tossing.
    -  (*): Corresponding authors.


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
@inproceedings{2020vasnet,
  title={Augmenting Vascular Disease Diagnosis by Vasculature-aware Unsupervised Learning},
  author={Wang, Yong and Ji, Mengqi and Jiang, Shengwei and Wang, Xukang and Wu, Jiamin and Duan, Feng and Fan, Jingtao and Huang, Laiqiang and Ma, Shaohua and Fang, Lu and Dai, Qionghai},
  booktitle={Nature Machine Intelligence},
  year={2020}
}

## Acknowledgments
The reference code includes [CycleGAN](https://github.com/junyanz/CycleGAN), [DRIT](https://github.com/HsinYingLee/DRIT) and [Unsupervised Deblur](https://github.com/ustclby/Unsupervised-Domain-Specific-Deblurring).
