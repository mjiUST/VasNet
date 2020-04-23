set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install pytorch torchvision cuda90 -c pytorch -y # add cuda90 if CUDA 9
conda install visdom dominate -c conda-forge -y # install visdom and dominate
