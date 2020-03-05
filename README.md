See the paper at: https://arxiv.org/abs/1612.00983

# Setup required for development
In order to use the repo correctly you need to:
1. Create a virtual environment for the ML libraries (tensorflow, keras, etc...)
2. Install tensorflow libraries 
3. Install rar tool
4. Run setup.sh to unzip train and test datasets

## 1.Create Python Virtual environment (for python 3.6):
[Optional]: Update pip3
```
sudo python3 -m pip3 install --upgrade pip
```
Install virtualenv
```
pip3 install virtualenv
```
Get the path to the python version:
```
which python3
```
Create the new virtual environment in a new folder:
```
virtualenv -p path/to/python3 path/to/venv
```
In order to use the virtual environment just do:
```
source path/to/venv/bin/activate
```
To deactivate the virtual environment:
```
deactivate
```
To delete the virtual environment:
```
rm -rf path/to/venv
```


## 2. Install tensorflow libraries
Check for pip upgrades:
```
pip install --upgrade pip
```
Install stable Tensorflow package with GPU support:
```
pip install --upgrade tensorflow
```
Tensorflow requires the NVIDIA software in [link](https://www.tensorflow.org/install/gpu#software_requirements)
Some commands to check software versions:
```
GPU drivers, CUDA and CUPTI:> nvidia-smi
cuDNN SDK:> cat `whereis cuda`/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
Note: CUPTI comes with CUDA
To check everything:
```
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
```
If there are some dynamic libraries that fail to load do:
```
sudo find / -iname "libName" #get the path that looks like /usr/local...
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/that/you/found #you can add this to ~/.bashrc
```


## 3. Install rar tool
Create and go to a folder for the rar tool installation:
```
mkdir ~/rar-tool
cd ~/rar-tool
```
Run the commands to install the rar tool:
```
wget https://www.rarlab.com/rar/rarlinux-x64-5.5.0.tar.gz
tar -zxvf rarlinux-x64-5.5.0.tar.gz
cd rar/
sudo cp -v rar unrar /usr/local/bin
```
## 4.Run setup.sh script
```
sh setup.sh
```


# food-image-classification-
ten-class food images and classification based on cnn in python



images: train 4654 images (128,128,3) and test 1168 images (128,128,3)

apple:1050 

banana: 310

broccoli: 327

burger': 519

egg: 626

frenchfry: 296

hotdog: 639

pizza: 1248

rice: 352

strawberry: 455






CNN model accuracy: over 90%!
