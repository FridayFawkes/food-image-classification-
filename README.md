See the paper at: https://arxiv.org/abs/1612.00983

# Setup required for development
In order to use the repo correctly you need to:
1. Create a virtual environment for the ML libraries (tensorflow, keras, etc...)
2. Install tensorflow libraries 
3. Install rar tool
4. Run setup.sh to unzip train and test datasets

## 1.Create Python Virtual environment (for python 3.6):
[Optional]: Update pip3.6
```
sudo python3.6 -m pip3.6 install --upgrade pip
```
Install virtualenv
```
pip3.6 install virtualenv
```
Get the path to the python version:
```
which python3.6
```
Create the new virtual environment in a new folder:
```
virtualenv -p path/to/python3.6 path/to/venv
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
TO DO

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
