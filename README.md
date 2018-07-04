#0. Environment setup

    conda create --name gait_27 python=2.7
    source activate gait_27
    pip install opencv-python=3.4.0.12 tqdm ezodf
    conda install matplotlib numpy pandas xlrd
    conda install -c conda-forge pyexcel tifffile
    conda install -c soumith pytorch
    conda install -c conda-forge pyexcel-ods3
    pip install pyexcel-ods

# 1. Install openpose

* I was not able to build openpose with opencv 3.something so far
* this might be a minor issue because in 18.04 ubuntu realease de default opencv-dev in ppa will be 3.something 


    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
    
    cd openpose
    
    sudo apt-get install libopencv-dev #will use opencv2.4
    sudo apt-get install cmake-qt-gui
    sudo ubuntu/install_cmake.sh 
    sudo ubuntu/install_cudnn.sh 
    sudo ubuntu/install_cuda.sh
    
* then do a reboot
* run cmake-gui from openpose root dir, add source and build path
* run configure, run generate

``` 
cd build/
make -j`nproc` #uses maximum number of processes
```

## 1.2 test Openpose

* from openpose root
```
./build/examples/openpose/openpose.bin --image_dir ./examples/media
```
     


# 2. Configure openpose path und preprocessing

* go to `gait_analysis/data_preprocessing/settings.py` and set `openpose_root=<your root dir for openpose>`
* also set which preprocessing pipeline you want to set up


    calculate_flow = False
    calculate_pose = True


# 3. start preprocessing

* start preprocessing by running `gait_analysis/main_preprocessing.py`

```bash
    source activate gait_27    #activate python environment
    # at first, use --example 1 option to just run on the first 2 person cases to check if it is working
    python main_preprocessing --dataset tum --data-root TUMGAIDimage -o TUMGAIDimage_preprocessed --example 1
    
    # this will go through all the person cases in TUMGAIDimage
    python main_preprocessing --dataset tum --data-root TUMGAIDimage -o TUMGAIDimage_preprocessed
```

* this is probably going to take a while ....