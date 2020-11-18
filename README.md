# LiPose
A Time-of-Flight Sensor based DL algorithm for 3D pose estimation

## Installation
Running LiPose code requires latest Anaconda.
* if using Linux Ubuntu 18.04
Run scripts
```shell script
conda create --name lipose python=3.7 --file lipose_env.txt
conda activate lipse
pip install tensorflow
pip install coremltools
```
* if using MacOS (for testing & evaluation only)
```shell script
conda create --name lipose python=3.7
conda activate lipose
conda install opencv
conda install imageio
pip install tensorflow
pip install coremltools
pip install matplotlib
pip install tensorflow_model_optimization
```
