# paper_code
# Installation Instructions
This code was developed under Linux (Debian wheezy, 64 bit) and was tested only in this environment.

Build Caffe and Python bindings as described in the official documentation. 
You will have to disable CuDNN support and enable C++ 11.

$ make all pycaffe

Install Python Click package (required for demo only)

$ pip install click

# Set PYTHONPATH variable

$ export PYTHONPATH=`pwd`/python

# Download Caffe Models

$ cd models/deepercut

$ ./download_models.sh

# Run Demo

$ cd python/pose

$ python ./pose_demo.py image.png --out_name=prediction