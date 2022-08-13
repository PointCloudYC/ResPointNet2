#!/bin/bash

# compile custom operators
cd ops/cpp_wrappers
sh compile_wrappers.sh
cd ../pt_custom_ops
python setup.py install --user
cd ../..

# pre-processing all datasets, you can modify it according to your needs.
# python datasets/PSNet5.py
