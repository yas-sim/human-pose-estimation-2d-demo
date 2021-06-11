#!/usr/bin/env bash

cp -r ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/demos/human_pose_estimation_3d_demo/python/pose_extractor .
cd pose_extractor
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../..
cp build/pose_extractor/pose_extractor.so .
