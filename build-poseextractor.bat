xcopy /E "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\demos\python_demos\human_pose_estimation_3d_demo\pose_extractor" pose_extractor\
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release ..
msbuild pose_extractor.sln /p:Configuration=Release
cd ..
copy build\pose_extractor\Release\pose_extractor.pyd .
