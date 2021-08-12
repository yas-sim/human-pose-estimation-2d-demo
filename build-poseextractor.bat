xcopy /E "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\demos\human_pose_estimation_3d_demo\python\pose_extractor" pose_extractor\
if not exist build (
    mkdir build
)
cd build
cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release -A x64 ..
cmake --build . --config release
cd ..
copy build\pose_extractor\Release\pose_extractor.pyd .
