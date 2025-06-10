rm -r ./build
mkdir build && cd build
cmake ../src -DCHATTY_CUDA_ENABLED=ON \
             -DCHATTY_USE_INT8=ON
make -j8
