
set -e
pushd ./
rm -rf ./build
mkdir build && cd build
cmake ../src -DCHATTY_CUDA_ENABLED=ON \
             -DCHATTY_x86_ENABLED=OFF \
             -DCHATTY_STATIC_LIB=OFF \
             -DCHATTY_USE_INT8=ON
make -j8
popd

# pushd ./
# rm -rf ./build_test
# mkdir build_test && cd build_test
# cmake ../src/test -DTEST_CUDA_ENABLED=ON
# make -j8
# popd
