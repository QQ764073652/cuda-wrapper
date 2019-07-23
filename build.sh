CUDA=9.2.148-1
CUDA_SHORT=9.2
CUDA_APT=9-2
UBUNTU_VERSION=ubuntu1604

mkdir -p release
gcc -I /usr/local/cuda/include/ cuda-wrapper.c -fPIC -shared -ldl -lcuda -o ./release/libcuda.so
gcc -I /usr/local/cuda/include/ cuda-wrapper2.c -fPIC -shared -ldl -lcuda -o ./release/libcuda2.so
gcc -I /usr/local/cuda/include/ cuda-wrapper3.c -fPIC -shared -ldl -lcuda -o ./release/libcuda3.so

find release -type f -name *so -exec mv '{}' '{}'.${CUDA_SHORT} \;