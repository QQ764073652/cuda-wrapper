CUDA=9.2.148-1
CUDA_SHORT=9.2
CUDA_APT=9-2
UBUNTU_VERSION=ubuntu1604

mkdir -p release
gcc -I /usr/local/cuda/include/ cuda-wrapper-prod-v1.2.c -fPIC -shared -ldl -lcuda -o ./release/libcuda.so

find release -type f -name *so -exec mv '{}' '{}'.${CUDA_SHORT} \;