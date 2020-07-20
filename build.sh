mkdir -p release
gcc -I /usr/local/cuda/include/ cuda-wrapper-prod-v1.2.c -fPIC -shared -ldl -lcuda -o ./release/libcuda.so
