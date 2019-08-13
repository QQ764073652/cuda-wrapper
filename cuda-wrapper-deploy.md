# base 
```
nvida-driver
nvidia-cuda 90/100
```
# build
```bash
mkdir -p release
gcc -I /usr/local/cuda/include/ tensorflow-cuda-wrapper-prod.c -fPIC -shared -ldl -lcuda -o libcuda-wrapper1.so
gcc -I /usr/local/cuda/include/ pytorch-runtime-cuda-wrapper-prod.c -fPIC -shared -ldl -lcuda -o libcuda-wrapper2.so
```
# prepare
```bash
# cu90
root@VM8035:/cuda-wrapper/test# ls -lrth /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0
lrwxrwxrwx 1 root root 20 Sep  2  2017 /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0 -> libcudart.so.9.0.176
cp /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0.176 /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudatr.so

# cu100 
(base) amax@amax:~$ ls -lrth /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.1
lrwxrwxrwx 1 root root 21 Aug  1 11:48 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.1 -> libcudart.so.10.1.168
cp /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.1.168 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so
```
# run
> WRAPPER_MAX_MEMORY 单位：byte
```bash
export LD_PRELOAD=libcuda-wrapper1.so
export WRAPPER_MAX_MEMORY=42949672960 
python tf-mnist.py

export LD_PRELOAD=libcuda-wrapper2.so
export WRAPPER_MAX_MEMORY=42949672960 
python pytorch-mnist.py
```