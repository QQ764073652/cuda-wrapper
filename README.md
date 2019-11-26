# CUDA wrapper

## Goal

Many deep-learning researchers are not aware of how many resources they are consuming very well. But all will go wrong when there's no free GPU memory. It's necessary to limit GPU memory usage per container to share the GPU across the containers.

We study the most popular machine learning and deep-learning framework Tensorflow, and find that it use some CUDA driver API such as `cuMemGetInfo_v2` and `cuMemAlloc_v2` to get the total/free memory on the node and use it. If we modify the return value from these API calls, we can make tensorflow believes that there's not that much memory on the device or there's not that much free memory on the device.

For other kind of usage of GPU memory, we also use a hash table from `CUdeviceptr` to `bytesize` to record every usage of the container by wrapping `cuMemAlloc` function.

## Approach

This work wraps the NVIDIA driver API by `dlsym` and captures every alloc and free usage by record the size. It will reads ENV to set the quota dynamically. When the app hits the quota, we will return a `CUDA_ERROR_OUT_OF_MEMORY` error to the last alloc function. Although each deep-learning framework handles the error in different ways, it works on tensorflow-based deep-learning jobs.

We record every alloc function like `cuMemAlloc` with both their pointer, the `CUdeviceptr` and size. When there's a `cuMemfree`, we use the given pointer to find out how much memory it has alloced. We use `pthread_mutex_lock` to avoid race condition.

## Usage

### Compile

```bash
gcc -I /usr/local/cuda/include/ cuda-wrapper2.c -fPIC -shared -ldl -lcuda -o ./release/libcuda2.so

```

### Use

[permanent link to libcuda2.so](https://github.com/yzs981130/cuda-wrapper/releases/latest/download/libcuda2.so.10.1)


```bash
LD_PRELOAD=/path/to/libcuda2.so python test.py

```


```bash
LD_PRELOAD=/path/to/libcuda2.so WRAPPER_MAX_MEMORY=4217928960 python test.py

LD_PRELOAD=/cuda-wrapper/release/libcuda2.so.9.2 WRAPPER_MAX_MEMORY=4217928960 python mnist.py
LD_PRELOAD=/cuda-wrapper/release/libcuda3.so.9.2 WRAPPER_MAX_MEMORY=4217928960 python cifar10-pytorch.py

```

### deploy
[cuda-wrapper-deploy.md](./cuda-wrapper-deploy.md)

### TODO
```
1、对于多进程的支持
desigin：多进程数据同步，进程锁，通过互斥量或文件锁解决
```
### Related Project
[pod-gpu-metrics-exporter](https://github.com/ruanxingbaozi/pod-gpu-metrics-exporter)
[k8s-device-plugin](https://github.com/ruanxingbaozi/k8s-device-plugin)

### Optional Project
[gpushare-device-plugin](https://github.com/AliyunContainerService/gpushare-device-plugin)
[gpushare-scheduler-extender](https://github.com/AliyunContainerService/gpushare-scheduler-extender)