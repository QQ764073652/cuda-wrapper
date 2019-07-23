# lib so
```
open("/cuda-wrapper/release/libcuda2.so.9.2", O_RDONLY|O_CLOEXEC) = 3
open("/usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0", O_RDONLY|O_CLOEXEC) = 4
open("/usr/lib/x86_64-linux-gnu/libcuda.so.1", O_RDONLY|O_CLOEXEC) = 4
open("/usr/lib/x86_64-linux-gnu/libcuda.so", O_RDONLY|O_CLOEXEC) = 3
```

# tf 获取gpu显存方法
GpuDriver::GetDeviceMemoryInfo
tensorflow/stream_executor/cuda/cuda_driver.cc:1495

# 测试
```
只保留cuMemGetInfo_v2 function就可以正常worker
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    before_func();
    CUresult(*fakecuMemGetInfo_v2)(size_t * , size_t * );
    fakecuMemGetInfo_v2 = dlsym(handle, "cuMemGetInfo_v2");
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    CUresult r;
    r = checkCudaErrors((*fakecuMemGetInfo_v2)(free, total));
    if (*free > total_quota) {
        *free = total_quota;
    }
    *total = total_quota;
    printf("cumemgetinfo: free : %zu, total : %zu\n", *free, *total);
    post_func();
    return r;
}
```