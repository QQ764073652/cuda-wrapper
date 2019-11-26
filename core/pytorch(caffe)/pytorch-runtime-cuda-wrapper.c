#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>

#define SIZE 10000

unsigned long long mod = 9973L;

static const char LIB_STRING[] = "libcudart.so";
static const char LIB_STRING_t[] = "/usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudatr.so";
static const char CONFIG_STRING[] = "WRAPPER_MAX_MEMORY";
static const char LOG_FILENAME[] = "/tmp/wrapper-log";

int open_flag = 0;
void *handle = NULL;
static size_t total_mem = 0L;
static size_t total_quota = 421792896L;
static size_t pytorch_offset_size = 500000000L;
static pthread_mutex_t mem_cnt_lock;
char *error;
char timebuf[30];


struct HashArray {
    unsigned long long key;
    size_t value;
    struct HashArray *next;
} allocsize[10000];

void getCurrentTime() {
    struct tm *sTm;
    time_t now = time(0);
    sTm = gmtime(&now);
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", sTm);
}

void addHash(unsigned long long key, size_t value) {
    // todo temp=key%mod
//    int temp = (key >> 51);
    int temp = (key % mod);
    if (allocsize[temp].key == 0) {
        allocsize[temp].key = key;
        allocsize[temp].value = value;
    } else if (allocsize[temp].key == key) {
        allocsize[temp].value = value;
    } else {
        struct HashArray *p = &allocsize[temp];
        while (p->key != key && p->next != NULL) {
            p = p->next;
        }
        if (p->key == key) {
            p->value = value;
        } else {
            p->next = (struct HashArray *) malloc(sizeof(struct HashArray));
            p = p->next;
            p->key = key;
            p->value = value;
            p->next = NULL;
        }
    }
    getCurrentTime();
    printf("AddHash: key: %lld value: %zu size(MB): %zu \n", key, value,
           (value / 1024 / 1024));
}

size_t getHash(unsigned long long key) {
    int temp = key % mod;
    struct HashArray *p = &allocsize[temp];
    if (p == NULL) {
        getCurrentTime();
        printf("GetHash key: %lld get hash miss \n", key);
        return 0;
    }
    while (p->key != key && p->next != NULL) {
        p = p->next;
    }
    if (p->key == key) {
        getCurrentTime();
        printf("GetHash key: %lld value: %zu size(MB): %zu \n", key, p->value, (p->value / 1024 / 1024));
        return p->value;
    } else {
        getCurrentTime();
        printf("GetHash key: %lld hash hit and miss\n", key);
        return 0;
    }
}

void set_quota() {
    char *q = NULL;
    q = getenv(CONFIG_STRING);
    if (q == NULL) {
        printf("set_quota: no env %s found. use default: %zu\n", CONFIG_STRING, total_quota);
    } else {
        total_quota = strtoull(q, NULL, 10);
        printf("SetQuota set total_quota: %zu\n", total_quota);
    }
}

void init_func() {
    if (open_flag == 0 && handle == NULL) {
        int fd;
//        fd = open(LOG_FILENAME, O_WRONLY | O_CREAT, 0644);
//        if (fd == -1) {
//            perror("open log file failed");
//            exit(1);
//        }
//
//        if (dup2(fd, 1) == -1) {
//            perror("dup2 failed");
//            exit(1);
//        }

        //char *error;
        handle = dlopen(LIB_STRING, RTLD_LAZY);
        if (!handle) {
            fprintf(stderr, "%s\n", dlerror());
            exit(1);
        }
        open_flag = 1;
        dlerror();
        pthread_mutex_init(&mem_cnt_lock, NULL);
        set_quota();
        getCurrentTime();
    }
}

int check_alloc_valid(size_t bytesize) {
    pthread_mutex_lock(&mem_cnt_lock);
    printf("CheckAlloc bytesize %zu, allocated_mem %zu, quota %zu, logic_free_mem: %zu ,allocate_mem(Mb) %zu \n",
           bytesize, total_mem, total_quota,
           (total_quota - total_mem - bytesize - pytorch_offset_size), (total_mem / 1024 / 1024));
    if (total_mem + bytesize + pytorch_offset_size > total_quota) {
        fprintf(stderr, "alloc %zu failed, total_mem %zu, quota %zu\n", bytesize, total_mem, total_quota);
        pthread_mutex_unlock(&mem_cnt_lock);
        return 0;
    }
    pthread_mutex_unlock(&mem_cnt_lock);
    return 1;
}


cudaError_t cudaMalloc(void **devPtr, size_t bytesize) {
    init_func();
    cudaError_t(*fakecudaMalloc)(
    void** , size_t );
    fakecudaMalloc = dlsym(handle, "cudaMalloc");
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    if (check_alloc_valid(bytesize)) {
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
        cudaError_t r = (*fakecudaMalloc)(devPtr, bytesize);
        if (cudaSuccess != r) {
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            pthread_mutex_unlock(&mem_cnt_lock);
        } else {
            unsigned long long p = (unsigned long long) (*devPtr);
            printf("CudaMalloc devPtr: %p %p \n", devPtr, *devPtr);
            addHash(p, bytesize);
            getCurrentTime();
            printf("CudaMalloc allocated_mem: %zu allocate_bytesize: %zu total_quota: %zu logic_free_mem: %zu \n",
                   total_mem, bytesize, total_quota, (total_quota - total_mem));
        }
        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
//    dlclose(handle);
}

cudaError_t cudaFree(void *devPtr) {
    init_func();
    void *hand;
    hand = dlopen(LIB_STRING_t, RTLD_LAZY);
    if (!hand) {
        fprintf(stderr, "%s\n", dlerror());
        exit(1);
    }
    dlerror();
    cudaError_t(*fakecudaFree)(
    void* );
    //todo handler,core dump其他框架可能不是最后出现
    fakecudaFree = dlsym(hand, "cudaFree");
//    fakecudaFree = dlsym(handle, "cudaFree");
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    cudaError_t r = (*fakecudaFree)(devPtr);
    if (r == CUDA_SUCCESS) {
        pthread_mutex_lock(&mem_cnt_lock);
        size_t tbytesize = getHash((unsigned long long) (devPtr));
        printf("CudaFree devPtr: %p %p \n", devPtr, &devPtr);
        total_mem -= tbytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
        getCurrentTime();
        printf("CudaFree allocated_mem: %zu reclaim_mem: %zu total_quota: %zu logic_free_mem: %zu\n",
               total_mem, tbytesize, total_quota, (total_quota - total_mem));
    }
    dlclose(hand);
    return r;
}


cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    init_func();
    cudaError_t(*fakecudaMemGetInfo)(size_t * , size_t * );
    fakecudaMemGetInfo = dlsym(handle, "cudaMemGetInfo");

    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    cudaError_t r = (*fakecudaMemGetInfo)(free, total);
    // 这里free值调整为logic free mem，防止过量申请
    *free = total_quota - total_mem - pytorch_offset_size;
    *total = total_quota;
    printf("cudaMemGetInfo free : %zu, total : %zu\n", *free, *total);
    return r;
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {

    init_func();
    cudaError_t(*fakecudaMallocPitch)(
    void** , size_t * , size_t, size_t);
    fakecudaMallocPitch = dlsym(handle, "cudaMallocPitch");

    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    size_t bytesize = width * height;
    if (check_alloc_valid(bytesize)) {
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
        cudaError_t r = (*fakecudaMallocPitch)(devPtr, pitch, width, height);
        if (cudaSuccess != r) {
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            pthread_mutex_unlock(&mem_cnt_lock);
        } else {
            unsigned long long p = (unsigned long long) (devPtr);
            addHash(p, bytesize);
            getCurrentTime();
            printf("cudaMallocPitch\nTime: %s  total_mem: %zu bytesize: %zu total_quota: %zu \n", timebuf, total_mem,
                   bytesize, total_quota);
        }
        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
}


cudaError_t cudaMallocManaged(void **devPtr, size_t bytesize, unsigned int flags) {
    init_func();
    cudaError_t(*fakecudaMallocManaged)(
    void** , size_t,
    unsigned int);
    fakecudaMallocManaged = dlsym(handle, "cudaMallocManaged");

    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    if (check_alloc_valid(bytesize)) {
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
        cudaError_t r = (*fakecudaMallocManaged)(devPtr, bytesize, flags);
        if (cudaSuccess != r) {
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            pthread_mutex_unlock(&mem_cnt_lock);
        } else {
            unsigned long long p = (unsigned long long) (devPtr);
            addHash(p, bytesize);
            getCurrentTime();
            printf("cudaMallocManaged\nTime: %s  total_mem: %zu bytesize: %zu total_quota: %zu \n", timebuf, total_mem,
                   bytesize, total_quota);
        }
        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
}

cudaError_t cudaGetDeviceCount(int *count) {
    init_func();
    cudaError_t(*fakecudaGetDeviceCount)(
    int*);
    fakecudaGetDeviceCount = dlsym(handle, "cudaGetDeviceCount");
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    cudaError_t r = (*fakecudaGetDeviceCount)(count);
    return r;
}
