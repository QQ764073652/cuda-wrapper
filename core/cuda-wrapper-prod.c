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
static const char LIB_STRING_STATIC[] = "libcuda.so";
static const char LIB_STRING_t[] = "/usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudatr.so";
static const char CONFIG_STRING[] = "WRAPPER_MAX_MEMORY";

int open_flag = 0;
int static_open_flag = 0;
void *handle = NULL;
void *staticHandle = NULL;
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

void addHash(unsigned long long key, size_t value) {
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
}

size_t getHash(unsigned long long key) {
    int temp = key % mod;
    struct HashArray *p = &allocsize[temp];
    if (p == NULL) {
        return 0;
    }
    while (p->key != key && p->next != NULL) {
        p = p->next;
    }
    if (p->key == key) {

        return p->value;
    } else {

        return 0;
    }
}

void set_quota() {
    char *q = NULL;
    q = getenv(CONFIG_STRING);
    if (q == NULL) {
    } else {
        total_quota = strtoull(q, NULL, 10);
    }
}

void init_func() {
    if (open_flag == 0 && handle == NULL) {
        //char *error;
        handle = dlopen(LIB_STRING, RTLD_LAZY);
        if (!handle) {
            exit(1);
        }
        open_flag = 1;
        dlerror();
        pthread_mutex_init(&mem_cnt_lock, NULL);
        set_quota();
    }
}

int check_alloc_valid(size_t bytesize) {
    pthread_mutex_lock(&mem_cnt_lock);
    if (total_mem + bytesize + pytorch_offset_size > total_quota) {
        pthread_mutex_unlock(&mem_cnt_lock);
        return 0;
    }
    pthread_mutex_unlock(&mem_cnt_lock);
    return 1;
}

// runtime
cudaError_t cudaMalloc(void **devPtr, size_t bytesize) {
    init_func();
    cudaError_t(*fakecudaMalloc)(
    void** , size_t );
    fakecudaMalloc = dlsym(handle, "cudaMalloc");
    if ((error = dlerror()) != NULL) {
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
            addHash(p, bytesize);
        }
        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
}

cudaError_t cudaFree(void *devPtr) {
    init_func();
    void *hand;
    hand = dlopen(LIB_STRING_t, RTLD_LAZY);
    if (!hand) {
        exit(1);
    }
    dlerror();
    cudaError_t(*fakecudaFree)(
    void* );
    fakecudaFree = dlsym(hand, "cudaFree");
    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    cudaError_t r = (*fakecudaFree)(devPtr);
    if (r == CUDA_SUCCESS) {
        pthread_mutex_lock(&mem_cnt_lock);
        size_t tbytesize = getHash((unsigned long long) (devPtr));
        total_mem -= tbytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
    }
    dlclose(hand);
    return r;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    init_func();
    cudaError_t(*fakecudaMemGetInfo)(size_t * , size_t * );
    fakecudaMemGetInfo = dlsym(handle, "cudaMemGetInfo");

    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    cudaError_t r = (*fakecudaMemGetInfo)(free, total);
    *free = total_quota - total_mem - pytorch_offset_size;
    *total = total_quota;
    return r;
}
// static
void before_func(){
    if (static_open_flag == 0 && staticHandle == NULL) {
        //char *error;
        staticHandle = dlopen(LIB_STRING_STATIC, RTLD_LAZY);
        if (!staticHandle) {
            exit(1);
        }
        static_open_flag = 1;
        dlerror();
    }
    pthread_mutex_init(&mem_cnt_lock, NULL);
    set_quota();
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    before_func();
    CUresult(*fakecuMemGetInfo_v2)(size_t * , size_t * );
    fakecuMemGetInfo_v2 = dlsym(staticHandle, "cuMemGetInfo_v2");
    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    CUresult r = (*fakecuMemGetInfo_v2)(free, total);
    // change free and total to proper value
    if (*free > total_quota) {
        *free = total_quota;
    }
    *total = total_quota;
    return r;
}