#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#define SIZE 10000

unsigned long long mod = 9973L;
static const char CONFIG_STRING[] = "WRAPPER_MAX_MEMORY";
static const char LIB_STRING[] = "libcuda.so";

int open_flag = 0;
void *handle = NULL;
static size_t total_mem = 0L;
static size_t total_quota = 4217928960L; //default set to 4GB
static pthread_mutex_t mem_cnt_lock;
char *error;
char timebuf[30];

struct HashArray {
    unsigned long long key;
    size_t value;
    struct HashArray *next;
} allocsize[10000];

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

CUresult __checkCudaErrors(CUresult err, const char *file, const int line) {
    return err;
}
void addHash(unsigned long long key, size_t value) {
    int temp = key >> 51;
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
    if (q != NULL) {
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
    }
    pthread_mutex_init(&mem_cnt_lock, NULL);
    set_quota();
}

void before_func() {

}


void post_func() {

}


CUresult cuInit(unsigned int Flags) {
    init_func();
    before_func();
    CUresult(*fakecuInit)(
    unsigned int);
    fakecuInit = dlsym(handle, "cuInit");
    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    post_func();
    CUresult r;
    r = checkCudaErrors((*fakecuInit)(Flags));
    return r;
}


CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    before_func();
    CUresult(*fakecuMemGetInfo_v2)(size_t * , size_t * );
    fakecuMemGetInfo_v2 = dlsym(handle, "cuMemGetInfo_v2");
    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    CUresult r;
    r = checkCudaErrors((*fakecuMemGetInfo_v2)(free, total));
    // change free and total to proper value
    if (*free > total_quota) {
        *free = total_quota;
    }
    *total = total_quota;
    post_func();
    return r;
}

int check_alloc_valid(size_t bytesize) {
    pthread_mutex_lock(&mem_cnt_lock);
    if (total_mem + bytesize > total_quota) {
        pthread_mutex_unlock(&mem_cnt_lock);
        return 0;
    }
    pthread_mutex_unlock(&mem_cnt_lock);
    return 1;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    before_func();
    CUresult(*fakecuMemAlloc_v2)(CUdeviceptr * , size_t);
    fakecuMemAlloc_v2 = dlsym(handle, "cuMemAlloc_v2");
    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    post_func();
    if (check_alloc_valid(bytesize)) {
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
        CUresult r = checkCudaErrors((*fakecuMemAlloc_v2)(dptr, bytesize));
        if (CUDA_SUCCESS != r) {
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            pthread_mutex_unlock(&mem_cnt_lock);
        } else {
            addHash((unsigned long long) dptr, bytesize);
        }
        return r;
    } else {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    before_func();
    CUresult(*fakecuMemFree_v2)(CUdeviceptr);
    fakecuMemFree_v2 = dlsym(handle, "cuMemFree_v2");
    if ((error = dlerror()) != NULL) {
        exit(1);
    }
    post_func();
    CUresult r = checkCudaErrors((*fakecuMemFree_v2)(dptr));
    if (CUDA_SUCCESS == r) {
        pthread_mutex_lock(&mem_cnt_lock);
        size_t tbytesize = getHash(dptr);
        total_mem -= tbytesize;
        pthread_mutex_unlock(&mem_cnt_lock);
    }
    return r;
}
