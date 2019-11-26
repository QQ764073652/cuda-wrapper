#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdarg.h>
#include <time.h>
#define SIZE 10000

unsigned long long mod = 9973L;

static const char LIB_STRING[] = "libcudart.so";
static const char CONFIG_STRING[] = "WRAPPER_MAX_MEMORY";

int open_flag = 0;
void *handle = NULL;
static size_t total_mem = 0L;
static size_t total_quota = 4217928960L;
static size_t pytorch_offset_size = 500000000L;
static pthread_mutex_t mem_cnt_lock;
char *error;
char timebuf[30];
//nvmlReturn_t nvmlresult;
//nvmlDevice_t nvmldevice;


struct HashArray
{
    unsigned long long key;
    size_t value;
    struct HashArray* next;
}allocsize[10000];

void getCurrentTime(char *buff) {
    struct tm *sTm;
    time_t now = time (0);
    sTm = gmtime (&now);
    strftime (buff, sizeof(buff), "%Y-%m-%d %H:%M:%S", sTm);
}

void write_log (const char *format, ...) {
    va_list arg;
    va_start (arg, format);
    time_t time_log = time(NULL);
    struct tm* tm_log = localtime(&time_log);
    FILE* fpt = fopen("/tmp/cuda-wrapper.logs","a");
    fprintf(fpt, "%04d-%02d-%02d %02d:%02d:%02d ", tm_log->tm_year + 1900, tm_log->tm_mon + 1, tm_log->tm_mday, tm_log->tm_hour, tm_log->tm_min, tm_log->tm_sec);

    vfprintf (fpt, format, arg);
    va_end (arg);
    fflush(fpt);
    fclose(fpt);
}
void addHash(unsigned long long key,size_t value) {
    int temp=key >> 19;
    if(allocsize[temp].key==0) {
        allocsize[temp].key=key;
        allocsize[temp].value=value;
        //printf("allocsize %lld %zu\n", key, value);
    }
    else if(allocsize[temp].key==key) {
        allocsize[temp].value=value;
    }
    else {
        struct HashArray *p=&allocsize[temp];
        while(p->key!=key&&p->next!=NULL) {
            p=p->next;
        }
        if(p->key==key) {
            p->value=value;
        }
        else {
            p->next=(struct HashArray*)malloc(sizeof(struct HashArray));
            p=p->next;
            p->key=key;
            p->value=value;
            p->next=NULL;
        }
    }
    getCurrentTime(timebuf);
    printf("addHash\nTime: %s  addHash: key: %lld value: %zu\n", timebuf, key, value);
}
size_t getHash(unsigned long long key) {
    int temp=key%mod;
    struct HashArray *p=&allocsize[temp];
    if (p == NULL) {
        printf("getHash miss\n");
        getCurrentTime(timebuf);
        printf("Time: %s  key: %lld \n", timebuf, key );
        return 0;
    }
    //printf("pkey: %lld\n", p->key);
    while(p->key!=key&&p->next!=NULL) {
        p=p->next;
    }
    if (p->key == key) {
        printf("getHash hit\n");
        getCurrentTime(timebuf);
        printf("Time: %s  key: %lld value: %zu \n", timebuf, key ,p->value);
        return p->value;
    }
    else {
        printf("hash hit and miss\n");
        getCurrentTime(timebuf);
        printf("Time: %s  key: %lld \n", timebuf, key );
        return 0;
    }
}

void set_quota() {
    char *q = NULL;
    q = getenv(CONFIG_STRING);
    if (q == NULL) {
        printf("set_quota: no env %s found. use default: %zu", CONFIG_STRING, total_quota);
    }
    else {
        total_quota = strtoull(q, NULL, 10);
        printf("set_quota: set total_quota: %zu", total_quota);
    }
}

void init_func() {
    if(open_flag == 0 && handle == NULL) {
        //char *error;
        handle = dlopen (LIB_STRING, RTLD_LAZY);
        if (!handle) {
            fprintf (stderr, "%s\n", dlerror());
            exit(1);
        }
        open_flag = 1;
        dlerror();
        pthread_mutex_init(&mem_cnt_lock, NULL);
        set_quota();
        write_log("Init!\n");
        getCurrentTime(timebuf);
        write_log("Time: %s  total_quota: %zu\n", timebuf, total_quota);

    }
}

int check_alloc_valid(size_t bytesize) {
    write_log("lock mem in check_alloc_valid\n");
    pthread_mutex_lock(&mem_cnt_lock);
    if(total_mem + bytesize + pytorch_offset_size  > total_quota) {
        fprintf (stderr, "alloc %zu failed, total_mem %zu, quota %zu\n", bytesize, total_mem,  total_quota);
        write_log ("alloc %zu failed, total_mem %zu, quota %zu\n", bytesize, total_mem,  total_quota);
        pthread_mutex_unlock(&mem_cnt_lock);
        return 0;
    }
    write_log("unlock mem in check\n");
    pthread_mutex_unlock(&mem_cnt_lock);
    return 1;
}


cudaError_t cudaMalloc( void** devPtr, size_t bytesize) {
    init_func();
    cudaError_t (*fakecudaMalloc)( void** , size_t );
    fakecudaMalloc = dlsym(handle, "cudaMalloc");
    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }

    if(check_alloc_valid(bytesize)) {
        write_log("lock mem in cumemalloc\n");
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        write_log("cudaMalloc:malloc %zu memory\n", bytesize);
        pthread_mutex_unlock(&mem_cnt_lock);
        write_log("unlock mem in cumemalloc\n");

        //nvml_memquery();
        cudaError_t r= (*fakecudaMalloc)( devPtr , bytesize);
        write_log("cudaSuccess:%s , cudaError: %s\n",cudaSuccess,r);
        //nvml_memquery();
        //todo cudaSuccess为null
        if(cudaSuccess != r) {
            write_log("lock if r != CUDA_SUCCESS\n");
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            write_log("ibs: %zu\n", bytesize);
            pthread_mutex_unlock(&mem_cnt_lock);
            write_log("unlock if r != CUDA_SUCCESS\n ");
        } else {
            unsigned long long p = (unsigned long long)(devPtr);
            //printf("cudaMalloc:??????????????????????%lld\n", p);
            addHash(p,bytesize);
            getCurrentTime(timebuf);
            write_log("Time: %s  total_mem: %zu bytesize: %zu total_quota: %zu \n", timebuf, total_mem, bytesize, total_quota);
            //nvml_memquery();
            //printf("%zu\n", getHash(*(unsigned long long*(*devPtr)));
        }

        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
    //dlclose(handle);
}

cudaError_t cudaFree( void* devPtr ) {

    init_func();
    cudaError_t (*fakecudaFree)( void* );
    fakecudaFree = dlsym(handle, "cudaFree");
    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }
    write_log("cudaFree:\n");
    //nvml_memquery();
    cudaError_t r= (*fakecudaFree)( devPtr );
    //nvml_memquery();
    //dlclose(handle);
    write_log("cudaFree: cudaError_t: %s,CUDA_SUCCESS:%s \n",r,cudaSuccess);
    //todo CUDA_SUCCESS为null
    if(r == cudaSuccess) {
//    if(1) {
        //printf("lock mem in cumemfree\n");
        pthread_mutex_lock(&mem_cnt_lock);
        size_t tbytesize = getHash((unsigned long long)(devPtr));
        //printf("cudaFree!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%lld\n", (unsigned long long) (devPtr));

        total_mem -= tbytesize;
        //printf("lock mem in cumemfree\n");
        pthread_mutex_unlock(&mem_cnt_lock);
        getCurrentTime(timebuf);
        write_log("Time: %s  total_mem: %zu bytesize: %zu total_quota: %zu \n", timebuf, total_mem, tbytesize, total_quota);
    }// else {
    //	printf("%d\n", r);
    //}
    return r;
}

cudaError_t cudaMemGetInfo( size_t* free , size_t* total) {
    init_func();
    cudaError_t (*fakecudaMemGetInfo)( size_t* , size_t* );
    fakecudaMemGetInfo = dlsym(handle, "cudaMemGetInfo");
    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }
    cudaError_t r= (*fakecudaMemGetInfo)( free, total );
    if (*free > total_quota) {
        *free = total_quota;
    }
    *total = total_quota;
    write_log("getInfofree : %zu, total : %zu\n", *free, *total);
    return r;
}


cudaError_t cudaMallocPitch( void** devPtr, size_t* pitch, size_t width, size_t height) {

    init_func();
    cudaError_t (*fakecudaMallocPitch)( void** , size_t* , size_t , size_t);
    fakecudaMallocPitch = dlsym(handle, "cudaMallocPitch");

    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }

    //dlclose(handle);

    size_t bytesize = width * height;
    if(check_alloc_valid(bytesize)) {
        //printf("lock mem in cumemallocPitch\n");
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        //printf("%d\n", dptr);
        //printf("ibs: %zu\n", bytesize);
        pthread_mutex_unlock(&mem_cnt_lock);
        //printf("unlock mem in cumemallocPitch\n");

        cudaError_t r= (*fakecudaMallocPitch)( devPtr, pitch, width, height );
        if(cudaSuccess != r) {
            //printf("Pitch: lock if r != CUDA_SUCCESS\n");
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            //printf("%d\n", dptr);
            //printf("ibs: %zu\n", bytesize);
            pthread_mutex_unlock(&mem_cnt_lock);
            //printf("Pitch: unlock if r != CUDA_SUCCESS\n ");
        } else {
            //printf("Pitch success\n");
            unsigned long long p = (unsigned long long)(devPtr);
            //printf("Pitch devPtr:%lld\n", p);
            addHash(p,bytesize);
            //printf(" %zu\n", bytesize);
            getCurrentTime(timebuf);
            write_log("cudaMallocPitch\nTime: %s  total_mem: %zu bytesize: %zu total_quota: %zu \n", timebuf, total_mem, bytesize, total_quota);
            //printf("%zu\n", getHash(*(unsigned long long*(*devPtr)));
        }
        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
}


cudaError_t cudaMallocManaged( void** devPtr, size_t bytesize, unsigned int flags) {
    init_func();
    cudaError_t (*fakecudaMallocManaged)( void** , size_t , unsigned int);
    fakecudaMallocManaged = dlsym(handle, "cudaMallocManaged");

    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }

    //dlclose(handle);

    if(check_alloc_valid(bytesize)) {
        //printf("lock mem in cumemallocManaged\n");
        pthread_mutex_lock(&mem_cnt_lock);
        total_mem += bytesize;
        //printf("%d\n", dptr);
        //printf("ibs: %zu\n", bytesize);
        pthread_mutex_unlock(&mem_cnt_lock);
        //printf("unlock mem in cumemallocManaged\n");

        cudaError_t r= (*fakecudaMallocManaged)( devPtr, bytesize, flags);
        if(cudaSuccess != r) {
            //printf("Managed: lock if r != CUDA_SUCCESS\n");
            pthread_mutex_lock(&mem_cnt_lock);
            total_mem -= bytesize;
            //printf("%d\n", dptr);
            //printf("ibs: %zu\n", bytesize);
            pthread_mutex_unlock(&mem_cnt_lock);
            //printf("Managed: unlock if r != CUDA_SUCCESS\n ");
        } else {
            //printf("Managed success\n");
            unsigned long long p = (unsigned long long)(devPtr);
            //printf("Managed devPtr:%lld\n", p);
            addHash(p,bytesize);
            //printf(" %zu\n", bytesize);
            getCurrentTime(timebuf);
            write_log("cudaMallocManaged\nTime: %s  total_mem: %zu bytesize: %zu total_quota: %zu \n", timebuf, total_mem, bytesize, total_quota);
            //printf("%zu\n", getHash(*(unsigned long long*(*devPtr)));
        }
        return r;
    } else {
        return cudaErrorMemoryAllocation;
    }
}


