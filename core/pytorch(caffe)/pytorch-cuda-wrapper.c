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
#include <stdarg.h>
#include <time.h>

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

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

CUresult __checkCudaErrors(CUresult err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line);
        //exit(-1);
    }
    return err;
}

/**
 * save log to file
 * @param format
 * @param ...
 */
void write_log(const char *format, ...) {
    va_list arg;
    va_start(arg, format);
    time_t time_log = time(NULL);
    struct tm *tm_log = localtime(&time_log);
    FILE *fpt = fopen("/tmp/cuda-wrapper.logs", "a");
    fprintf(fpt, "%04d-%02d-%02d %02d:%02d:%02d ", tm_log->tm_year + 1900, tm_log->tm_mon + 1, tm_log->tm_mday,
            tm_log->tm_hour, tm_log->tm_min, tm_log->tm_sec);

    vfprintf(fpt, format, arg);
    va_end(arg);
    fflush(fpt);
    fclose(fpt);
}

void set_quota() {
    char *q = NULL;
    q = getenv(CONFIG_STRING);
    if (q == NULL) {
        printf("set_quota: no env %s found. use default: %zu", CONFIG_STRING, total_quota);
    } else {
        total_quota = strtoull(q, NULL, 10);
        printf("set_quota: set total_quota: %zu", total_quota);
    }
}

void init_func() {
    if (open_flag == 0 && handle == NULL) {
        //char *error;
        handle = dlopen(LIB_STRING, RTLD_LAZY);
        if (!handle) {
            fprintf(stderr, "%s\n", dlerror());
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
    write_log("init function");
    init_func();
    before_func();
    printf("init!!!\n");
    CUresult(*fakecuInit)(
    unsigned int);
    fakecuInit = dlsym(handle, "cuInit");

    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    post_func();
    CUresult r;
    r = checkCudaErrors((*fakecuInit)(Flags));
    return r;
}

/**
 * 修改total为total_quota
 * 修改free>total_quota时为total_quota
 * @param free
 * @param total
 * @return
 */
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    write_log("cumemgetinfo: free : %zu, total : %zu\n", *free, *total);
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

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    write_log("cumemgetinfo: free : %zu, total : %zu\n", *free, *total);
    init_func();
    cudaError_t(*fakecudaMemGetInfo)(size_t * , size_t * );
    fakecudaMemGetInfo = dlsym(handle, "cudaMemGetInfo");

    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    cudaError_t r = (*fakecudaMemGetInfo)(free, total);
    //dlclose(handle);
    if (*free > total_quota) {
        *free = total_quota;
    }
    *total = total_quota;
    printf("cumemgetinfo: free : %zu, total : %zu\n", *free, *total);
    return r;
}