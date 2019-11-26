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

void set_quota() {
    char *q = NULL;
    q = getenv(CONFIG_STRING);
    if (q == NULL) {
        write_log("set_quota: no env %s found. use default: %zu", CONFIG_STRING, total_quota);
    }
    else {
        total_quota = strtoull(q, NULL, 10);
        write_log("set_quota: set total_quota: %zu", total_quota);
    }
}


void init_func() {
    write_log("init function\n");
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
/**
 * 获取gpu memory info
 * @param free
 * @param total
 * @return
 */
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

