// myverifypasswd.c
#include <stdio.h>
#include <string.h>
#include "mystrcmp.h"

void main(int argc,char **argv) {
    char passwd[] = "password";
    if (argc < 2) {
        printf("usage: %s <password>\n",argv[0]);
        return;
    }
    if (!mystrcmp(passwd,argv[1])) {
        printf("Correct Password!\n");
        return;
    }
    printf("Invalid Password!\n");
}

// mystrcmp.h
#include <stdio.h>
int mystrcmp(const char *s1, const char *s2);

// mystrcmp.c
#include <stdio.h>
#include <string.h>
#include "mystrcmp.h"

int mystrcmp(const char *s1, const char *s2)
{
    return strcmp(s1,s2);           //正常字串比较
}

// myhack.c
#include <stdio.h>
#include <string.h>

int mystrcmp(const char *s1, const char *s2)
{
    // always return 0, which means s1 equals to s2
    return 0;
}

/**
#编译动态链接库
gcc mystrcmp.c -fPIC -shared -o libmystrcmp.so
#编译主程序
gcc myverifypasswd.c -L. -lmystrcmp -o myverifypasswd
#编译替换库
gcc myhack.c -fPIC -shared -o myhack.so
#设置LD_PRELOAD环境变量,库中的同名函数在程序运行时优先调用
export LD_PRELOAD="./myhack.so"
#运行主程序myverifypasswd
./myverifypasswd passwd
> Correct Password!
*/