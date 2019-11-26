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


/**
#编译动态链接库
gcc mystrcmp.c -fPIC -shared -o libmystrcmp.so
#编译主程序
gcc myverifypasswd.c -L. -lmystrcmp -o myverifypasswd
#显示、确认依赖关系
ldd myverifypasswd
#运行主程序myverifypasswd
./myverifypasswd passwd
> Invalid Password!
*/