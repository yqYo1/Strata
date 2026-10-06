#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    fprintf(stderr,"DIAGNOSTIC-START\n"); fflush(stderr);
    puts("READY"); fflush(stdout);
    char *line=NULL; size_t capacity=0;
    while (1) {
        if (!strcmp(argv[1],"blocked")) { pause(); continue; }
        ssize_t n=getline(&line,&capacity,stdin);
        if (n<0) { free(line); return 91; }
        if (!strcmp(line,"QUIT\n")) { free(line); return 0; }
        unsigned long long sum=0;
        for (ssize_t i=0;i<n;++i) sum=(sum*131+(unsigned char)line[i])&0xffffffffULL;
        fprintf(stderr,"DIAGNOSTIC-REQUEST %zd\n",n); fflush(stderr);
        printf("ACK %zd %llu\n",n,sum); fflush(stdout);
    }
}
