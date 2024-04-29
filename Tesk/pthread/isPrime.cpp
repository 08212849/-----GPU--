#include<stdlib.h>
#include <pthread.h>
#include <stdio.h>
#define ThreadNum 10
#define VERLEN 1000
pthread_mutex_t mutex;
int primeNum;

bool isprime(int x){
    bool istrue = true;
    for(int i=2;i*i<=x;i++){
        if(x % i == 0){
            istrue = false;
            break;
        }
    }
    return istrue;
}

void* solveprime(void* threadid){
    int offset = (long)threadid;
    int start = offset * VERLEN + 1;
    int end = start + VERLEN;
    
    for(int i = start; i < end;i++){
        if(i > 1){
            if(isprime(i)){
                // printf("%d is prime\n",i);
                pthread_mutex_lock(&mutex);
                primeNum++;
                pthread_mutex_unlock(&mutex);
            }
        }
    }
    pthread_exit(NULL);
}

int main(){
    pthread_t threads[ThreadNum];
    for(int i=0;i<ThreadNum;i++){
        pthread_create(&threads[i],NULL,solveprime,(void*)i);
    }
    for(int i=0;i<ThreadNum;i++){
        pthread_join(threads[i],NULL);
    }
    printf("a[2:10000] primeNum = %d", primeNum);
    
}