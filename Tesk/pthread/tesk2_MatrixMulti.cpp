#include <pthread.h>
#include<bits/stdc++.h>
#include <sys/time.h>
using namespace std;
#define NUM_THREADS 2
typedef long long ll;
// 随机生成两个矩阵(500*300,300*400)，矩阵相乘要求完成串行和并行两个功能。


const int A_x = 500, A_y = 300, B_x = 300, B_y = 400;
ll matrixA[A_x*A_y],matrixB[B_x*B_y];
ll matrixAns[A_x*B_y];

void init(){
    for(int i = 0; i < A_x*A_y; i++) {
        matrixA[i] = rand() % 10 + 1;
    }
    for(int i = 0; i < B_x*B_y; i++) {
        matrixB[i] = rand() % 10 + 1;
    }
}

void seriaCal(){
    memset(matrixAns,0,sizeof matrixAns);
    for(int i=0;i<A_x;i++)
        for(int j=0;j<B_y;j++){
            ll tmp = 0;
            for(int k=0;k<A_y;k++)
                tmp += matrixA[i*A_y + k] * matrixB[k*B_y + j];
            matrixAns[i*B_y + j] = tmp;
        }
}

void printMatrix(string s){
    if(s == "ans"){
        for(int i=0;i<A_x;i++){
            for(int j=0;j<B_y;j++)
                printf("%lld ",matrixAns[i*A_y + j]);
            printf("\n");
        }
    }
    else if(s == "B"){
        for(int i=0;i<A_x;i++){
            for(int j=0;j<A_y;j++)
                printf("%lld ",matrixA[i*A_y + j]);
            printf("\n");
        }
    }
    else{
        for(int i=0;i<B_x;i++){
            for(int j=0;j<B_y;j++)
                printf("%lld ",matrixB[i*B_y + j]);
            printf("\n");
        }
    }
}

void *parallelCal(void *threadid)
{
    ll VECLEN = A_x * B_y / NUM_THREADS;
    ll offset = (long) threadid;
    ll start = VECLEN * offset + 1;
    ll end = start + VECLEN;

    for(ll i = start; i <= end; i++){
        ll row = i / A_x, vol = i % A_x;
        ll tmp = 0;
        for(ll k = 0;k < A_y;k++)
            tmp += matrixA[row * A_y + k]*matrixB[k * B_y + vol];
        matrixAns[i] = tmp;

    }

    pthread_exit((void *) 0);
}

int main(int argc, char *argv[])
{
    pthread_t threads[NUM_THREADS];
    void *status;
    int ThErr;

    init();

    timeval starttime,endtime;
    gettimeofday(&starttime,0);

    for (int i = 0; i < NUM_THREADS; i++)
    {
        // printf("In main: creating thread %ld\n", i);
        ThErr = pthread_create(&threads[i], NULL, parallelCal, (void *) i);
        if (ThErr)
        {
            printf("ERROR; return code from pthread_create() is %d\n", ThErr);
            exit(-1);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], &status);
    }
    
    
    
    gettimeofday(&endtime,0);
    double timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
    timeuse /= 1000000;
    // printMatrix("B");
    // printMatrix("A");
    // printMatrix("ans");
    printf("parallelCal : %.10f s\n",timeuse);

    gettimeofday(&starttime,0);
    seriaCal();
    gettimeofday(&endtime,0);
    timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
    timeuse /= 1000000;
    printf("seriaCal : %.10f s\n",timeuse);

    pthread_exit(NULL);
}
