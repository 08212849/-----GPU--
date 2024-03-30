#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 2
const long VECLEN = 50000;
long threadsSum;
pthread_mutex_t mutexsum;


void *CalSum(void *threadid)
{
    long offset = (long) threadid;
    long start = VECLEN * offset + 1;
    long end = start + VECLEN;
    long mysum = 0;

    for(long i = start; i <= end; i++){
        mysum += i;
    }

    pthread_mutex_lock(&mutexsum);
    threadsSum += mysum;
    printf("Thread %ld did %ld to %ld: mysum=%ld global sum=%ld\n", offset, start,end, mysum, threadsSum);
    pthread_mutex_unlock(&mutexsum);

    pthread_exit((void *) 0);
}

int main(int argc, char *argv[])
{
    pthread_t threads[NUM_THREADS];
    void *status;
    int ThErr;

    pthread_mutex_init(&mutexsum, NULL);


    for (int i = 0; i < NUM_THREADS; i++)
    {
        // printf("In main: creating thread %ld\n", i);
        ThErr = pthread_create(&threads[i], NULL, CalSum, (void *) i);
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
    
    printf("ThreadsSum =  %ld\n", threadsSum);
    
    pthread_mutex_destroy(&mutexsum);
    pthread_exit(NULL);
}