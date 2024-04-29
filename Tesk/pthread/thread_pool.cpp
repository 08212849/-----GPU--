#include<stdio.h>
#include<pthread.h>
#include <stdlib.h>

#define MAX_QUEUE 20
#define MAX_THEADS 10

typedef struct{
    void*(*func)(void*);
    void* args;
}task_t;

typedef struct{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int num_threads;
    int size,front,rear;
    pthread_t threads[MAX_THEADS];
    task_t queue[MAX_QUEUE];
    bool shutdown;

}thread_pool_t;

void* work_routinue(void* args){
    thread_pool_t *pool = (thread_pool_t*)args;
    while(true){
        pthread_mutex_lock(&pool->mutex);
        while(pool->size == 0 && !pool->shutdown)
            pthread_cond_wait(&pool->cond, &pool->mutex);
        if(pool->size == 0 && pool->shutdown){
            pthread_mutex_unlock(&pool->mutex);
            pthread_exit(NULL);
        }
        task_t task = pool->queue[pool->front];
        pool->front = (pool->front + 1) % MAX_QUEUE;
        pool->size--;
        pthread_mutex_unlock(&pool->mutex);
        task.func(task.args);
    }
}

void pool_init(thread_pool_t* pool,int num_threads){
    pool->num_threads = num_threads;
    pool->shutdown = false;
    pthread_cond_init(&pool->cond,NULL);
    pthread_mutex_init(&pool->mutex,NULL);
    pool->front = pool->rear = pool->size = 0;
    for(int i=0; i < num_threads;i++){
        pthread_create(&pool->threads[i],NULL,work_routinue,pool);
        // printf("pthread %d is created.\n",i);
    }
}

void add_task(thread_pool_t* pool, void*(*func)(void*),void*args){
    pthread_mutex_lock(&pool->mutex);
    if(pool->size == MAX_QUEUE){
        puts("任务队列已满");
        pthread_mutex_unlock(&pool->mutex);
        return;
    }
    task_t task = {.func = func,.args = args};
    pool->queue[pool->rear] = task;
    pool->rear = (pool->rear + 1)%MAX_QUEUE;
    pool->size++;
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
}

void tool_shutdown(thread_pool_t* pool){
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->num_threads; i++)
        pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
}

void* test(void* args){
    int* tid = (int*) args;
    printf("num: %d, pthread_id: %lu\n",tid,pthread_self());
    return NULL;
}

int main(){
    thread_pool_t pool;
    pool_init(&pool,4);
    printf("pool init() finished.\n");
    for(int i=0;i<10;i++){
        add_task(&pool,test,(void*)i);
        printf("task %d is added.\n",i);
    }
    
    tool_shutdown(&pool);
    return 0;

}