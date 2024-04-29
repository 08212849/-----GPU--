#include<stdio.h>
#include<math.h>
#include<sys/time.h>

const int nX = 1<<14;
const int nY = 1<<14;

// 计时函数
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

void __global__ addMaxtrix1Dto1D(int *a,  int *b,  int *c);
void check(const int *a, const int *b);
void HostaddMaxtrix(int *a, int *b,int *c);

int main(int argc, char *argv[]){
    const int N = nX * nY;
    const int M = sizeof(int) * N;

    int *a = (int*) malloc(M);
    int *b = (int*) malloc(M);
    int *c = (int*) malloc(M);
    int *d = (int*) malloc(M);

    for(int i=0;i<N;i++){
        a[i] = random() % 10;
        b[i] = random() % 10;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, M);
    cudaMalloc((void **)&d_b, M);
    cudaMalloc((void **)&d_c, M);

    double iStart = cpuSecond();
    HostaddMaxtrix(a, b, d);
    double iElaps = (cpuSecond() - iStart);
    printf("cpu addMatrix 1D&1D(%d & %d) time: %.10f seconds\n",nX, nY, iElaps);

    cudaMemcpy(d_a, a, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, M, cudaMemcpyHostToDevice);
    
    // 16384 = 64 * 64 * 256 *256
    dim3 block_size(64);
    dim3 grid_size((nX + block_size.x - 1)/block_size.x);
    printf("grid_size: (%d)\n",grid_size.x);
    printf("block_size: (%d)\n",block_size.x);

    iStart = cpuSecond();
    addMaxtrix1Dto1D<<<grid_size, block_size>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    iElaps = (cpuSecond() - iStart);
    printf("gpu addMatrix 1D&1D(%d & %d) time: %.10f seconds\n",nX, nY, iElaps);
    
    cudaMemcpy(c, d_c, M, cudaMemcpyDeviceToHost);
    check(c, d);

    free(a);
    free(b);
    free(c);
    free(d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

void __global__ addMaxtrix1Dto1D(int *a,  int *b,  int *c){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    for(int iy = 0;iy < nY;iy++){
        int idx = iy*nX + ix;
        c[idx] = a[idx] + b[idx];
    }
    
}

void check(const int *a, const int *b){
    const int N = nX * nY;
    bool hasError = false;
    int errorNum = 0;
    for(int i=0;i<N;i++){
        if(a[i] != b[i]){
            hasError = true;
            errorNum ++;
            // printf("GPU result: %d CPU result: %d\n",a[i],b[i]);
        }
    }
    printf("%s\n", hasError ? "Has errors" : "No errors");
    printf("error NUM:%d \n",errorNum);
}

void HostaddMaxtrix(int *a, int *b,int *c){
    for(int i=0;i<nX*nY;i++){
        c[i] = a[i] + b[i];
    }
}