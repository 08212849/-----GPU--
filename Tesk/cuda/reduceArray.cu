#include<stdio.h>

#define TILE_DIM 32
#define N 1000

__global__ void reduceNeighbored(int *d_x, int *d_y)
{
	int tid = threadIdx.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int *x = d_x + blockIdx.x*blockDim.x;
	if (index >= N) return;
	for (int strize = 1; strize < blockDim.x; strize *= 2)
	{
 		if (tid % (2 * strize) == 0)
 		{ 	
            x[tid] += x[tid + strize]; 
        }
		__syncthreads();
 	}
 	if (tid == 0) { d_y[blockIdx.x] = x[0]; }
}

__global__ void reduceNeighbored3(int *d_x,int *d_y)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = tid + blockDim.x * blockIdx.x;
    __shared__ int s_y[TILE_DIM];
    s_y[tid] = (n < N)? d_x[n]:0;
    __syncthreads();
    
    for(int offset = blockDim.x >> 1;offset > 0;offset >>= 1){
        if(tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    
    if(tid == 0) d_y[bid] = s_y[0];
    // 使用原子函数
    // if(tid == 0) atomicAdd(&d_y[0],s_y[0]);
}

__global__ void reduce4(int *d_x,int *y){
    int tx = threadIdx.x;
    int tid = tx + blockDim.x * blockIdx.x;
    int temp = 0;
    __shared__ int S[TILE_DIM];
    for(int i = tid; i < N;i += gridDim.x * blockDim.x)
        temp += d_x[i];
    S[tx] = temp;

    for(int offset = blockDim.x >>1 ;offset > 0;offset >>=1){
        int double_kill = -1;
        if(tx < offset){
            double_kill = S[tx] + S[tx + offset];
        }
        __syncthreads();
        if(tx < offset){
            S[tx] = double_kill;
        }
        __syncthreads();
    }

    if(blockDim.x * blockIdx.x < N)
        if(tx == 0) atomicAdd(&y[0],S[0]);
}

int main(){
    int nx = N * sizeof(int);
    int* x = (int*)malloc(nx);
    int* y = (int*)malloc(nx);

    int* d_x,*d_y;
    cudaMalloc((void**)&d_x, nx);
    cudaMalloc((void**)&d_y, nx);

    int rResult = 0;
    for(int i=0;i<N;i++){
        x[i] = i+1;
        rResult += x[i];
    }
    cudaMemcpy(d_x,x,nx,cudaMemcpyHostToDevice);

    int grid_x = (N + TILE_DIM - 1) / TILE_DIM;
    dim3 grid_size(grid_x);
    dim3 block_size(TILE_DIM);

    reduce4<<<grid_size,block_size>>>(d_x,d_y);
    // reduceNeighbored<<<grid_size,block_size>>>(d_x,d_y);
    cudaDeviceSynchronize();
    cudaMemcpy(y,d_y,nx,cudaMemcpyDeviceToHost);

    int ans = 0;
    for(int i=0;i<grid_x;i++){
        ans += y[i];
        printf("y[%d]: %d\n",i,y[i]);
    }
        
    
    printf("rResult :%d ans: %d\n",rResult,ans);


    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
}