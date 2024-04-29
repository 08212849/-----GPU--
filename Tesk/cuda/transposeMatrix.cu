#include<stdio.h>

const int Rn = 400;
const int Cn = 10;

const int TILE_DIM = 32;
const int grid_size_x = (Cn + TILE_DIM - 1) / TILE_DIM;
const int grid_size_y = (Rn + TILE_DIM - 1) / TILE_DIM;
const dim3 block_size(TILE_DIM, TILE_DIM);
const dim3 grid_size(grid_size_x, grid_size_y);

//转置任意大小矩阵
__global__ void transpose(int * A,int* B,int rowNum, int colNum){
    __shared__ int S[TILE_DIM][TILE_DIM + 1];  //避免bank冲突
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;

    if(nx1 < colNum && ny1 < rowNum){
        S[threadIdx.y][threadIdx.x] = A[ny1 * colNum + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if(nx2 < colNum && ny2 < rowNum){
        B[nx2* rowNum + ny2] = S[threadIdx.x][threadIdx.y];
    }
 
}

//转置N * N矩阵
__global__ void transpose1(const int *A, int *B, const int N){  
    __shared__ int S[TILE_DIM][TILE_DIM+1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    //顺序读
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    __syncthreads();
    //顺序写
    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
}

int main(){
    int num = Rn * Cn;
    int nBytes = num * sizeof(int);
  
    //cpu空间分配
    int* A = (int*)malloc(nBytes);
    int* B = (int*)malloc(nBytes);
    int* C = (int*)malloc(nBytes);

    for(int i=0;i<num;i++){
        A[i] = i;
    }

    // for(int i=0;i<num;i++){
    //     printf("%d ",A[i]);
    //     if(i != 0 && (i+1) % Cn == 0) printf("\n");
    // }

    int* d_A, * d_B, *d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);
    cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
    transpose<<<grid_size, block_size >>> (d_A, d_B, Rn, Cn);
    // transpose1<<<grid_size, block_size >>> (d_A, d_C, Rn);
    cudaMemcpy(B, d_B, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, nBytes, cudaMemcpyDeviceToHost);
    for(int i=0;i<num;i++){
        printf("%d ",B[i]);
        if((i+1) % Rn == 0) printf("\n");
    }
    // for(int i=0;i<num;i++){
    //     printf("%d ",C[i]);
    //     if((i+1) % Rn == 0) printf("\n");
    // }
    int errorNum = 0;
    for(int i=0;i<num;i++){
        if(B[i] != C[i]) {
            errorNum++;
            // printf("B: %d, C: %d\n",B[i],C[i]);
        }
    }
    printf("%d\n",errorNum);
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}