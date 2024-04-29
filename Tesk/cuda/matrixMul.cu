#include<stdio.h>
typedef long long int;
const int TILE_DIM = 32;
const int M = 4,N = 10, K = 2;

__global__ void matrixmul(const int *A, const int *B, int *C){   //M*N x N*K
    __shared__ int A_S[TILE_DIM][TILE_DIM];
    __shared__ int B_S[TILE_DIM][TILE_DIM];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;
    int value = 0;
    for (int ph = 0; ph < N / TILE_DIM + 1; ph++) {
        if (row < M && ph * TILE_DIM + tx < N)
            A_S[ty][tx] = A[row * N + ph * TILE_DIM + tx];
        else
            A_S[ty][tx] = 0;
        if (col < K && ph * TILE_DIM + ty < N) 
            B_S[ty][tx] = B[(ph * TILE_DIM + ty) * K + col];
        else
            B_S[ty][tx] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++)
            value += A_S[ty][k] * B_S[k][tx];
        __syncthreads();
    }

    if (row < M && col < K)
        C[row*K+col]=value;
}

int main(){
    int na = M * N * sizeof(int);
    int nb = N * K * sizeof(int);
    int nc = K * M * sizeof(int);
    int* A = (int*)malloc(na);
    int* B = (int*)malloc(nb);
    int* C = (int*)malloc(nc);
    int* D = (int*)malloc(nc);

    int* d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a, na);
    cudaMalloc((void**)&d_b, nb);
    cudaMalloc((void**)&d_c, nc);

    for(int i=0;i<M*N;i++){
        A[i] = 1;
        printf("%d ",A[i]);
        if((i+1) % N == 0) printf("\n");
    }
    for(int i=0;i<N*K;i++){
        B[i] = 1;
        printf("%d ",B[i]);
        if((i+1) % K == 0) printf("\n");
    }
    for(int i=0;i<M*K;i++){
        D[i] = 0;
    }

    for(int i=0;i<M;i++)
        for(int j=0;j<K;j++)
            for(int k=0;k<N;k++){
                D[i*K + j] += A[i*N + k] * B[k*K + j];
            }
    cudaMemcpy(d_a,A,na,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,nb,cudaMemcpyHostToDevice);

    int maxV = max(max(M,N),K);
    int grid_x = (maxV + TILE_DIM - 1) / TILE_DIM;
    int grid_y = (maxV + TILE_DIM - 1) / TILE_DIM;
    dim3 grid_size(grid_x, grid_y);
    dim3 block_size(TILE_DIM,TILE_DIM);


    matrixmul<<<grid_size,block_size>>>(d_a,d_b,d_c);
    // reduceNeighbored<<<grid_size,block_size>>>(d_x,d_y);
    cudaDeviceSynchronize();
    cudaMemcpy(C,d_c,nc,cudaMemcpyDeviceToHost);

    int errorNum = 0;
    for(int i=0;i<M*K;i++){
        printf("%d ",C[i]);
        if((i+1) % K == 0) printf("\n");
        if(C[i] != D[i]) errorNum++;
    }   
    printf("errNum: %d\n",errorNum); 

    free(A);
    free(B);
    free(C);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

