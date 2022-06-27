#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

using namespace std;

//length of matrix
int N = 1024;
// if matrix size is less than limit it will run with noraml matrix multiplication
int Limit = 1024;
int BLOCK_SIZE = 32;

//generate random data for the matrix
void generateMatrix(int *M){
    for(int i = 0 ; i < N*N ; i++){
        M[i] = rand()%100;
    }
}


//spilt big matric to 4 parts 
__global__ void split(int *X11, int *X12, int *X21, int *X22, int *X, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		X11[i * n + j] = X[i * 2 * n + j];
		X12[i * n + j] = X[i * 2 * n + j + n];
		X21[i * n + j] = X[(i + n) * 2 * n + j];
		X22[i * n + j] = X[(i + n) * 2 * n + j + n];
	}
}


//add two matrix A and B and put the result in C
__global__ void add(int *A, int *B, int *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = A[i * n + j] + B[i * n + j];
	}
}


//sun two matrixes A abd B and put the result in C
__global__ void sub(int *A, int *B, int *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = A[i * n + j] - B[i * n + j];
	}
}


//multiplicate two matrixes A and B and put the result in C
__global__ void mul(int *A, int *B, int *C, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = 0;
		for(int k = 0; k < n; k++) {
			C[i * n + j] += A[i * n + k] * B[k * n + j];
		}
	}
}


// merge C11 and c12 and C21 and C22 together and put them in C
__global__ void merge(int *C11, int *C12, int *C21, int *C22, int *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * 2 * n + j] = C11[i * n + j];
		C[i * 2 * n + j + n] = C12[i * n + j];
		C[(i + n) *2 * n + j] = C21[i * n + j];
		C[(i + n) * 2 * n + j + n] = C22[i * n + j];
	}
}


//sterassen algorithm for matrix multiplication
void sterassen(int* A , int* B, int* C , int n){
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	int *A_gpu, *B_gpu, *C_gpu;
	cudaMalloc((void **)&A_gpu, sizeof(int) * n * n);
	cudaMalloc((void **)&B_gpu, sizeof(int) * n * n);
	cudaMalloc((void **)&C_gpu, sizeof(int) * n * n);
	cudaMemcpy(A_gpu, A, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(int) * n * n, cudaMemcpyHostToDevice);

	//in N is less than limit calculate the multiplication with cuda mul
	if (n <= Limit){
		dim3 grid((size_t)ceil((int)n / (int)block.x), (size_t)ceil((int)n / (int)block.y));
		mul<<<grid, block>>>(A_gpu, B_gpu, C_gpu, n);
		cudaThreadSynchronize();
		cudaMemcpy(C, C_gpu, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
		cudaFree(A_gpu);
		cudaFree(B_gpu);
		cudaFree(C_gpu);
		return;
	}
	
	//allocate the data
	int *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22, *M1, *M2 ,*M3 , *M4 , *M5 , *M6 , *M7 , *T1 , *T2;
	int m = n / 2 ;
	cudaMalloc((void **)&A11, sizeof(int) * m * m);
	cudaMalloc((void **)&A12, sizeof(int) * m * m);
	cudaMalloc((void **)&A21, sizeof(int) * m * m);
	cudaMalloc((void **)&A22, sizeof(int) * m * m);
	cudaMalloc((void **)&B11, sizeof(int) * m * m);
	cudaMalloc((void **)&B12, sizeof(int) * m * m);
	cudaMalloc((void **)&B21, sizeof(int) * m * m);
	cudaMalloc((void **)&B22, sizeof(int) * m * m);
	cudaMalloc((void **)&C11, sizeof(int) * m * m);
	cudaMalloc((void **)&C12, sizeof(int) * m * m);
	cudaMalloc((void **)&C21, sizeof(int) * m * m);
	cudaMalloc((void **)&C22, sizeof(int) * m * m);
	cudaMalloc((void **)&M1, sizeof(int) * m * m);
	cudaMalloc((void **)&M2, sizeof(int) * m * m);
	cudaMalloc((void **)&M3, sizeof(int) * m * m);
	cudaMalloc((void **)&M4, sizeof(int) * m * m);
	cudaMalloc((void **)&M5, sizeof(int) * m * m);
	cudaMalloc((void **)&M6, sizeof(int) * m * m);
	cudaMalloc((void **)&M7, sizeof(int) * m * m);
	cudaMalloc((void **)&T1, sizeof(int) * m * m);
	cudaMalloc((void **)&T2, sizeof(int) * m * m);
	dim3 grid((size_t)ceil((int)m / (int)block.x), (size_t)ceil((int)m / (int)block.y));
	//split the matrix A to 4 parts
	split<<<grid, block>>>(A11, A12, A21, A22, A_gpu, m);
	cudaThreadSynchronize();
	//split the matrix B to 4 parts
	split<<<grid, block>>>(B11, B12, B21, B22, B_gpu, m);
	cudaThreadSynchronize();
	
	//M1
	add<<<grid, block>>>(B12, B22, T1, m);
	cudaThreadSynchronize();
	sterassen(A11 , T1 ,M1 , m);
	cudaThreadSynchronize();
	//M2
	add<<<grid, block>>>(A11, A12, T1, m);
	cudaThreadSynchronize();
	sterassen(T1 , B22 , M2 , m);
	cudaThreadSynchronize();
	//M3
	add<<<grid, block>>>(A21, A22, T1, m);
	cudaThreadSynchronize();
	sterassen(T1 , B11, M3 , m);
	cudaThreadSynchronize();
	//M4
	sub<<<grid, block>>>(B21, B11, T1, m);
	cudaThreadSynchronize();
	sterassen(A22 , T1 , M4 , m );
	cudaThreadSynchronize();
	//M5
	add<<<grid, block>>>(A11, A22, T1, m);
	cudaThreadSynchronize();
	add<<<grid, block>>>(B11, B22, T2, m);
	cudaThreadSynchronize();
	sterassen(T1 , T2 , M5 , m);
	cudaThreadSynchronize();
	//M6
	sub<<<grid, block>>>(A12, A22, T1, m);
	cudaThreadSynchronize();
	add<<<grid, block>>>(B21, B22, T2, m); 
	cudaThreadSynchronize();
	sterassen(T1 , T2 , M6 , m);
	cudaThreadSynchronize();
	//M7
	sub<<<grid, block>>>(A11, A21, T1, m);
	cudaThreadSynchronize();
	add<<<grid, block>>>(B11, B12, T2, m);
	cudaThreadSynchronize();
	sterassen(T1 , T2 , M7 , m);
	cudaThreadSynchronize();

	//C11
	add<<<grid, block>>>(M5, M4, T1, m);
	cudaThreadSynchronize();
	sub<<<grid, block>>>(T1, M2, T2, m);
	cudaThreadSynchronize();
	add<<<grid, block>>>(T2, M6, C11, m);
	cudaThreadSynchronize();

	//C12
	add<<<grid, block>>>(M1, M2, C12, m);
	cudaThreadSynchronize();

	//C21
	add<<<grid, block>>>(M3, M4, C21, m);
	cudaThreadSynchronize();

	//C22
	add<<<grid, block>>>(M5, M1, T1, m);
	cudaThreadSynchronize();
	sub<<<grid, block>>>(T1, M3, T2, m);
	cudaThreadSynchronize();
	sub<<<grid, block>>>(T2, M7, C22, m);
	cudaThreadSynchronize();

	//merege the C11 , C12 , C21 , C22
	merge<<<grid, block>>>(C11, C12, C21, C22, C_gpu, m);	
	cudaThreadSynchronize();

	cudaMemcpy(C, C_gpu, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
	
	//free the allocated memory
	cudaFree(A11); 
	cudaFree(A12); 
	cudaFree(A21); 
	cudaFree(A22); 
	cudaFree(B11); 
	cudaFree(B12); 
	cudaFree(B21); 
	cudaFree(B22); 
	cudaFree(T1);
	cudaFree(T2);
	cudaFree(M1);
	cudaFree(M2);
	cudaFree(M3);
	cudaFree(M4);
	cudaFree(M5);
	cudaFree(M6);
	cudaFree(M7);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}

int main(){

	// size of the matrixes
    size_t bytes = N * N * sizeof(int);
    //allocate data for the hosr
    int *h_A;
    int *h_B;
    int *h_C;
    h_A = (int *)malloc(bytes);
    h_B = (int *)malloc(bytes);
    h_C = (int *)malloc(bytes);

    //generate random matrix
    generateMatrix(h_A);
    generateMatrix(h_B);

    clock_t start = clock();
	sterassen(h_A,h_B,h_C,N);
    clock_t end = clock();
    double time_taken = end- start;
    cout<<time_taken<<endl;

}