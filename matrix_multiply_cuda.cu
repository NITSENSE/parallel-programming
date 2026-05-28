#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

// Чтение матрицы из текстового файла (float)
std::vector<float> read_matrix(const std::string& filename, int n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка открытия " << filename << std::endl;
        exit(1);
    }
    std::vector<float> mat(n * n);
    for (int i = 0; i < n * n; ++i) {
        if (!(file >> mat[i])) {
            std::cerr << "Ошибка чтения данных из " << filename << std::endl;
            exit(1);
        }
    }
    return mat;
}

// Запись матрицы в файл с высокой точностью
void write_matrix(const std::string& filename, const float* mat, int n) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка записи в " << filename << std::endl;
        exit(1);
    }
    file << std::fixed << std::setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << mat[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
}

// CUDA-ядро: умножение матриц с использованием разделяемой памяти
// BLOCK_SIZE — размер тайла (компилируется с -DBLOCK_SIZE=...)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Загрузка тайла A
        int a_row = row;
        int a_col = tile * BLOCK_SIZE + tx;
        if (a_row < n && a_col < n)
            As[ty][tx] = A[a_row * n + a_col];
        else
            As[ty][tx] = 0.0f;

        // Загрузка тайла B
        int b_row = tile * BLOCK_SIZE + ty;
        int b_col = col;
        if (b_row < n && b_col < n)
            Bs[ty][tx] = B[b_row * n + b_col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Частичное произведение
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Вспомогательная функция для проверки ошибок CUDA
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Формат: " << argv[0] << " <размер_матрицы>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Размер должен быть положительным числом." << std::endl;
        return 1;
    }

    // Загрузка исходных матриц (CPU)
    std::vector<float> h_A = read_matrix("data/A.txt", n);
    std::vector<float> h_B = read_matrix("data/B.txt", n);
    std::vector<float> h_C(n * n, 0.0f);

    // Выделение памяти GPU
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(float)));

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Конфигурация сетки
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Замер времени (только выполнение ядра)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;

    // Копирование результата обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Запись результата в файл
    write_matrix("data/C.txt", h_C.data(), n);

    // Очистка памяти GPU
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Вывод метрик
    double gflops = (2.0 * n * n * n) / seconds / 1e9;
    std::cout << "Размер матрицы: " << n << "x" << n << std::endl;
    std::cout << "Время: " << seconds << " секунд" << std::endl;
    std::cout << "Производительность: " << gflops << " GFLOPS" << std::endl;

    return 0;
}