#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <mpi.h>

using namespace std;

// Чтение матрицы из текстового файла (row-major)
vector<double> read_matrix(const string& filename, int n) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка открытия " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    vector<double> mat(n * n);
    for (int i = 0; i < n * n; ++i) {
        if (!(file >> mat[i])) {
            cerr << "Ошибка чтения из " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    return mat;
}

// Запись матрицы в файл с высокой точностью
void write_matrix(const string& filename, const vector<double>& mat, int n) {
    ofstream file(filename);
    if (!file) {
        cerr << "Ошибка записи в " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    file << fixed << setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << mat[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
}

// Локальное умножение: перемножает фрагмент строк A на полную матрицу B
void local_multiply(const double* local_A, const double* B, double* local_C,
                    int local_rows, int n) {
    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = local_A[i * n + k];
            for (int j = 0; j < n; ++j) {
                local_C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) cerr << "Формат: " << argv[0] << " <размер_матрицы>" << endl;
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) cerr << "Размер должен быть положительным." << endl;
        MPI_Finalize();
        return 1;
    }

    // Процесс 0 читает матрицы и рассылает размер
    vector<double> A, B;
    if (rank == 0) {
        A = read_matrix("data/A.txt", n);
        B = read_matrix("data/B.txt", n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Рассылка матрицы B целиком всем процессам
    if (rank != 0) B.resize(n * n);
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк матрицы A (используем Scatterv)
    vector<int> sendcounts(size), displs(size);
    vector<int> recvcounts(size), rdispls(size); // для Gatherv

    int base_rows = n / size;
    int remainder = n % size;
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int rows = (i < remainder) ? base_rows + 1 : base_rows;
        sendcounts[i] = rows * n;
        displs[i] = offset * n;
        recvcounts[i] = rows * n;
        rdispls[i] = offset * n;
        offset += rows;
    }

    int local_rows = (rank < remainder) ? base_rows + 1 : base_rows;
    vector<double> local_A(local_rows * n);
    vector<double> local_C(local_rows * n, 0.0);

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Барьер и замер времени только для вычислений
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    local_multiply(local_A.data(), B.data(), local_C.data(), local_rows, n);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Сбор результирующей матрицы на процессе 0
    vector<double> C;
    if (rank == 0) C.resize(n * n);
    MPI_Gatherv(local_C.data(), local_rows * n, MPI_DOUBLE,
                C.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Запись результата и вывод времени
    if (rank == 0) {
        write_matrix("data/C.txt", C, n);
        double elapsed = end_time - start_time;
        double gflops = (2.0 * n * n * n) / elapsed / 1e9;
        cout << "Размер матрицы: " << n << "x" << n << endl;
        cout << "Число процессов: " << size << endl;
        cout << "Время: " << elapsed << " секунд" << endl;
        cout << "Производительность: " << gflops << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}
