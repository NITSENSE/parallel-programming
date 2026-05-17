#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

using namespace std;

vector<double> read_matrix(const string& filename, int n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия " << filename << endl;
        exit(1);
    }
    vector<double> mat(n * n);
    for (int i = 0; i < n * n; ++i) {
        if (!(file >> mat[i])) {
            cerr << "Ошибка чтения данных из " << filename << endl;
            exit(1);
        }
    }
    return mat;
}

void write_matrix(const string& filename, const vector<double>& mat, int n) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка записи в " << filename << endl;
        exit(1);
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

void multiply_omp(const vector<double>& A, const vector<double>& B, vector<double>& C, int n, int num_threads) {
    C.assign(n * n, 0.0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        cerr << "Формат: " << argv[0] << " <размер_матрицы> [число_потоков]" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        cerr << "Размер должен быть положительным числом." << endl;
        return 1;
    }

    int num_threads = omp_get_max_threads();  // по умолчанию все доступные
    if (argc == 3) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) {
            cerr << "Число потоков должно быть положительным." << endl;
            return 1;
        }
    }

    vector<double> A = read_matrix("data/A.txt", n);
    vector<double> B = read_matrix("data/B.txt", n);
    vector<double> C;

    auto t1 = chrono::steady_clock::now();
    multiply_omp(A, B, C, n, num_threads);
    auto t2 = chrono::steady_clock::now();
    chrono::duration<double> elapsed = t2 - t1;

    write_matrix("data/C.txt", C, n);

    double gflops = (2.0 * n * n * n) / elapsed.count() / 1e9;
    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Число потоков: " << num_threads << endl;
    cout << "Время: " << elapsed.count() << " секунд" << endl;
    cout << "Производительность: " << gflops << " GFLOPS" << endl;

    return 0;
}
