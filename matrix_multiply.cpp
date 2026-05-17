#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <iomanip>   // для std::setprecision

using namespace std;

// Чтение матрицы из файла: подряд идущие числа, порядок row-major
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

// Запись матрицы в файл с высокой точностью (15 знаков после запятой)
void write_matrix(const string& filename, const vector<double>& mat, int n) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка записи в " << filename << endl;
        exit(1);
    }
    file << fixed << setprecision(15);   // фиксированная точка, 15 цифр после запятой
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << mat[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
}

// Умножение матриц C = A * B, порядок циклов i-k-j
void multiply(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    C.assign(n * n, 0.0);
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
    if (argc != 2) {
        cerr << "Формат: " << argv[0] << " <размер_матрицы>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        cerr << "Размер должен быть положительным числом." << endl;
        return 1;
    }

    // Загрузка исходных данных
    vector<double> A = read_matrix("data/A.txt", n);
    vector<double> B = read_matrix("data/B.txt", n);
    vector<double> C;

    // Замер времени
    auto t1 = chrono::steady_clock::now();
    multiply(A, B, C, n);
    auto t2 = chrono::steady_clock::now();
    chrono::duration<double> elapsed = t2 - t1;

    // Сохранение результата с высокой точностью
    write_matrix("data/C.txt", C, n);

    // Вывод метрик
    double gflops = (2.0 * n * n * n) / elapsed.count() / 1e9;
    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Время: " << elapsed.count() << " секунд" << endl;
    cout << "Производительность: " << gflops << " GFLOPS" << endl;

    return 0;
}