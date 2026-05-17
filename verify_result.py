import sys
import numpy as np

def verify(n: int):
    # Загружаем матрицы из текстовых файлов (без заголовков)
    A = np.loadtxt("data/A.txt", dtype=np.float64).reshape(n, n)
    B = np.loadtxt("data/B.txt", dtype=np.float64).reshape(n, n)
    C = np.loadtxt("data/C.txt", dtype=np.float64).reshape(n, n)

    # Эталонное умножение через NumPy
    C_expected = A @ B

    if np.allclose(C, C_expected, rtol=1e-10, atol=1e-12):
        print("Проверка пройдена: результат C++ совпадает с NumPy.")
    else:
        print("Проверка НЕ пройдена! Обнаружены расхождения.")
        max_err = np.max(np.abs(C - C_expected))
        print(f"Максимальная абсолютная ошибка: {max_err}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python verify_result.py <размер>")
        sys.exit(1)
    n = int(sys.argv[1])
    verify(n)
