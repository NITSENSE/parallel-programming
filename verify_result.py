import sys
import numpy as np

def verify(n: int):
    A = np.loadtxt("data/A.txt", dtype=np.float32).reshape(n, n)
    B = np.loadtxt("data/B.txt", dtype=np.float32).reshape(n, n)
    C = np.loadtxt("data/C.txt", dtype=np.float32).reshape(n, n)
    C_expected = A @ B
    if np.allclose(C, C_expected, rtol=1e-4, atol=1e-5):
        print("Проверка пройдена: результат CUDA совпадает с NumPy.")
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