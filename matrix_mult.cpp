#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <mpi.h>

using namespace std;
using namespace chrono;

vector<vector<int>> readMatrix(const string& filename, int size) {
    vector<vector<int>> matrix(size, vector<int>(size));
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}

void writeResult(const vector<vector<int>>& matrix, double time, long long operations,
    int num_processes, int rank, const string& filename) {
    if (rank == 0) {
        ofstream file(filename);
        file << "Результирующая матрица:\n";
        for (const auto& row : matrix) {
            for (int val : row) {
                file << setw(8) << val << " ";
            }
            file << "\n";
        }
        file << "\n";
        file << "Время выполнения: " << time << " мкс (" << time / 1000000.0 << " с)\n";
        file << "Объем задачи: " << operations << " операций\n";
        file << "Размер матрицы: " << matrix.size() << "x" << matrix.size() << "\n";
        file << "Количество процессов: " << num_processes << "\n";
    }
}

int main(int argc, char** argv) {
    system("chcp 65001 > nul");
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int size;
    string base = "C:/Users/gayvo/sekas/";

    if (rank == 0) {
        cout << "Введите размер матрицы: ";
        cin >> size;
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<int>> A, B, C;

    if (rank == 0) {
        A = readMatrix(base + "matrix_a.txt", size);
        B = readMatrix(base + "matrix_b.txt", size);
        C.assign(size, vector<int>(size, 0));
    }

    auto start = high_resolution_clock::now();

    int rows_per_process = size / num_processes;
    int remainder = size % num_processes;

    int start_row, end_row;
    if (rank < remainder) {
        start_row = rank * (rows_per_process + 1);
        end_row = start_row + rows_per_process;
    }
    else {
        start_row = rank * rows_per_process + remainder;
        end_row = start_row + rows_per_process - 1;
    }

    int local_rows = end_row - start_row + 1;

    vector<int> flat_B(size * size);
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                flat_B[i * size + j] = B[i][j];
            }
        }
    }

    vector<int> local_B(size * size);
    MPI_Bcast(flat_B.data(), size * size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.assign(size, vector<int>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                B[i][j] = flat_B[i * size + j];
            }
        }
    }

    vector<int> flat_A_local(local_rows * size);

    if (rank == 0) {
        for (int p = 0; p < num_processes; p++) {
            int p_start_row, p_end_row;
            if (p < remainder) {
                p_start_row = p * (rows_per_process + 1);
                p_end_row = p_start_row + rows_per_process;
            }
            else {
                p_start_row = p * rows_per_process + remainder;
                p_end_row = p_start_row + rows_per_process - 1;
            }

            int p_local_rows = p_end_row - p_start_row + 1;
            vector<int> p_flat_A(p_local_rows * size);

            for (int i = p_start_row; i <= p_end_row; i++) {
                for (int j = 0; j < size; j++) {
                    p_flat_A[(i - p_start_row) * size + j] = A[i][j];
                }
            }

            if (p == 0) {
                flat_A_local = p_flat_A;
            }
            else {
                MPI_Send(p_flat_A.data(), p_local_rows * size, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(flat_A_local.data(), local_rows * size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    vector<vector<int>> local_A(local_rows, vector<int>(size));
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < size; j++) {
            local_A[i][j] = flat_A_local[i * size + j];
        }
    }
    vector<vector<int>> local_C(local_rows, vector<int>(size, 0));

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < size; j++) {
            int sum = 0;
            for (int k = 0; k < size; k++) {
                sum += local_A[i][k] * B[k][j];
            }
            local_C[i][j] = sum;
        }
    }

    if (rank == 0) {
        for (int i = start_row; i <= end_row; i++) {
            for (int j = 0; j < size; j++) {
                C[i][j] = local_C[i - start_row][j];
            }
        }
        for (int p = 1; p < num_processes; p++) {
            int p_start_row, p_end_row;
            if (p < remainder) {
                p_start_row = p * (rows_per_process + 1);
                p_end_row = p_start_row + rows_per_process;
            }
            else {
                p_start_row = p * rows_per_process + remainder;
                p_end_row = p_start_row + rows_per_process - 1;
            }

            int p_local_rows = p_end_row - p_start_row + 1;
            vector<int> p_flat_C(p_local_rows * size);

            MPI_Recv(p_flat_C.data(), p_local_rows * size, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = p_start_row; i <= p_end_row; i++) {
                for (int j = 0; j < size; j++) {
                    C[i][j] = p_flat_C[(i - p_start_row) * size + j];
                }
            }
        }
    }
    else {
        vector<int> p_flat_C(local_rows * size);
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < size; j++) {
                p_flat_C[i * size + j] = local_C[i][j];
            }
        }
        MPI_Send(p_flat_C.data(), local_rows * size, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    double duration_double = static_cast<double>(duration);

    long long operations = 2LL * size * size * size;
    double max_duration;
    MPI_Reduce(&duration_double, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        writeResult(C, max_duration, operations, num_processes, rank, base + "result_mpi.txt");

        cout << "Результат сохранен в " << base << "result_mpi.txt\n";
        cout << "Время выполнения: " << max_duration << " мкс (" << max_duration / 1000000.0 << " с)\n";
        cout << "Объем задачи: " << operations << " операций\n";
        cout << "Размер матрицы: " << size << "x" << size << "\n";
        cout << "Количество процессов: " << num_processes << "\n";
    }

    MPI_Finalize();
    return 0;
}