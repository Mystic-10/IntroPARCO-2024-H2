/*
 *    Nicolò Bellè
 *    238178
 *    DELIVERABLE 2
 *    nicolo.belle@studenti.unitn.it
 */

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>
#define N_RUNS 10
int rank, size;

double wt_symSequential, wt_symMPI, wt_sequentialTransposition, wt_mpiTransposition, wt_mpiTransposition_total;

//-->init n with scanf: matrix size
int init_matrixSize();

//-->check_input_(atoi)
bool check_invalid_input_atoi(int argc, char const *argv);

//-->allocate_matrix_malloc
float **allocate_matrix(int r, int c);

//-->deallocate_matrix_free
void deallocate_matrix(float **matrix, const int r);

//-->init_random_float_matrix
float **init_random_float_matrix(const int r, const int c);

//-->printMatrix (delete internal comment to print all matrix)
void printMatrix(float **printMatrix, const int r, const int c);

//-->checkSymmetrical
bool checkSym(float **M, const int r, const int c);
bool checkSymMPI(float **M, const int r, const int c);

//-->transpose
float **matTranspose(float **Mc, const int r, const int c);
float **matTransposeMPI(float **Mc, const int r, const int c);

//-->checkEqualMatrix
bool checkEqualMatrix(float **Mc, float **Tc, const int r, const int c);

// int main(int argc, char const *argv[])
int main(int argc, char **argv)
{
    // init_mpi
    MPI_Init(&argc, &argv);

    // init_rank_and_size_mpi
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));

    //----------(scanf)----------
    // int n = init_matrixSize(); //if uncomment line 'scanf' --> comment lines 'atoi'
    //----------           end           ----------

    //----------(atoi)-----------
    if (check_invalid_input_atoi(argc, argv[1])) // if uncomment lines 'atoi' --> comment line 'scanf'
    {
        return 0;
    }
    int n = atoi(argv[1]);
    //----------           end           ----------

    if (rank == 0)
    {
        //-->print_size_matrix_n
        printf("\n-------------------------------- MATRIX SIZE (n) = %d --------------------------------\n", n);
    }

    if ((n % size != 0)) {
        if (rank == 0) {
            printf("The matrix size (%d) must be divided evenly by the number of processes (%d).\n", n, size);
        }
        MPI_Finalize();
        return 1;
    }

    //-->declarationMatrix
    float **M, **T, **T_mpi;

    //-->init_random_float_matrix
    M = init_random_float_matrix(n, n);

    //-->UNCOMMENT_TO_PRINT_MAIN_MATRIX
    /*if (rank == 0)
    {
        printf("\n-->PRINT MAIN MATRIX:\n");
        printMatrix(M, n, n);
    }*/

    if (rank == 0)
    {
        //-->checkSymmetric_sequential
        checkSym(M, n, n);
    }

    //-->checkSymmetric_mpi
    checkSymMPI(M, n, n);

    if (rank == 0)
    {
        //-->print_wall_clock_time_symmetrical_check
        printf("\n-->WALL CLOCK TIME SYMMETRICAL CHECK:\n");
        printf("\t-SEQUENTIAL: wall clock time SYMMETRICAL check = %12.4g sec\n", wt_symSequential);
        printf("\t-MPI: wall clock time SYMMETRICAL check: %12.4g sec\n", wt_symMPI);
    }

    double total_dataTransfered_bandwidth = 2.0 * n * n * sizeof(float);
    double avg_time_parallel_sequential;
    if (rank == 0)
    {
        //-->init_file_sequential
        FILE *file_si = fopen("deliverable2_seq.csv", "a");
        if (file_si == NULL)
        {
            perror("ERROR: failed to open file\n");
            return 1;
        }
        //-->header_file_sequential
        if (n == 64)
        {
            fprintf(file_si, "NUMBER_OF_PROCESSES=%d\n", size);
            fprintf(file_si, "SIZE_n;AVG_TIME_SEQ;\n");
        }

        //-->run_sequential_transposition
        double total_time_parallel_sequential;
        total_time_parallel_sequential = 0.0;
        for (int r = 0; r < N_RUNS; r++)
        {
            T = matTranspose(M, n, n);
            total_time_parallel_sequential += wt_sequentialTransposition;
        }

        //-->compute_avg_time_sequential
        avg_time_parallel_sequential = total_time_parallel_sequential / N_RUNS;

        //-->bandwidth_sequential
        double bandwidth_sequential = total_dataTransfered_bandwidth / (avg_time_parallel_sequential * 1e9);

        printf("\n-->STATS MATRICES TRANSPOSITION:\n");
        //-->print_stats_sequential
        printf("\t-SEQUENTIAL:");
        printf("\n\t average wall clock time TRANSPOSED MATRIX: %7.9f sec\n", avg_time_parallel_sequential);
        printf("\t effective BANDWIDTH: %3.3f GB/s\n", bandwidth_sequential);

        fprintf(file_si, "%11d;%7.9f\n", n, avg_time_parallel_sequential);
        //-->close_csv
        fclose(file_si);
    }

    FILE *file;
    if (rank == 0)
    {
        //-->init_file_mpi
        file = fopen("deliverable2_mpi.csv", "a");
        if (file == NULL)
        {
            perror("ERROR: failed to open file\n");
            return 1;
        }
        //-->header_file_mpi
        if (n == 64)
        {
            fprintf(file, "NUMBER_OF_PROCESSES=%d\n", size);
            fprintf(file, "n;AVG_TIME_TOT_MPI;AVG_TIME_FOR_MPI;SPEED_UP;BANDWIDTH;EFFICIENCY\n");
        }
    }

    //-->run_mpi
    T_mpi = matTransposeMPI(M, n, n);
    if (rank == 0)
    {
        //-->compute_stats_mpi: avg time, speed up and bandwidth
        double avg_time_parallel_mpi = wt_mpiTransposition;
        double avg_time_parallel_mpi_total = wt_mpiTransposition_total;
        double avg_speedup_mpi = avg_time_parallel_sequential / avg_time_parallel_mpi;
        double avg_efficiency_mpi = avg_speedup_mpi / size;
        double bandwidth_mpi = total_dataTransfered_bandwidth / (avg_time_parallel_mpi * 1e9);
        //-->print_stats_mpi
        printf("\n\t-MPI:");
        printf("\n\t average wall clock time, speed up and bandwidth TRANSPOSED MATRIX:\n");
        //-->header_terminal
        printf("\n\t  AVG TIME TOT  |  AVG TIME FOR   | SPEED UP |  BANDWIDTH  |  EFFICIENCY\n");
        printf("\t%7.9f sec | %7.9f sec |  %5.3f   |    %3.3f    | %8.2f%% \n", avg_time_parallel_mpi_total, avg_time_parallel_mpi, avg_speedup_mpi, bandwidth_mpi, avg_efficiency_mpi * 100);

        //-->print_stats_file_csv
        fprintf(file, "%11d;%7.9f;%7.9f;%6.2f;%3.3f;%8.2f\n", n, avg_time_parallel_mpi_total, avg_time_parallel_mpi, avg_speedup_mpi, bandwidth_mpi, avg_efficiency_mpi * 100);

        //-->close_csv
        fclose(file);
    }

    //-->UNCOMMENT_TO_PRINT_TRANSPOSED_MATRIX
    /*if (rank == 0)
    {
        printf("\n-->PRINT TRASPOSED MATRIX (sequential):\n");
        printMatrix(T, n, n);
        printf("\n-->PRINT TRASPOSED MATRIX (mpi):\n");
        printMatrix(T_mpi, n, n);
    }*/

    if (rank == 0)
    {
        // checkEqualMatrix
        printf("\nExample to test the function: main matrix compared to transposed matrix");
        printf("\n(must be NOT EQUAL if the array does not contain the same values or size != 1)");
        checkEqualMatrix(M, T, n, n);
        printf("\n");
        printf("Example to test the function: transposed matrix sequential compared to transposed matrix mpi");
        printf("\n(must be EQUAL)");
        checkEqualMatrix(T, T_mpi, n, n);
        printf("\n");
    }

    // deallocate all matrix
    deallocate_matrix(M, n);
    deallocate_matrix(T, n);
    free(T_mpi);

    if (rank == 0)
    {
        printf("\nAll matrices DEALLOCATED\n\n");
    }

    // terminate_mpi
    MPI_Finalize();

    return 0;
} //-->end_main

//-->init n with scanf: matrix size
int init_matrixSize()
{

    long int n;
    if (rank == 0)
    {
        bool check_returnScanf, check_nPowerOfTwo, check_outOfRange;
        do
        {
            printf("\nINSERT n: ");
            fflush(stdin);
            check_returnScanf = scanf("%ld", &n);
            if (check_returnScanf != true || n <= 0)
            {
                printf("\nINVALID INPUT: 'n' must be an integer greater than 0!\n");
                check_returnScanf = false;
            }
            if (n < INT_MIN || n > INT_MAX)
            {
                printf("\nINVALID INPUT: 'n' out of integer range!\n");
                check_outOfRange = true;
            }
            else
            {
                check_outOfRange = false;
            }
            if ((n > 0) && ((n & (n - 1)) == 0))
            {
                check_nPowerOfTwo = true;
            }
            else
            {
                check_nPowerOfTwo = false;
            }
            if (check_nPowerOfTwo != true && check_returnScanf)
            {
                printf("\nINVALID INPUT: 'n' must be a power of two!\n");
            }

        } while (check_returnScanf != true || n <= 0 || check_nPowerOfTwo != true || check_outOfRange == true);
    }
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    return (int)n;
}

//-->check_input_(atoi)
bool check_invalid_input_atoi(int argc, char const *argv)
{

    bool is_in_argv = true, is_numeric = true, check_nPowerOfTwo;
    if (argc < 2)
    {
        if (rank == 0)
        {
            printf("\nINVALID INPUT: insert size 'n'!\n\n");
        }
        return true;
    }
    for (int i = 0; argv[i] != '\0'; i++)
    {
        if (!isdigit(argv[i]))
        {
            if (rank == 0)
            {
                printf("\nINVALID INPUT: 'n' must be an integer greater than 0!\n\n");
            }
            return true;
        }
    }
    char *endptr;
    errno = 0;
    long int n = strtol(argv, &endptr, 10);
    if (errno == ERANGE || n < INT_MIN || n > INT_MAX)
    {
        if (rank == 0)
        {
            printf("\nINVALID INPUT: 'n' out of integer range!\n\n");
        }
        return 1;
    }
    if (n <= 0)
    {
        if (rank == 0)
        {
            printf("\nINVALID INPUT: 'n' must be an integer greater than 0!\n\n");
        }

        return true;
    }
    if ((n > 0) && ((n & (n - 1)) == 0))
    {
        return false;
    }
    else
    {
        if (rank == 0)
        {
            printf("\nINVALID INPUT: 'n' must be a power of two!\n\n");
        }
        return true;
    }
}

//-->allocate_matrix_malloc
float **allocate_matrix(int r, int c)
{
    float **matrix = (float **)malloc(r * sizeof(float *));
    if (matrix == NULL)
    {
        if (rank == 0)
        {
            printf("\nERROR: Memory allocation failed! (allocate_matrix)\n");
        }
        return NULL;
    }
    for (int i = 0; i < c; i++)
    {
        matrix[i] = (float *)malloc(c * sizeof(float));
        if (matrix[i] == NULL)
        {
            if (rank == 0)
            {
                printf("\nERROR: Memory allocation failed for row %d! (allocate_matrix)\n", i);
            }
            return NULL;
        }
    }
    return matrix;
}

//-->deallocate_matrix_free
void deallocate_matrix(float **matrix, const int r)
{
    if (matrix == NULL)
    {
        return;
    }
    for (int i = 0; i < r; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

//-->init_random_float_matrix
float **init_random_float_matrix(const int r, const int c)
{
    float **Mc = allocate_matrix(r, c);
    if (Mc == NULL)
    {
        if (rank == 0)
        {
            printf("\nERROR: Memory allocation failed! (init_matrix)\n");
        }
        return NULL;
    }

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            Mc[i][j] = ((float)rand() / RAND_MAX) * 10.0;
        }
    }
    if (rank == 0)
    {
        printf("\nNEW MATRIX: created\n");
    }
    return Mc;
}

//-->printMatrix (delete internal comment to print matrix)
void printMatrix(float **printMatrix, const int r, const int c)
{
    if (printMatrix == NULL)
    {
        printf("\nERROR: unable to print, matrix null!\n");
        return;
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%0.3f ", printMatrix[i][j]);
        }
        printf("\n");
    }
}

//-->checkSymmetrical_sequential
bool checkSym(float **M, const int r, const int c)
{
    bool check = true;
    if (M == NULL)
    {
        if (rank == 0)
        {
            printf("\nERROR: unable to check symmetric, matrix null! (sequential)\n");
        }
        return false;
    }
    if (r != c)
    {
        check = false;
    }

    double wt1_sym, wt2_sym;
    wt1_sym = MPI_Wtime();
    for (int i = 0; i < r; i++)
    {
        for (int j = i + 1; j < c; j++)
        {
            if (M[i][j] != M[j][i])
                check = false;
        }
    }
    wt2_sym = MPI_Wtime();
    wt_symSequential = wt2_sym - wt1_sym;

    if (rank == 0)
    {
        printf("\nCHECK SYMMETRIC (sequential): ");
        if (check)
        {
            printf("SI\n");
        }
        else
        {
            printf("NO\n");
        }
    }

    return check;
}

//-->checkSymmetrical_omp
bool checkSymMPI(float **M, const int r, const int c)
{
    int check = 1;
    if (M == NULL)
    {
        if (rank == 0)
        {
            printf("ERROR: unable to check symmetric, matrix null! (omp)\n");
        }
        return 0;
    }
    if (r % size != 0)
    {
        if (rank == 0)
        {
            printf("Error: unable to distribute matrix to processes\n");
        }
        return 0;
    }
    if (r != c)
    {
        check = 0;
    }

    if (check)
    {
        int check_local = 1;
        int r_forProcess = r / size;
        int first_row_forProcess = rank * r_forProcess;
        double wt1_symMPI, wt2_symMPI;
        if (rank == 0)
        {

            wt1_symMPI = MPI_Wtime();
        }
        for (int i = 0; i < r_forProcess; i++)
        {
            for (int j = first_row_forProcess + i + 1; j < c; j++)
            {
                if (M[first_row_forProcess + i][j] != M[j][first_row_forProcess + i])
                {
                    check_local = 0;
                }
            }
        }
        if (rank == 0)
        {
            wt2_symMPI = MPI_Wtime();
            wt_symMPI = wt2_symMPI - wt1_symMPI;
        }
        MPI_Reduce(&check_local, &check, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        printf("CHECK SYMMETRIC (mpi): ");
        if (check)
        {
            printf("SI\n");
        }
        else
        {
            printf("NO\n");
        }
    }
    return check;
}

//-->transpose_sequential
float **matTranspose(float **Mc, const int r, const int c)
{
    if (Mc == NULL)
    {
        if (rank == 0)
        {
            printf("\nERROR: unable to transpose, matrix null! (sequential)\n");
        }
        return NULL;
    }
    float **Tc = allocate_matrix(r, c);
    if (Tc == NULL)
    {
        if (rank == 0)
        {
            printf("\nERROR: Memory allocation failed! (sequential)\n");
        }
        return NULL;
    }

    double wt1, wt2;
    if (rank == 0)
    {
        wt1 = MPI_Wtime();
    }

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            Tc[j][i] = Mc[i][j];
        }
    }
    if (rank == 0)
    {
        wt2 = MPI_Wtime();
        wt_sequentialTransposition = wt2 - wt1;
    }

    return Tc;
}

//-->transpose_omp
float **matTransposeMPI(float **Mc, const int r, const int c)
{
    // chech_if_main_matrix_is_NULL
    if (Mc == NULL)
    {
        if (rank == 0)
        {
            printf("ERROR: unable to transpose, matrix null! (mpi)\n");
        }
        return NULL;
    }

    // allocation_of_return_matrix
    float **T_mpi = NULL;
    if (rank == 0)
    {
        T_mpi = allocate_matrix(r, c);
        if (T_mpi == NULL)
        {
            printf("ERROR: Memory allocation failed! (mpi)\n");
            return NULL;
        }
    }
    // number_of_rows_per_process
    int r_forProcess = r / size;
    int first_row_forProcess = rank * r_forProcess;

    // copy_matrix_from_nonContiguous_to_contiguous_matrix
    float *transposed_contiguous_matrix = NULL;
    if (rank == 0)
    {
        transposed_contiguous_matrix = (float *)malloc(r * c * sizeof(float));
    }

    double wtMPI1_total, wtMPI2_total;
    if (rank == 0)
    {
        wtMPI1_total = MPI_Wtime();
    }

    float *mpi_transposed_local = (float *)malloc(r_forProcess * c * sizeof(float));
    
    double wtMPI1, wtMPI2;
    if (rank == 0)
    {
        wtMPI1 = MPI_Wtime();
    }

    for (int i = 0; i < r_forProcess; i++)
    {
        for (int j = 0; j < c; j++)
        {
            mpi_transposed_local[i * c + j] = Mc[j][i + first_row_forProcess];
        }
    }

    if (rank == 0)
    {  
        wtMPI2 = MPI_Wtime();
        wt_mpiTransposition = wtMPI2 - wtMPI1;
    }

    MPI_Gather(mpi_transposed_local, r_forProcess * c, MPI_FLOAT, transposed_contiguous_matrix, r_forProcess * c, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        wtMPI2_total = MPI_Wtime();
        wt_mpiTransposition_total = wtMPI2_total - wtMPI1_total;
        for (int i = 0; i < r; i++)
        {
            T_mpi[i] = &transposed_contiguous_matrix[i * r];
        }

        
    }
    free(mpi_transposed_local);

    return T_mpi;
}

//-->checkEqualMatrix
bool checkEqualMatrix(float **Mc, float **Tc, const int r, const int c)
{
    if (Mc == NULL || Tc == NULL)
    {
        printf("ERROR: unable to check symmetric, matrix null! (implicit)\n");
        return false;
    }
    printf("\n-->CHECK EQUAL: ");
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            if (Mc[i][j] != Tc[i][j])
            {
                printf("NO\n");
                return false;
            }
        }
    }
    printf("SI\n");
    return true;
}
