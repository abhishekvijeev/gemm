#include <boost/program_options.hpp>
#include <iostream>

#include <parse_cmdline.h>

namespace po = boost::program_options;

static constexpr int DEFAULT_M = 8;
static constexpr int DEFAULT_N = 8;
static constexpr int DEFAULT_K = 8;
static constexpr float DEFAULT_ALPHA = 0.0;
static constexpr float DEFAULT_BETA = 0.0;
// constexpr unsigned long long FLOAT_OPERATIONS = 2ULL * M * N * K;
static constexpr unsigned int DEFAULT_ITERATIONS = 100U;
static constexpr float DEFAULT_ERROR_MARGIN = 0.01;
static constexpr bool DEFAULT_PRINT_MATRICES = false;

int M;
int N;
int K;
float ALPHA;
float BETA;
unsigned int ITERATIONS;
float ERROR_MARGIN;
bool PRINT_MATRICES;

void parse_cmdline_options(int ac, char **av)
{
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "display this help message")
            ("M", po::value<int>(&M)->default_value(DEFAULT_M),
                    "number of rows in matrices A and C")
            ("N", po::value<int>(&N)->default_value(DEFAULT_N),
                    "number of columns in matrices B and C")
            ("K", po::value<int>(&K)->default_value(DEFAULT_K),
                    "number of columns in matrix A and number of rows in matrix B")
            ("alpha", po::value<float>(&ALPHA)->default_value(DEFAULT_ALPHA),
                    "alpha")
            ("beta", po::value<float>(&BETA)->default_value(DEFAULT_BETA),
                    "beta")
            ("iter", po::value<unsigned int>(&ITERATIONS)->default_value(DEFAULT_ITERATIONS),
                    "number of iterations to execute")
            ("err-margin", po::value<float>(&ERROR_MARGIN)->default_value(DEFAULT_ERROR_MARGIN),
                    "floating point error margin for correctness verification")
            ("print-matrices", po::value<bool>(&PRINT_MATRICES)->default_value(DEFAULT_PRINT_MATRICES),
                    "if set to true, input and output matrices will be printed to stdout")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            exit(0);
        }

        std::cout << std::endl << "Configuration:" << std::endl;
        std::cout << "\tM = " << M << ", N = " << N << ", K = " << K << std::endl;
        std::cout << "\tAlpha = " << ALPHA << ", Beta = " << BETA << std::endl;
        std::cout << "\tIterations: " << ITERATIONS << std::endl;
        // std::cout << "\tError Margin: " << ERROR_MARGIN << std::endl;
        // std::cout << "\tDisplay Matrices: " << PRINT_MATRICES << std::endl;
        std::cout << std::endl;
    }
    catch(std::exception& e) {
        std::cout << e.what() << "\n";
    }
}