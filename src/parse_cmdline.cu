#include <boost/program_options.hpp>
#include <cutlass/numeric_types.h>
#include <iostream>

#include <parse_cmdline.h>

namespace po = boost::program_options;

static constexpr int DEFAULT_DIM = 128;
static const float DEFAULT_ALPHA = 1.0f;
static const float DEFAULT_BETA = 0.0f;
static constexpr unsigned int DEFAULT_ITERATIONS = 100U;
static constexpr float DEFAULT_KERNEL = 1;
static constexpr bool DEFAULT_PRINT_MATRICES = false;

int DIM;
float ALPHA;
float BETA;
unsigned int ITERATIONS;
int KERNEL;
bool PRINT_MATRICES;

void parse_cmdline_options(int ac, char **av)
{
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "display this help message")
            ("dim", po::value<int>(&DIM)->default_value(DEFAULT_DIM),
                    "dimension for each square matrix")
            ("alpha", po::value<float>(&ALPHA)->default_value(DEFAULT_ALPHA),
                    "alpha")
            ("beta", po::value<float>(&BETA)->default_value(DEFAULT_BETA),
                    "beta")
            ("iterations", po::value<unsigned int>(&ITERATIONS)->default_value(DEFAULT_ITERATIONS),
                    "number of iterations to execute")
            ("kernel", po::value<int>(&KERNEL)->default_value(DEFAULT_KERNEL),
                    "the kernel number to run")
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
        std::cout << "\tSquare Matrix Dimension: " << DIM << std::endl;
        std::cout << "\tAlpha = " << ALPHA << ", Beta = " << BETA << std::endl;
        std::cout << "\tIterations: " << ITERATIONS << std::endl;
        std::cout << "\tKernel Number: " << KERNEL << std::endl;
        std::cout << std::endl;
    }
    catch(std::exception& e) {
        std::cout << e.what() << "\n";
    }
}