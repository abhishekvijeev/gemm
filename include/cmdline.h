#include "cutlass/util/command_line.h"

class CommandLineArgs
{
    public:
        CommandLineArgs():
            help(false),
            dim(128),
            kernel(1),
            alpha(1.0),
            beta(0.0),
            iterations(100)
        { }

        void parse(int argc, char const** argv)
        {
            cutlass::CommandLine cmdline(argc, argv);
            if (cmdline.check_cmd_line_flag("help")) {
                help = true;
                return;
            }

            cmdline.get_cmd_line_argument("dim", dim);
            cmdline.get_cmd_line_argument("kernel", kernel);
            cmdline.get_cmd_line_argument("alpha", alpha, cutlass::half_t(1.0));
            cmdline.get_cmd_line_argument("beta", beta, cutlass::half_t(0.0));
            cmdline.get_cmd_line_argument("iterations", iterations);
        }

        std::ostream& print_usage(std::ostream& out) const
        {
            out << "\nCUTLASS GEMM\n\n"
                << "Options:\n\n"
                << "  --help                      If specified, displays this usage statement\n\n"
                << "  --dim=<int>                 Sets the GEMM square matrix dimension\n"
                << "  --kernel=<int>              GEMM kernel number\n"
                << "  --alpha=<f32>               Epilogue scalar alpha\n"
                << "  --beta=<f32>                Epilogue scalar beta\n\n"
                << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

            out << "\n\nExamples:\n\n"
                << "$ " << "main" << " --dim=128 --kernel=1 --alpha=2 --beta=0.707 \n\n";

            return out;
        }

        bool help;
        int dim;
        int kernel;
        cutlass::half_t alpha;
        cutlass::half_t beta;
        int iterations;
};