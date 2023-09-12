#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/host_tensor.h>

#include <cstdio>
#include <iostream>

#include <sgemm.h>

int main(int argc, char const **argv)
{
    cutlass::CommandLine cmd(argc, argv);
    bool help;
    int M, N, K;
    float alpha, beta;

    if (cmd.check_cmd_line_flag("help")) {
        std::cout << "help set\n";
        help = true;
        return;
    }

    cmd.get_cmd_line_argument("M", M, 8);
    cmd.get_cmd_line_argument("N", N, 8);
    cmd.get_cmd_line_argument("K", K, 8);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 1.0f);

    printf("M = %d, N = %d, K = %d\n", M, N, K);
    printf("Alpha = %.2f\n", alpha);
    printf("Beta = %.2f\n", beta);

    sgemm(M, N, K, alpha, beta);
}