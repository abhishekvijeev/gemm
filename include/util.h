#include <cublas_v2.h>

static const char *cublasGetErrorString(cublasStatus_t error);

/**
 * Panic wrapper for unwinding CUBLAS runtime errors
 */
#define CUBLAS_CHECK(status)                                                    \
  {                                                                             \
    cublasStatus_t error = status;                                              \
    if (error != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "Got bad cublas status: " << cublasGetErrorString(error)     \
                << " at line: " << __LINE__ << std::endl;                       \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

static const char *cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

                                                                            