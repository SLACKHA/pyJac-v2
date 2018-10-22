/* A set of simple definitions that allow for assert-style error checking in
 * C-kernel setup
 *
 * Nicholas Curtis - 2017
 */

#include "error_check.hpp"

void cpu_assert(bool x, const char* message, const char *file, int line) {
    if (!x)
    {
        fprintf(stderr, "cpu_assert: %s %s %d\n", message, file, line);
        exit(-1);
    }
}
