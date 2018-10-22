/* A set of simple definitions that allow for assert-style error checking in
 * C-kernel setup
 *
 * Nicholas Curtis - 2017
 */

#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void cpu_assert(bool x, const char* message, const char *file, int line);
#define cassert(ans, message) { cpu_assert((ans), (message), __FILE__, __LINE__); }

#endif