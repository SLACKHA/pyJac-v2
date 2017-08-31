#ifndef READ_IC_H
#define READ_IC_H

void read_initial_conditions(
    const char* filename, unsigned int NUM, double* phi_host,
    double* param_host, const char order);

#endif