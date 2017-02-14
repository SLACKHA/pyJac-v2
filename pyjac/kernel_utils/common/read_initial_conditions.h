#ifndef READ_IC_H
#define READ_IC_H

void read_initial_conditions(const char* filename, unsigned int NUM, double* T_host, double* P_host,
    double* conc_host, const char order);

#endif