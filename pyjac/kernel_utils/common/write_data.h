/*
	write_data.h - a shared convenience function to write output of species /
	Jacobian kernel calls to file for validation

	\author Nick Curtis
	\date Oct 2017
*/
#ifndef WRITE_DATA_H
#define WRITE_DATA_H

#include <string.h>
#include <stdio.h>
#include <assert.h>

static void write_data(const char* filename, double* arr, size_t var_size)
{
	FILE* fp = fopen(filename, "wb");
	if (fp == NULL)
	{
		const char* err = "Error opening file for data output: ";
		size_t buffsize = strlen(filename) + strlen(err) * sizeof(char);
		char* buff = (char*)malloc(buffsize);
		snprintf(buff, buffsize, "%s%s", err, filename);
		printf("%s\n", buff);
		free(buff);
		exit(-1);
	}
    assert(fwrite(arr, sizeof(double), var_size, fp) == var_size
        && "Wrong filesize written.");
    assert(!fclose(fp) && "Error writing to file");
}

#endif
