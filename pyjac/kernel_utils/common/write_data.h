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

void write_data(char* filename, double* arr, size_t var_size)
{
	fp = fopen(filename, "wb");
	if (fp != NULL)
	{
		char* err = "Error opening file for data output: ";
		size_t buffsize = strlen(filename) + strlen(err) * sizeof(char);
		char* buff = (char*)malloc(buffsize);
		snprintf(buf, buffsize, "%s%s", err, filename);
		cassert(0, buff);
		free(buff);
	}
    cassert(fwrite(arr, sizeof(double), var_size, fp) == sizeof(double) * var_size,
        "Wrong filesize written.");
    assert(!fclose(fp), "Error writing to file");
}

#endif
