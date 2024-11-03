#include<math.h>
#pragma once

double c_func(int n)
{
	int i;
	double result = 0.0;
	for(i=1; i<n; i++)
		result = result + sqrt(i);
	return result;
}
