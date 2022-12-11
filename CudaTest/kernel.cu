#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <string>
struct Result {
	int* arr;
	int size;
};
__global__ void find(char* t, char* p, size_t p_size, int* pos) {
	int threadid = threadIdx.x;
	bool match = true;
	for (int i = 0; i < p_size; i++) {
		if (t[threadid + i] != p[i]) {
			match = false;
			break;
		}
	}
	if (match) {
		pos[threadid] = threadid;
	}
	else {
		pos[threadid] = 1024;
	}
}
/*
Creates a separate gpu thread per character in text, dont go over 1024
*/
Result findSubStr(char *text, char *pattern) {
	const size_t t_size = strlen(text);
	const size_t p_size = strlen(pattern);
	char* t;
	char* p;
	int* pos;
	const size_t tc = t_size - p_size + 1;
	cudaMallocManaged((void**) &t, t_size*sizeof(char));
	cudaMallocManaged((void**)&p, p_size * sizeof(char));
	cudaMallocManaged((void**)&pos, tc*sizeof(int)); 
	strncpy(t, text, strlen(text));
	strncpy(p, pattern, strlen(pattern));
	find <<<1, tc>>> (t, p, p_size, pos);
	cudaDeviceSynchronize();
	int count = 0;
	int* result;
	for (int i = 0; i < tc; i++) {
		if (pos[i] != 1024) {
			if (count == 0) {
				result = (int*)malloc(count * sizeof(int));
			}
			result[count] = pos[i];
			count = count++;
		}
	}
	cudaFree(t);
	cudaFree(p);
	cudaFree(pos);
	Result res = { result, count };
	return res;
}
int main()
{
	char* text;
	char* pattern;
	text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaworldaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaworld";
	pattern = "world";
	Result res = findSubStr(text, pattern); 
	printf("result set size:%i\n",res.size);
	for (int i = 0; i < res.size; i++) {
		printf("found at position:%i\n",res.arr[i]);
	}
	return 0;
}

