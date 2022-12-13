#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
struct Result {
	int* arr;
	int size;
};
__global__ void find(char* t, char* p, size_t p_size, size_t* pos) {
	int threadid = threadIdx.x;
	bool match = true;
	int count = 0;
	for (int i = 0; i < p_size; i++) {
		count = count+1;
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
Creates a separate gpu thread per character in text,
do not make text larger than 1024 characters.
*/
std::vector<size_t> findSubStr(char *text, char *pattern) {
	const size_t t_size = strlen(text);
	const size_t p_size = strlen(pattern);
	char* t;
	char* p;
	size_t* pos;
	const size_t tc = t_size - p_size+1;
	// allocate memory unified memory for GPU and CPU
	cudaMallocManaged((void**) &t, t_size*sizeof(char));
	cudaMallocManaged((void**)&p, p_size * sizeof(char));
	cudaMallocManaged((void**)&pos, tc*sizeof(size_t)); 
	// copy params into pointers
	strncpy(t, text, strlen(text));
	strncpy(p, pattern, strlen(pattern));
	find <<<1, tc>>> (t, p, p_size, pos); // run kernel

	cudaDeviceSynchronize(); // wait for kernel to finish
	std::vector<size_t> result;
	for (int i = 0; i < tc; i++) {
		if (pos[i] != 1024) {
			result.push_back(pos[i]);
		}
	}
	cudaFree(t);
	cudaFree(p);
	cudaFree(pos);
	return result;
}
int main()
{
	char* text;
	char* pattern;
	text = "asfasfasfasfagjakgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalgjalkegjalekgjalgljeglajgelkejglaegjkalegjalekgjalekgjalekgjaelkgjalkegjlaekgjlaekgdjalekgjaelkgjalkegjalekgjalaekgjlaekgdjalekgjaelkgjalkegjalekgjalkegjaeglkjealgkjaeglkjaeglkjaeghariohqetåc,x.-zsögjlkeöh+3501<dglövc mxnc.testvc.xzm-wåpYURWIÅåwpyoijweÅOIEWJRLKXCVZ,N.N,CZVX.N, å";
	pattern = "test";
	std::vector<size_t> result = findSubStr(text, pattern);
	printf("Matches found at indices: \n");
	for (int i = 0; i < result.size(); i++) {
		printf("%i ", result[i]);
	}
	return 0;
}

