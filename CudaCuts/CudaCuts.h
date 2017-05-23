/********************************************************************************************
* Implementing Graph Cuts on CUDA using algorithm given in CVGPU '08                       ** 
* paper "CUDA Cuts: Fast Graph Cuts on GPUs"                                               **  
*                                                                                          **   
* Copyright (c) 2008 International Institute of Information Technology.                    **  
* All rights reserved.                                                                     **  
*                                                                                          ** 
* Permission to use, copy, modify and distribute this software and its documentation for   ** 
* educational purpose is hereby granted without fee, provided that the above copyright     ** 
* notice and this permission notice appear in all copies of this software and that you do  **
* not sell the software.                                                                   **  
*                                                                                          **
* THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR    **
* OTHERWISE.                                                                               **  
*                                                                                          **
* Created By Vibhav Vineet.                                                                ** 
********************************************************************************************/

#ifndef _CUDACUTS_H_
#define _CUDACUTS_H_

/*Header files included*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "cuda.h"

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


/* ***************************************************
 * Kernels which perform the push, pull and         **
 * relabel operations. It has the kernel            **
 * which performs the bfs operation. n-edgeweights  **
 * and t-edgeweights are also calculated here       ** 
 * **************************************************/

#define datacost(pix,lab)     (datacost[(pix)*num_Labels+(lab)] )
#define smoothnesscost(lab1,lab2) (smoothnesscost[(lab1)+(lab2)*num_Labels] )

/*****************************************************************
 * CONTROL_M -- this decides after how many iterations          **
 * m should be changed from 1 to 2. Here m equal to 1           **
 * means one push-pull operation followed by one relabel        **
 * operation and m equal to 2 means two push-pull operations    ** 
 * followed by one relabel operation.                           **
 * **************************************************************/

#define CONTROL_M 40


class CudaCuts {
public:
	/********************************************************************
	* cudaCutsInit(width, height, numOfLabels) function sets the      **
	* width, height and numOfLabels of grid. It also initializes the  **
	* block size  on the device and finds the total number of blocks  **
	* running in parallel on the device. It calls checkDevice         **
	* function which checks whether CUDA compatible device is present **
	* on the system or not. It allocates the memory on the host and   **
	* the device for the arrays which are required through the        **
	* function call h_mem_init and segment_init respectively. This    **
	* function returns 0 on success or -1 on failure if there is no   **
	* * * CUDA compatible device is present on the system             **
	* *****************************************************************/

	int cudaCutsInit(int, int, int);

	/**************************************************
	* function checks whether any CUDA compatible   **
	* device is present on the system or not. It    **
	* returns the number of devices present on the  **
	* system.                                       **
	* ***********************************************/

	int checkDevice();

	/**************************************************
	* h_mem_init returns allocates and intializes   **
	* memory on the host                            **
	* ***********************************************/

	void h_mem_init();

	/***************************************************************
	* This function allocates memory for n-edges, t-edges,       **
	* height and mask function, pixelLabels and intializes them  **
	* on the device.                                             **
	* ************************************************************/

	void d_mem_init();

	/********************************************************
	* This function copies the dataTerm from the host to  **
	* device and also copies the data into datacost array **
	* of size width * height * numOfLabels                **
	* *****************************************************/

	int cudaCutsSetupDataTerm();

	/*************************************************************
	* This function copies the smoothnessTerm from the host to  **
	* device and also copies the data into smoothnesscost array **
	* of size numOfLabels * numOfLabels                         **
	* ***********************************************************/

	int cudaCutsSetupSmoothTerm();

	/*************************************************************
	* As in our case, when the graph is grid, horizotal and    **
	* vertical cues can be specified. The hcue and vcue array  **
	* of size width * height stores these respectively.        **
	* ***********************************************************/

	int cudaCutsSetupHCue();
	int cudaCutsSetupVCue();

	/*********************************************************
	* This function constructs the graph on the device.    **
	* ******************************************************/

	int cudaCutsSetupGraph();

	/************************************************************
	* The function calls the Cuda Cuts optimization algorithm **
	* and the bfs algorithm so as to assign a label to each   **
	* pixel                                                   **
	* *********************************************************/

	int cudaCutsAtomicOptimize();
	int cudaCutsStochasticOptimize();

	/***********************************************************
	* This function calls three kernels which performs the   **
	* push, pull and relabel operation                       **
	* ********************************************************/

	void cudaCutsStochastic();
	void cudaCutsAtomic();

	/**********************************************************
	* This finds which of the nodes are in source set and   **
	* sink set                                              **
	* *******************************************************/

	void bfsLabeling();

	/****************************************************************
	* This function assigns a label to each pixel and stores them **
	* in pixelLabel array of size width * height                  **
	* *************************************************************/

	int cudaCutsGetResult();

	/************************************************************
	* De-allocates all the memory allocated on the host and   **
	* the device.                                             **
	* *********************************************************/

	void cudaCutsFreeMem();

	////////////////////////////////////////////////////////////
	//Global Variables declared                               //
	////////////////////////////////////////////////////////////

	/*************************************************
	* n-edges and t-edges                          **
	* **********************************************/

	int *d_left_weight, *d_right_weight, *d_down_weight, *d_up_weight, *d_push_reser, *d_sink_weight;
	int *s_left_weight, *s_right_weight, *s_down_weight, *s_up_weight, *s_push_reser, *s_sink_weight;
	int *d_pull_left, *d_pull_right, *d_pull_down, *d_pull_up;

	int *d_stochastic, *d_stochastic_pixel, *d_terminate;

	/*************************************************
	* Emergu parameters stored                     **
	* **********************************************/

	int *dataTerm, *smoothTerm, *hCue, *vCue;
	int *dDataTerm, *dSmoothTerm, *dHcue, *dVcue, *dPixelLabel;


	/*************************************************
	* Height and mask functions are stored         **
	* **********************************************/

	int  *d_relabel_mask, *d_graph_heightr, *d_graph_heightw;

	/*************************************************
	* Grid and Block parameters                    **
	* **********************************************/

	int graph_size, size_int, width, height, graph_size1, width1, height1, depth, num_Labels;
	int blocks_x, blocks_y, threads_x, threads_y, num_of_blocks, num_of_threads_per_block;

	/***************************************************
	* Label of each pixel is stored in this function **
	* *************************************************/

	int *pixelLabel;

	bool *d_pixel_mask, h_over, *d_over, *h_pixel_mask;
	int *d_counter, *h_graph_height;
	int *h_reset_mem;
	int cueValues, deviceCheck, deviceCount;

	int *h_stochastic, *h_stochastic_pixel, *h_relabel_mask;
	int counter;

};

/* **********************************************************
* CUDA Cuts kernel functions                              **
* *********************************************************/

__global__ void
kernel_push1_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down,
int *g_pull_up, int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1);

__global__ void
kernel_relabel_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down,
int *g_pull_up, int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1);

__global__ void
kernel_relabel_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, /*int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up, */
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_stochastic, int *g_block_num);

__global__ void
kernel_push2_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1);

__global__ void
kernel_End(int *g_stochastic, int *g_count_blocks, int *g_counter);


__global__ void
kernel_push1_start_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish);

__global__ void
kernel_push1_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, /*int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up,*/
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_stochastic, int *g_block_num);

__global__ void
kernel_push2_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish, int *g_block_num);

__global__ void
kernel_bfs_t(int *g_push_reser, int  *g_sink_weight, int *g_graph_height, bool *g_pixel_mask,
int vertex_num, int width, int height, int vertex_num1, int width1, int height1);

__global__ void
kernel_push_stochastic1(int *g_push_reser, int *s_push_reser, int *g_count_blocks, bool *g_finish, int *g_block_num, int width1);

__global__ void
kernel_push_atomic2(int *g_terminate, int *g_push_reser, int *s_push_reser, int *g_block_num, int width1);


__global__ void
kernel_push_stochastic2(int *g_terminate, int *g_relabel_mask, int *g_push_reser, int *s_push_reser, int *d_stochastic, int *g_block_num, int width1);

__global__ void
kernel_push1_start_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish);


__global__ void
kernel_bfs(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_graph_height, bool *g_pixel_mask, int vertex_num, int width, int height,
int vertex_num1, int width1, int height1, bool *g_over, int *g_counter);

__device__
void add_edge(int from, int to, int cap, int rev_cap, int type, int *d_left_weight,
int *d_right_weight, int *d_down_weight, int *d_up_weight);

__device__
void add_tweights(int i, int cap_source, int  cap_sink, int *d_push_reser, int *d_sink_weight);

__device__
void add_term1(int i, int A, int B, int *d_push_reser, int *d_sink_weight);

__device__
void add_t_links_Cue(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser, int *d_sink_weight,
int *dPixelLabel, int *dDataTerm, int width, int height, int num_labels);

__device__
void add_t_links(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser, int *d_sink_weight,
int *dPixelLabel, int *dDataTerm, int width, int height, int num_labels);

__device__
void add_term2(int x, int y, int A, int B, int C, int D, int type, int *d_left_weight,
int *d_right_weight, int *d_down_weight, int *d_up_weight, int *d_push_reser, int *d_sink_weight);

__device__
void set_up_expansion_energy_G_ARRAY(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser,
int *d_sink_weight, int *dPixelLabel, int *dDataTerm, int *dSmoothTerm,
int width, int height, int num_labels);

__device__
void set_up_expansion_energy_G_ARRAY_Cue(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser,
int *d_sink_weight, int *dPixelLabel, int *dDataTerm, int *dSmoothTerm,
int *dHcue, int *dVcue, int width, int height, int num_labels);



__global__
void CudaWeightCue(int alpha_label, int *d_left_weight, int *d_right_weight, int *d_down_weight,
int *d_up_weight, int *d_push_reser, int *d_sink_weight, int *dPixelLabel,
int *dDataTerm, int *dSmoothTerm, int *dHcue, int *dVcue, int width, int height, int num_labels);


__global__
void CudaWeight(int alpha_label, int *d_left_weight, int *d_right_weight, int *d_down_weight,
int *d_up_weight, int *d_push_reser, int *d_sink_weight, int *dPixelLabel,
int *dDataTerm, int *dSmoothTerm, int width, int height, int num_labels);

__global__
void adjustedgeweight(int *d_left_weight, int *d_right_weight, int *d_down_weight, int *d_up_weight,
int *d_push_reser, int *d_sink_weight, int *temp_left_weight, int *temp_right_weight,
int *temp_down_weight, int *temp_up_weight, int *temp_push_reser, int *temp_sink_weight,
int width, int height, int graph_size, int width1, int height1, int graph_size1);

__global__
void copyedgeweight(int *d_left_weight, int *d_right_weight, int *d_down_weight, int *d_up_weight,
int *d_push_reser, int *d_sink_weight, int *temp_left_weight, int *temp_right_weight,
int *temp_down_weight, int *temp_up_weight, int *temp_push_reser, int *temp_sink_weight,
int *d_pull_left, int *d_pull_right, int *d_pull_down, int *d_pull_up, int *d_relabel_mask,
int *d_graph_heightr, int *d_graph_heightw, int width, int height, int graph_size, int width1, int height1, int graph_size1);

#endif
