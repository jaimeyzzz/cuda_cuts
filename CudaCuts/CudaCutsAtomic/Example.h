
/***********************************************************************************************
 * * Implementing Graph Cuts on CUDA using algorithm given in CVGPU '08                       **
 * * paper "CUDA Cuts: Fast Graph Cuts on GPUs"                                               **
 * *                                                                                          **
 * * Copyright (c) 2008 International Institute of Information Technology.                    **
 * * All rights reserved.                                                                     **
 * *                                                                                          **
 * * Permission to use, copy, modify and distribute this software and its documentation for   **
 * * educational purpose is hereby granted without fee, provided that the above copyright     **
 * * notice and this permission notice appear in all copies of this software and that you do  **
 * * not sell the software.                                                                   **
 * *                                                                                          **
 * * THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR    **
 * * OTHERWISE.                                                                               **
 * *                                                                                          **
 * * Created By Vibhav Vineet.                                                                **
 * ********************************************************************************************/

#ifndef _IMAGE_WRITE_H_
#define _IMAGE_WRITE_H_

#include "CudaCuts.h"

void write_image();
void load_files( char *filename);
void writeImage();
void initFinalImage() ; 
void LoadDataFile( char *filename, int &width, int &height, int &nLabels, int *&dataCostArray, int *&smoothCostArray, int *&hCue, int *&vCue);


int **pixel_values, **out_pixel_values;


#endif
