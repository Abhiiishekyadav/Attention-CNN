// cnn.h

#ifndef CNN_H
#define CNN_H

#include <cmath>
#include <random>
#include <ap_fixed.h>
#include <hls_math.h>
#include "weights_attentionblock.h"
#include "weights_conv.h"
#include "weights_dense.h"

double relu(double x);
void relu2(double input[],int size);
void sigmoid(double input[], int size);
void softmax(const double input[], int size, double output[]);
void addPadding(const double image[],double output[],int width, int height, int depth, int padding);
void convolution(const double flattenedImageinput[], const double kernels[], double pad_output[],double conv_output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const double biases[], int numKernels , char a_f,char p, int stride);
void globalAveragePooling(const double image[], double output[], int imageWidth, int numChannels);
void expand_dimensions(const double input[],double output[],int width,int height,int channels, double y[]);
void fullyConnectedLayer(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize, char a_f) ;
void attentionblock(double flattenedImageinput[],const double conv_kernels[],double conv_output[],int imageWidth, int imageDepth,const double conv_biases[],int size_conv_output, double ga_output[],double fc1_output[],const double fc1_weights[],const double fc1_bias[],int fc1_inputSize,double fc2_output[],const double fc2_weights[],const double fc2_bias[],double expand_output[]);
void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]);

#endif // CNN_H
