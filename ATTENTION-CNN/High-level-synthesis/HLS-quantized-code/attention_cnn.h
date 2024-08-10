// cnn.h

#ifndef CNN_H
#define CNN_H
#include <ap_fixed.h>
#include <cmath>

#include <cmath>
#include <random>
#include <ap_fixed.h>
#include <hls_math.h>
#include "weights_attentionblock.h"
#include "weights_conv.h"
#include "weights_dense.h"

ap_fixed<8,4> relu(ap_fixed<8,4> x);
void relu2(ap_fixed<8,4> input[],ap_fixed<8,4> size);
void sigmoid(ap_fixed<8,4> input[], ap_fixed<4,2> size);
void softmax(const ap_fixed<8,4> input[], ap_fixed<4,2> size, ap_fixed<8,4> output[]);
void addPadding(const ap_fixed<8,4> image[],ap_fixed<8,4> output[],ap_fixed<4,2> width, ap_fixed<4,2> height, ap_fixed<4,2> depth, ap_fixed<4,2> padding);
void convolution(const ap_fixed<8,4> flattenedImageinput[], const ap_fixed<8,4> kernels[], ap_fixed<8,4> pad_output[],ap_fixed<8,4> conv_output[], ap_fixed<4,2> imageWidth, ap_fixed<4,2> imageHeight, ap_fixed<4,2> imageDepth, ap_fixed<4,2> k_h, ap_fixed<4,2> k_w, const ap_fixed<8,4> biases[], ap_fixed<4,2> numKernels , char a_f,char p, ap_fixed<4,2> stride);
void globalAveragePooling(const ap_fixed<8,4> image[], ap_fixed<8,4> output[], ap_fixed<4,2> imageWidth, ap_fixed<4,2> numChannels);
void expand_dimensions(const ap_fixed<8,4> input[],ap_fixed<8,4> output[],ap_fixed<4,2> width,ap_fixed<4,2> height,ap_fixed<4,2> channels, ap_fixed<8,4> y[]);
void fullyConnectedLayer(const ap_fixed<8,4> input[], ap_fixed<8,4> output[], const ap_fixed<8,4> weights[], const ap_fixed<8,4> bias[], ap_fixed<4,2> inputSize, ap_fixed<4,2> outputSize, char a_f) ;
void attentionblock(ap_fixed<8,4> flattenedImageinput[],const ap_fixed<8,4> conv_kernels[],ap_fixed<8,4> conv_output[],ap_fixed<4,2> imageWidth, ap_fixed<4,2> imageDepth,const ap_fixed<8,2> conv_biases[],ap_fixed<4,2> size_conv_output, ap_fixed<8,4> ga_output[],ap_fixed<8,4> fc1_output[],const ap_fixed<8,4> fc1_weights[],const ap_fixed<8,4> fc1_bias[],ap_fixed<4,2> fc1_inputSize,ap_fixed<8,4> fc2_output[],const ap_fixed<8,4> fc2_weights[],const ap_fixed<8,4> fc2_bias[],ap_fixed<8,4> expand_output[]);
void CNN(const ap_fixed<8,4> flattenedImage[], ap_fixed<4,2> imageWidth, ap_fixed<4,2> imageHeight, ap_fixed<8,4> output[]);

#endif // CNN_H
