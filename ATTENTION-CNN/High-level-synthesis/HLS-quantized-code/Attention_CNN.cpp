#include "attention_cnn.h"
#pragma inline

#include <ap_fixed.h>
#include <math.h>
typedef ap_fixed<8, 4> doub;
typedef ap_fixed<4, 2> intt;

// ReLU function
doub relu(doub x) {
    // Use an explicit cast to ap_fixed
    return (x < doub(0)) ? doub(0) : x;
}

void relu2(doub input[],intt size){
    for(intt i=0;i<size;i++){
        input[i]=relu(input[i]);
    }
}


// Convert fixed-point to double
double to_double(doub x) {
    return static_cast<double>(x);
}

// Convert double to fixed-point
doub to_fixed_point(double x) {
    return static_cast<doub>(x);
}
void sigmoid(doub input[], intt size) {
    for (intt i = 0; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
        // Convert input to double for exp calculation
        double input_double = to_double(input[i]);
        // Compute the sigmoid function
        double exp_val = std::exp(-input_double);
        double result = 1.0 / (1.0 + exp_val);
        // Convert result back to fixed-point
        input[i] = to_fixed_point(result);
    }
}






// Softmax function
void softmax(const doub input[], intt size, doub output[]) {
    // Compute the maximum value for numerical stability
    doub max_val = input[0];
    for (intt i = 1; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute the sum of exponentials
    double sum_exp = 0.0;
    for (intt i = 0; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
        double exp_val = std::exp(to_double(input[i]) - to_double(max_val));
        sum_exp += exp_val;
    }

    // Convert sum_exp to fixed-point
    doub sum_exp_fixed = to_fixed_point(sum_exp);

    // Compute the softmax output
    for (intt i = 0; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
        double exp_val = std::exp(to_double(input[i]) - to_double(max_val));
        output[i] = to_fixed_point(exp_val / to_double(sum_exp_fixed));
    }
}

// Function to add padding to each channel of a 3D image
void addPadding(const doub image[],doub output[],intt width, intt height, intt depth, intt padding) {

    intt new_width = width + 2 * padding;
    intt new_height = height + 2 * padding;

    intt len = new_width * new_height * depth;
    for(intt i = 0; i < len; i++){
        output[i]=0;
    }

    for (intt z = 0; z < depth; ++z) {
        for (intt y = 0; y < height; ++y) {
            for (intt x = 0; x < width; ++x) {
                intt original_index = z * width * height + y * width + x;
                intt padded_index = (z * new_height + y + padding) * new_width + (x + padding);
                output[padded_index] = image[original_index];
            }
        }
    }
}

// Convulation operation
void convolution(const doub flattenedImageinput[], const doub kernels[], doub pad_output[],doub conv_output[], intt imageWidth, intt imageHeight, intt imageDepth, intt k_h, intt k_w, const doub biases[], intt numKernels , char a_f,char p, intt stride)
{
    intt padding=0;
    if(p=='p'){
        padding = (k_h - 1) / 2;
    }
    intt new_width = imageWidth + 2 * padding;
    intt new_height = imageHeight + 2 * padding;
    if(p=='p'){
        addPadding(flattenedImageinput,pad_output,imageWidth,imageHeight,imageDepth,padding);
    }
    intt output_h = (new_height - k_h) / stride + 1;
    intt output_w = (new_width - k_w) / stride + 1;

    #pragma HLS UNROLL factor=2

    for (intt k = 0; k < numKernels; ++k)
    {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        #pragma HLS UNROLL factor=2

        for (intt i = 0; i < output_h; ++i)
        {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
             #pragma HLS UNROLL factor=2

            for (intt j = 0; j < output_w; ++j)
            {
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
                 #pragma HLS PIPELINE II=2

                conv_output[k * output_h * output_w + i * output_w + j] = biases[k];
                for (intt kd = 0; kd < imageDepth; ++kd)
                {
                    #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                    for (intt ki = 0; ki < k_h; ++ki)
                    {
                        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                        for (intt kj = 0; kj < k_w; ++kj)
                        {
                            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                            intt input_i = i * stride + ki;
                            intt input_j = j * stride + kj;
                            conv_output[k * output_h * output_w + i * output_w + j] +=
                                pad_output[input_j + input_i * imageWidth + kd * imageHeight * imageWidth] * kernels[k * (k_h * k_w * imageDepth) + ki * k_w + kd * k_h * k_w + kj];
                        }
                    }
                }
                if(a_f=='R'){
                    conv_output[k * output_h * output_w + i * output_w + j] = relu(conv_output[k * output_h * output_w + i * output_w + j]);
                }
            }
        }
    }
}

// global average pooling layer
void globalAveragePooling(const doub image[], doub output[], intt imageWidth, intt numChannels) {
    intt i_h = imageWidth;
    intt i_w = i_h;
    intt total_elements = i_h * i_w;

    for (intt c = 0; c < numChannels; ++c) {
        doub sum_val = 0.0;
        for (intt i = 0; i < i_h; ++i) {
            for (intt j = 0; j < i_w; ++j) {
                sum_val += image[c * total_elements + i * i_w + j];
            }
        }
        doub avg_val = sum_val / total_elements;
        output[c] = avg_val;
    }
}

// function to expand dimension of output to the dimension of input .  (inputs * tf.expand_dims(tf.expand_dims(y, 1), 1))
void expand_dimensions(const doub input[],doub output[],intt width,intt height,intt channels, doub y[]){
    intt count=0;
    for(intt i=0;i<channels;i++){
        doub mul_val = y[i];
        for (intt k = 0; k < height; ++k) {
            for (intt j = 0; j < width; ++j) {
                output[count] = input[count]*mul_val;
                count++;
            }
        }
    }
}

// fully connected dense layer
void fullyConnectedLayer(const doub input[], doub output[], const doub weights[], const doub bias[], intt inputSize, intt outputSize, char a_f) {
    for (intt i = 0; i < outputSize; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = bias[i];
        for (intt j = 0; j < inputSize; ++j) {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
            output[i] += input[j] * weights[i * inputSize + j];
        }

        if (a_f=='R') {
        	   output[i] = (output[i] < doub(0)) ? doub(0) : output[i];
        }
    }
}

// generalized attention block function
void attentionblock(doub flattenedImageinput[],const doub conv_kernels[],doub conv_output[],intt imageWidth, intt imageDepth,const doub conv_biases[],intt size_conv_output, doub ga_output[],doub fc1_output[],const doub fc1_weights[],const doub fc1_bias[],intt fc1_inputSize,doub fc2_output[],const doub fc2_weights[],const doub fc2_bias[],doub expand_output[]){
    intt numKernels =  imageDepth/8;
    intt fc1_outputSize = imageDepth;
    convolution(flattenedImageinput,conv_kernels,flattenedImageinput,conv_output,imageWidth,imageWidth,imageDepth,1,1, conv_biases,numKernels ,'S','e',1);
    // y = self.conv(inputs)
    relu2(conv_output,size_conv_output);
    //     y = self.relu(y)
    globalAveragePooling(conv_output, ga_output,imageWidth,numKernels);
    //     y = self.pool(y)
    fullyConnectedLayer(ga_output, fc1_output, fc1_weights,fc1_bias,fc1_inputSize,fc1_outputSize,'S') ;
    //     y = self.fc1(y)
    relu2(fc1_output,fc1_outputSize);
    //     y = self.relu(y)
    fullyConnectedLayer(fc1_output,fc2_output,fc2_weights,fc2_bias,fc1_outputSize,fc1_outputSize,'S' ) ;
    //     y = self.fc2(y)
    sigmoid(fc2_output,fc1_outputSize);
    //     y = self.sigmoid(y)
    expand_dimensions(flattenedImageinput,expand_output,imageWidth,imageWidth,imageDepth,fc2_output);
    //     return inputs * tf.expand_dims(tf.expand_dims(y, 1), 1)
}

//void CNN(const doub flattenedImage[], intt imageWidth, intt imageHeight, doub output[]) {
//    #pragma HLS inttERFACE m_axi port=flattenedImage bundle=gmem0 offset=slave depth=7840000
//    #pragma HLS inttERFACE m_axi port=output bundle=gmem1 offset=slave depth=10000
//    #pragma HLS inttERFACE s_axilite port=imageWidth bundle=control
//    #pragma HLS inttERFACE s_axilite port=imageHeight bundle=control
//    #pragma HLS inttERFACE s_axilite port=flattenedImage bundle=control
//    #pragma HLS inttERFACE s_axilite port=output bundle=control
//    #pragma HLS inttERFACE s_axilite port=return bundle=control
//
//    // Convolutional layer 1
//    static doub conv1Output[112*112*64];
//    static doub conv1_pad_output[230*230*3];
//    #pragma HLS ARRAY_PARTITION variable=conv1Output cyclic factor=64
//    #pragma HLS ARRAY_PARTITION variable=conv1_pad_output cyclic factor=3
//    convolution(flattenedImage, convolution1_weights, conv1_pad_output, conv1Output, 224, 224, 3, 7, 7, convolution1_bias, 64, 'S', 'p', 2);
//
//    // Attention 1
//    static doub attention1_output[112*112*64];
//    static doub at1_conv_output[112*112*8];
//    static doub at1_ga_output[8];
//    static doub at1_fc1_output[64];
//    static doub at1_fc2_output[64];
//    #pragma HLS ARRAY_PARTITION variable=attention1_output cyclic factor=64
//    #pragma HLS ARRAY_PARTITION variable=at1_conv_output cyclic factor=8
//    attentionblock(conv1Output, at1_convolution_weights, at1_conv_output, 112, 64, at1_convolution_biases, 112*112*8, at1_ga_output, at1_fc1_output, at1_fc1_weights, at1_fc1_biases, 8, at1_fc2_output, at1_fc2_weights, at1_fc2_biases, attention1_output);
//
//    // Convolutional layer 2
//    static doub conv2Output[56*56*128];
//    static doub conv2_pad_output[114*114*64];
//    #pragma HLS ARRAY_PARTITION variable=conv2Output cyclic factor=128
//    #pragma HLS ARRAY_PARTITION variable=conv2_pad_output cyclic factor=64
//    convolution(attention1_output, convolution2_weights, conv2_pad_output, conv2Output, 112, 112, 64, 3, 3, convolution2_bias, 128, 'S', 'p', 2);
//
//    // Attention 2
//    static doub attention2_output[56*56*128];
//    static doub at2_conv_output[56*56*16];
//    static doub at2_ga_output[16];
//    static doub at2_fc1_output[128];
//    static doub at2_fc2_output[128];
//    #pragma HLS ARRAY_PARTITION variable=attention2_output cyclic factor=128
//    #pragma HLS ARRAY_PARTITION variable=at2_conv_output cyclic factor=16
//    attentionblock(conv2Output, at2_convolution_weights, at2_conv_output, 56, 128, at2_convolution_biases, 56*56*16, at2_ga_output, at2_fc1_output, at2_fc1_weights, at2_fc1_biases, 16, at2_fc2_output, at2_fc2_weights, at2_fc2_biases, attention2_output);
//
//    // Convolutional layer 3
//    static doub conv3Output[28*28*256];
//    static doub conv3_pad_output[58*58*128];
//    #pragma HLS ARRAY_PARTITION variable=conv3Output cyclic factor=256
//    #pragma HLS ARRAY_PARTITION variable=conv3_pad_output cyclic factor=128
//    convolution(attention2_output, convolution3_weights, conv3_pad_output, conv3Output, 56, 56, 128, 3, 3, convolution3_bias, 256, 'S', 'p', 2);
//
//    // Attention 3
//    static doub attention3_output[28*28*256];
//    static doub at3_conv_output[28*28*32];
//    static doub at3_ga_output[32];
//    static doub at3_fc1_output[256];
//    static doub at3_fc2_output[256];
//    #pragma HLS ARRAY_PARTITION variable=attention3_output cyclic factor=256
//    #pragma HLS ARRAY_PARTITION variable=at3_conv_output cyclic factor=32
//    attentionblock(conv3Output, at3_convolution_weights, at3_conv_output, 28, 256, at3_convolution_biases, 28*28*32, at3_ga_output, at3_fc1_output, at3_fc1_weights, at3_fc1_biases, 32, at3_fc2_output, at3_fc2_weights, at3_fc2_biases, attention3_output);
//
//    // Convolutional layer 4
//    static doub conv4Output[14*14*512];
//    static doub conv4_pad_output[30*30*256];
//    #pragma HLS ARRAY_PARTITION variable=conv4Output cyclic factor=512
//    #pragma HLS ARRAY_PARTITION variable=conv4_pad_output cyclic factor=256
//    convolution(attention3_output, convolution4_weights, conv4_pad_output, conv4Output, 28, 28, 256, 3, 3, convolution4_bias, 512, 'S', 'p', 2);
//
//    // Attention 4
//    static doub attention4_output[14*14*512];
//    static doub at4_conv_output[14*14*64];
//    static doub at4_ga_output[64];
//    static doub at4_fc1_output[512];
//    static doub at4_fc2_output[512];
//    #pragma HLS ARRAY_PARTITION variable=attention4_output cyclic factor=512
//    #pragma HLS ARRAY_PARTITION variable=at4_conv_output cyclic factor=64
//    attentionblock(conv4Output, at4_convolution_weights, at4_conv_output, 14, 512, at4_convolution_biases, 14*14*64, at4_ga_output, at4_fc1_output, at4_fc1_weights, at4_fc1_biases, 64, at4_fc2_output, at4_fc2_weights, at4_fc2_biases, attention4_output);
//
//    // Convolutional layer 5
//    static doub conv5Output[7*7*512];
//    static doub conv5_pad_output[16*16*512];
//
//    #pragma HLS ARRAY_PARTITION variable=conv5Output cyclic factor=512
//    #pragma HLS ARRAY_PARTITION variable=conv5_pad_output cyclic factor=512
//    convolution(attention4_output, convolution5_weights, conv5_pad_output, conv5Output, 14, 14, 512, 3, 3, convolution5_bias, 512, 'S', 'p', 2);
//
//    // Global average pooling layer
//    static doub gb_layeroutput[512];
//    globalAveragePooling(conv5Output, gb_layeroutput, 7, 512);
//
//    // Dense layer 1
//    static doub fully1_output[2048];
//    fullyConnectedLayer(gb_layeroutput, fully1_output, dense1_weights, dense1_biases, 512, 2048, 'R');
//
//    // Dense layer 2
//    static doub fully2_output[2];
//    fullyConnectedLayer(fully1_output, fully2_output, dense2_weights, dense2_biases, 2048, 2, 'S');
//}
//


void CNN(const doub flattenedImage[], intt imageWidth, intt imageHeight, doub output[]) {

#pragma HLS INTERFACE m_axi port=flattenedImage bundle=gmem0 offset=slave depth=7840000
 #pragma HLS INTERFACE m_axi port=output bundle=gmem1 offset=slave depth=10000
 #pragma HLS INTERFACE s_axilite port=imageWidth bundle=control
 #pragma HLS INTERFACE s_axilite port=imageHeight bundle=control
 #pragma HLS INTERFACE s_axilite port=return bundle=control
    // Convolutional layer 1
   doub conv1Output[112*112*64];
   doub conv1_pad_output[230*230*3];
    convolution(flattenedImage, convolution1_weights,conv1_pad_output,conv1Output, 224, 224,3, 7,7 , convolution1_bias,64, 'S','p',2);

    // attention 1
    doub attention1_output[112*112*64];
    doub at1_conv_output[112*112*8];
    doub at1_ga_output[8];
    doub at1_fc1_output[64];
    doub at1_fc2_output[64];
    attentionblock(conv1Output,at1_convolution_weights,at1_conv_output,112,64,at1_convolution_biases,112*112*8,at1_ga_output,at1_fc1_output,at1_fc1_weights,at1_fc1_biases,8,at1_fc2_output,at1_fc2_weights,at1_fc2_biases,attention1_output);

     // Convolutional layer 2
     doub conv2Output[56*56*128];
   doub conv2_pad_output[114*114*64];
    convolution(attention1_output, convolution2_weights,conv2_pad_output,conv2Output, 112,112,64, 3,3 , convolution2_bias,128, 'S','p',2);

    // attention 2
    doub attention2_output[56*56*128];
    doub at2_conv_output[56*56*16];
    doub at2_ga_output[16];
    doub at2_fc1_output[128];
    doub at2_fc2_output[128];
    attentionblock(conv2Output,at2_convolution_weights,at2_conv_output,56,128,at2_convolution_biases,56*56*16,at2_ga_output,at2_fc1_output,at2_fc1_weights,at2_fc1_biases,16,at2_fc2_output,at2_fc2_weights,at2_fc2_biases,attention2_output);

      // Convolutional layer 3
   doub conv3Output[28*28*256];
   doub conv3_pad_output[58*58*128];
    convolution(attention2_output, convolution3_weights,conv3_pad_output,conv3Output, 56,56,128,3,3 , convolution3_bias,256, 'S','p',2);

    // attention 3
    doub attention3_output[28*28*256];
    doub at3_conv_output[28*28*32];
    doub at3_ga_output[32];
    doub at3_fc1_output[256];
    doub at3_fc2_output[256];
    attentionblock(conv3Output,at3_convolution_weights,at3_conv_output,28,256,at3_convolution_biases,28*28*32,at3_ga_output,at3_fc1_output,at3_fc1_weights,at3_fc1_biases,32,at3_fc2_output,at3_fc2_weights,at3_fc2_biases,attention3_output);

    // Convolutional layer 4
   doub conv4Output[14*14*512];
   doub conv4_pad_output[30*30*256];
    convolution(attention3_output, convolution4_weights,conv4_pad_output,conv4Output, 28,28,256,3,3 , convolution4_bias,512, 'S','p',2);

    // attention 4
    doub attention4_output[14*14*512];
    doub at4_conv_output[14*14*64];
    doub at4_ga_output[64];
    doub at4_fc1_output[512];
    doub at4_fc2_output[512];
    attentionblock(conv4Output,at4_convolution_weights,at4_conv_output,14,512,at4_convolution_biases,14*14*64,at4_ga_output,at4_fc1_output,at4_fc1_weights,at4_fc1_biases,64,at4_fc2_output,at4_fc2_weights,at4_fc2_biases,attention4_output);

    // Convolutional layer 5
   doub conv5Output[7*7*512];
   doub conv5_pad_output[16*16*512];
//#pragma HLS bind_storage variable=conv5_pad_output type=RAM_1P impl=LUTRAM

    convolution(attention4_output, convolution5_weights,conv5_pad_output,conv5Output, 14,14,512,3,3, convolution5_bias,512, 'S','p',2);

   // global average pooling layer
   doub gb_layeroutput[512];
   globalAveragePooling(conv5Output,gb_layeroutput,7,512);

   // dense layer 1
    doub fully1_output[2048];
//#pragma HLS bind_storage variable=fully1_output type=RAM_1P impl=LUTRAM

//#pragma HLS DATAFLOW
    //#pragma HLS ARRAY_PARTITION variable=fully1_output complete dim=1

    fullyConnectedLayer(gb_layeroutput,fully1_output,dense1_weights,dense1_biases,512,2048,'R');

    //dense layer 2
    doub fully2_output[2];
    fullyConnectedLayer(fully1_output,fully2_output,dense2_weights,dense2_biases,2048,2,'S');
}
