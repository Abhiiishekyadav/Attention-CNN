#include "attention_cnn.h"
#pragma inline

double relu(double x)
{
    return (x < 0) ? 0 : x;
}

void relu2(double input[],int size){
    for(int i=0;i<size;i++){
        input[i]=relu(input[i]);
    }
}


void sigmoid(double input[], int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = 1 / (1 + std::exp(-input[i]));
    }
}

void softmax(const double input[], int size, double output[]) {
     double max_val = input[0];
    for (int i = 1; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        sum_exp += std::exp(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = std::exp(input[i] - max_val) / sum_exp;
        }

}

// Function to add padding to each channel of a 3D image
void addPadding(const double image[],double output[],int width, int height, int depth, int padding) {
    
    int new_width = width + 2 * padding;
    int new_height = height + 2 * padding;
 
    int len = new_width * new_height * depth;
    for(int i = 0; i < len; i++){
        output[i]=0;
    }

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int original_index = z * width * height + y * width + x;
                int padded_index = (z * new_height + y + padding) * new_width + (x + padding);          
                output[padded_index] = image[original_index];
            }
        }
    }
}

// Convulation operation
void convolution(const double flattenedImageinput[], const double kernels[], double pad_output[],double conv_output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const double biases[], int numKernels , char a_f,char p, int stride)
{
    int padding=0;
    if(p=='p'){
        padding = (k_h - 1) / 2;
    }
    int new_width = imageWidth + 2 * padding;
    int new_height = imageHeight + 2 * padding;
    if(p=='p'){
        addPadding(flattenedImageinput,pad_output,imageWidth,imageHeight,imageDepth,padding);
    }
    int output_h = (new_height - k_h) / stride + 1;
    int output_w = (new_width - k_w) / stride + 1;

    #pragma HLS UNROLL factor=2 

    for (int k = 0; k < numKernels; ++k)
    {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        #pragma HLS UNROLL factor=2

        for (int i = 0; i < output_h; ++i)
        {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
             #pragma HLS UNROLL factor=2

            for (int j = 0; j < output_w; ++j)
            {
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
                 #pragma HLS PIPELINE II=2

                conv_output[k * output_h * output_w + i * output_w + j] = biases[k]; 
                for (int kd = 0; kd < imageDepth; ++kd)
                {
                    #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                    for (int ki = 0; ki < k_h; ++ki)
                    {
                        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                        for (int kj = 0; kj < k_w; ++kj)
                        {
                            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                            int input_i = i * stride + ki;
                            int input_j = j * stride + kj;
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
void globalAveragePooling(const double image[], double output[], int imageWidth, int numChannels) {
    int i_h = imageWidth;
    int i_w = i_h;
    int total_elements = i_h * i_w;

    for (int c = 0; c < numChannels; ++c) {
        double sum_val = 0.0;
        for (int i = 0; i < i_h; ++i) {
            for (int j = 0; j < i_w; ++j) {
                sum_val += image[c * total_elements + i * i_w + j];
            }
        }
        double avg_val = sum_val / total_elements;
        output[c] = avg_val;
    }
}

// function to expand dimension of output to the dimension of input .  (inputs * tf.expand_dims(tf.expand_dims(y, 1), 1))
void expand_dimensions(const double input[],double output[],int width,int height,int channels, double y[]){
    int count=0;
    for(int i=0;i<channels;i++){
        double mul_val = y[i];
        for (int k = 0; k < height; ++k) {
            for (int j = 0; j < width; ++j) {
                output[count] = input[count]*mul_val;
                count++;
            }
        }
    }
}

// fully connected dense layer
void fullyConnectedLayer(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize, char a_f) {
    for (int i = 0; i < outputSize; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
            output[i] += input[j] * weights[i * inputSize + j];
        }

        if (a_f=='R') {
            output[i] = (output[i] < 0) ? 0 : output[i];
        }
    }
}

// generalized attention block function 
void attentionblock(double flattenedImageinput[],const double conv_kernels[],double conv_output[],int imageWidth, int imageDepth,const double conv_biases[],int size_conv_output, double ga_output[],double fc1_output[],const double fc1_weights[],const double fc1_bias[],int fc1_inputSize,double fc2_output[],const double fc2_weights[],const double fc2_bias[],double expand_output[]){
    int numKernels =  imageDepth/8;
    int fc1_outputSize = imageDepth;
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


void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]) {
#pragma HLS INTERFACE m_axi port=flattenedImage bundle=gmem0 offset=slave depth=7840000
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 offset=slave depth=10000
    #pragma HLS INTERFACE s_axilite port=imageWidth bundle=control
    #pragma HLS INTERFACE s_axilite port=imageHeight bundle=control
    #pragma HLS INTERFACE s_axilite port=flattenedImage bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Convolutional layer 1
   double conv1Output[112*112*64];
   double conv1_pad_output[230*230*3];
    convolution(flattenedImage, convolution1_weights,conv1_pad_output,conv1Output, 224, 224,3, 7,7 , convolution1_bias,64, 'S','p',2);

    // attention 1
    double attention1_output[112*112*64];
    double at1_conv_output[112*112*8];
    double at1_ga_output[8];
    double at1_fc1_output[64];
    double at1_fc2_output[64];
    attentionblock(conv1Output,at1_convolution_weights,at1_conv_output,112,64,at1_convolution_biases,112*112*8,at1_ga_output,at1_fc1_output,at1_fc1_weights,at1_fc1_biases,8,at1_fc2_output,at1_fc2_weights,at1_fc2_biases,attention1_output);

     // Convolutional layer 2
     double conv2Output[56*56*128];
   double conv2_pad_output[114*114*64];
    convolution(attention1_output, convolution2_weights,conv2_pad_output,conv2Output, 112,112,64, 3,3 , convolution2_bias,128, 'S','p',2);

    // attention 2
    double attention2_output[56*56*128];
    double at2_conv_output[56*56*16];
    double at2_ga_output[16];
    double at2_fc1_output[128];
    double at2_fc2_output[128];
    attentionblock(conv2Output,at2_convolution_weights,at2_conv_output,56,128,at2_convolution_biases,56*56*16,at2_ga_output,at2_fc1_output,at2_fc1_weights,at2_fc1_biases,16,at2_fc2_output,at2_fc2_weights,at2_fc2_biases,attention2_output);

      // Convolutional layer 3
   double conv3Output[28*28*256];
   double conv3_pad_output[58*58*128];
    convolution(attention2_output, convolution3_weights,conv3_pad_output,conv3Output, 56,56,128,3,3 , convolution3_bias,256, 'S','p',2);

    // attention 3
    double attention3_output[28*28*256];
    double at3_conv_output[28*28*32];
    double at3_ga_output[32];
    double at3_fc1_output[256];
    double at3_fc2_output[256];
    attentionblock(conv3Output,at3_convolution_weights,at3_conv_output,28,256,at3_convolution_biases,28*28*32,at3_ga_output,at3_fc1_output,at3_fc1_weights,at3_fc1_biases,32,at3_fc2_output,at3_fc2_weights,at3_fc2_biases,attention3_output);

    // Convolutional layer 4
   double conv4Output[14*14*512];
   double conv4_pad_output[30*30*256];
    convolution(attention3_output, convolution4_weights,conv4_pad_output,conv4Output, 28,28,256,3,3 , convolution4_bias,512, 'S','p',2);

    // attention 4
    double attention4_output[14*14*512];
    double at4_conv_output[14*14*64];
    double at4_ga_output[64];
    double at4_fc1_output[512];
    double at4_fc2_output[512];
    attentionblock(conv4Output,at4_convolution_weights,at4_conv_output,14,512,at4_convolution_biases,14*14*64,at4_ga_output,at4_fc1_output,at4_fc1_weights,at4_fc1_biases,64,at4_fc2_output,at4_fc2_weights,at4_fc2_biases,attention4_output);

    // Convolutional layer 5
   double conv5Output[7*7*512];
   double conv5_pad_output[16*16*512];
    convolution(attention4_output, convolution5_weights,conv5_pad_output,conv5Output, 14,14,512,3,3, convolution5_bias,512, 'S','p',2);

   // global average pooling layer
   double gb_layeroutput[512];
   globalAveragePooling(conv5Output,gb_layeroutput,7,512);

   // dense layer 1
    double fully1_output[2048];
    fullyConnectedLayer(gb_layeroutput,fully1_output,dense1_weights,dense1_biases,512,2048,'R');

    //dense layer 2
    double fully2_output[2];
    fullyConnectedLayer(fully1_output,fully2_output,dense2_weights,dense2_biases,2048,2,'S');
}


