#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <random>
#include "test_image.h"
#include "attention_cnn.h"
using namespace std;

int main() {
     int output_final[1];
    int ido=0;
    for(int k=0;k<1;k++){
    double flattenedImage[224*224*3];
    int idx = 0;
    for(int d=0;d<3;d++){
    for (int i = 0; i < 224; ++i) {
        for (int j = 0; j < 224; ++j) {
            flattenedImage[idx++] = test_image[ido++];
        }
    }
    }
        double output[2];  // Assuming the output size is 2
        CNN(flattenedImage,224,224, output);
       auto maxElementIterator = std::max_element(output, output + 2);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
    }

    return 0;
}
