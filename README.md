1) Model Selection has been done using Python in the Google-Colab & related (.ipynb) file provided in the model selection folder.
2) Basic building blocks of DNN such as Convolution, Attention CNN block, Dense layer, ReLu, and SoftMax activations, etc. are provided as a High-level synthesis repository. For different applications the mentioned template, the model's structure, and the trained weights aggregate into the high-level description of the model and are provided in different cpp files consist of 32-bit float model and 8-bit INT model descriptions.
3) The obtained high-level description of the fully-trained transform into RTL logic through the HLS process using the Vitis-HLS compiler and the obtained RTL are provided in the RTL and IP integration folder.
4) The pynq overlay python script and other relevant files (bit, hwh, xmodel) are given in overlay files of the pynq-overlay repository and other vitis-ai folders consisting of Deep learning processor unit information.
   

NOTE: 1) Each (.ipynb) file consists of the results of DPU mapping and generated (.xmodel) file and (.h5) file through the Vitis-ai compiler. This repository only consists of RTL implementation of the attention CNN model for the visdrone19 drone imagery dataset.  To evaluate models for other applications you may follow the given procedure. 


In case of any inconvenience drop a mail to respective email id: pmi2017003@iiita.ac.in, yadav.49@iitj.ac.in
