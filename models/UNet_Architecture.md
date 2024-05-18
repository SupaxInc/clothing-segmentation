# U-Net Architecture

- Architecture Overview
    1. **Contracting Path (Left Side of U):**
        - The contracting path consists of multiple blocks, each typically containing two 3x3 convolutional layers followed by a ReLU activation function.
        - Each block is usually followed by a 2x2 max pooling operation with stride 2, which reduces the spatial dimensions of the feature maps by half and increases the feature depth. This down sampling step helps in extracting increasingly abstract features from the input image at multiple scales.
    2. **Bottleneck (Bottom of U):**
        - At the bottom of the U, there is a bottleneck which usually contains two 3x3 convolutions and ReLU activations without any pooling. This part is crucial as it bridges the contracting and expansive paths.
    3. **Expansive Path (Right Side of U):**
        - The expansive path mirrors the contracting path but in reverse. It typically includes a series of up sampling steps, each followed by a 2x2 transposed convolution (or up-convolution) that doubles the dimensions of the feature maps.
        - After up sampling, the feature map is concatenated with the correspondingly cropped feature map from the contracting path. This step is crucial and is known as **skip connection**. It helps the network recover spatial information lost during down sampling.
        - Each up sampling is followed by two 3x3 convolutions and ReLU activations.
    4. **Final Layer:**
        - The architecture typically concludes with a 1x1 convolution that maps the final feature maps to the desired number of classes for segmentation.

## The Architecture

### Model

It uses a sequential model that runs the contained sub modules in sequence to simplify the construction of the neural network. These sub modules consists of a double convolution layer with a 3x3 filter. 

You can see in the architecture above each block consists of two 3x3 conv (blue arrow) layers and a ReLU activation layer. 

### Input Channels

Specifies the number of channels for the input image. E.g. a standard RGB image would have 3 channels (red, green, and blue).

### Output Channels

Specifies the number of filters that will be applied to the input image determining the number of feature maps the convolution layer will produce. 

### Kernel Size (Filters)

- Each **`Conv2d`** layer uses a 3x3 kernel (or filter). This is a small window (a matrix of weights) that moves across the input image or the previous feature map.
    - These matrix of weights are learned when training the neural network
- A 3x3 filter is a common choice because it's large enough to capture basic spatial structures like edges and corners, but small enough to keep computational costs reasonable. It effectively balances detail and computational efficiency.