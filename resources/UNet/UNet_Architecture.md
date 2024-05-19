# U-Net Architecture

![alt text](u-net-architecture.png)

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

## The Module

It uses a sequential module that runs the contained sub modules in sequence to simplify the construction of the neural network. These sub modules consists of a double convolution layer with a 3x3 filter. 

It performs three sets of operations: convolution, batch normalization, and a ReLU activation.

You can see in the architecture above each block consists of two 3x3 conv (blue arrow) layers and a ReLU activation layer. There are 9 blocks in total.

## 2D Convolution Layer

Applies a filter to the input to extract features by sliding over the input spatially. It helps in detecting features like edges, textures, etc.

### Input Channels

Specifies the number of channels for the input image. E.g. a standard RGB image would have 3 channels (red, green, and blue).

### Output Channels

Specifies the number of filters that will be applied to the input image determining the number of feature maps the convolution layer will produce. 

### Kernel Size (Filters)

- Each **`Conv2d`** layer uses a 3x3 kernel (or filter). This is a small window (a matrix of weights) that moves across the input image or the previous feature map.
    - These matrix of weights are learned when training the neural network
    - A 3x3 filter is a common choice because it's large enough to capture basic spatial structures like edges and corners, but small enough to keep computational costs reasonable. It effectively balances detail and computational efficiency.

### Stride

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f87cabf2-8d22-410c-bb4c-b00e5c7c3bac/bca960eb-fbd6-4122-8b2b-26d7743d4d42/Untitled.png)

Controls the number of pixels you skip as you slide the filter across the input. A stride of 1 means the filter moves one pixel at a time, while a stride of 2 skips every other pixel. Adjusting the stride changes the dimensionality of the output feature map.

**Learn more here:** https://medium.com/swlh/convolutional-neural-networks-part-2-padding-and-strided-convolutions-c63c25026eaa

### Padding

Adds layers of zeros around the input image to allow the convolution to be applied to bordering elements of the input image. A padding of 1 means adding a one-pixel border on every edge.

Padding is useful as it prevents a shrinking output feature map which throws away information at the edges. See performing a convolution below to understand the problem further.

**Learn more here:** https://medium.com/swlh/convolutional-neural-networks-part-2-padding-and-strided-convolutions-c63c25026eaa

## Performing Convolution

Convolution is a mathematical operation used primarily in signal processing and also in image processing. In the context of convolutional neural network, you are sliding a filter (matrix of weights) across an input image or previous feature map and computing the dot product of the filter and the local region it covers it.

![alt text](performing_convolution.png)

In the example above, we performed two convolutions by convolving a 3x3 filter across a 6x6 input. The first output is equaled to 0 by convolving the 3x3 filter on the green square and grabbing the dot product. Similarly to the yellow square.

Another key observation is that the output image became smaller and this is due to **valid convolution** (use of no padding). 

The formula for the output size in a valid convolution is: `n-f/s + 1` , where

- `n` is the input size
- `f` is the filter size
- `s` is the stride

![alt text](output_images.png)


### Valid Convolution

Another key observation is that the output image became smaller and this is due to **valid convolution** (use of no padding). 

The formula for the output size in a valid convolution is: `n-f/s + 1` , where

- `n` is the input size
- `f` is the (kernel) filter size
- `s` is the stride

The reduction in size of the input will depend on the kernel size and the stride.

### Same Convolution

In a same convolution type, padding is added to the input so that the output feature map retains the same spatial dimensions (height and width) as the input feature map. 

The formula to achieve a “same convolution” is: `Padding = f-s / 2` , where

- `f` is the filter size
- `s` is the stride
- The result you get is the padding you should use to achieve same convolution.

In valid convolution, there are problems you’ll run into such as a shrinking output and throwing away information at the edges. This is shown in the picture below.

 ![alt text](same_convolution.png)

When convolving a 3x3 filter in a 6x6 input image, the green shaded square will only be slid across once while the yellow shaded square has been convolved multiple times over. Hence, the pixels on the corners or edges are used much less during computation of the output thus throwing away information. The 6x6 input image ends up being a 4x4 input image.

However, using the padding formula, we are able to get a padding to achieve “same convolution”, allowing us to preserve the size of the original image.