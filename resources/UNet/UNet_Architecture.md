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

---

## The Module

It uses a sequential module that runs the contained sub modules in sequence to simplify the construction of the neural network. These sub modules consists of a double convolution layer with a 3x3 filter. 

It performs three sets of operations: convolution, batch normalization, and a ReLU activation.

You can see in the architecture above each block consists of two 3x3 conv (blue arrow) layers and a ReLU activation layer. There are 9 blocks in total.

## 2D Convolution Layer

Applies a filter to the input to extract features by sliding over the input spatially. It helps in detecting features like edges, textures, etc.

### Input Channels

Specifies the number of channels for the input image. E.g. a standard RGB image would have 3 channels (red, green, and blue).

---

### Output Channels

Specifies the number of filters that will be applied to the input image determining the number of feature maps the convolution layer will produce. E.g. a 1 output channel is a binary image segmentation task that just focuses on object and background or a 5 output channel is a multi-class channel that focuses on multiple categories such as different types of clothing, background, skin, etc.

---

### Kernel Size (Filters)

- Each **`Conv2d`** layer uses a 3x3 kernel (or filter). This is a small window (a matrix of weights) that moves across the input image or the previous feature map.
    - These matrix of weights are learned when training the neural network
    - A 3x3 filter is a common choice because it's large enough to capture basic spatial structures like edges and corners, but small enough to keep computational costs reasonable. It effectively balances detail and computational efficiency.

---

### Stride

![alt text](strided_convolution.png)

Controls the number of pixels you skip as you slide the filter across the input. A stride of 1 means the filter moves one pixel at a time, while a stride of 2 skips every other pixel. Adjusting the stride changes the dimensionality of the output feature map.

**Learn more here:** https://medium.com/swlh/convolutional-neural-networks-part-2-padding-and-strided-convolutions-c63c25026eaa

---

### Padding

Adds layers of zeros around the input image to allow the convolution to be applied to bordering elements of the input image. A padding of 1 means adding a one-pixel border on every edge.

Padding is useful as it prevents a shrinking output feature map which throws away information at the edges. See performing a convolution below to understand the problem further.

**Learn more here:** https://medium.com/swlh/convolutional-neural-networks-part-2-padding-and-strided-convolutions-c63c25026eaa

---


## ReLU Activation Layer

![alt text](activation_functions.png)

**`ReLU`** stands for Rectified Linear Unit, and it is a type of activation function that is commonly used in neural networks, especially in CNNs. The function itself is quite simple:

ReLU(𝑥)=max⁡(0,𝑥), this means that for each input we check the max between x and 0.

In our specific case, it helps apply non-linearity to the linear transformation performed after convolution and helps maintain the non-linearity in the presence of normalized inputs (batch normalization).

**Learn more here:** https://medium.com/@shrutijadon/survey-on-activation-functions-for-deep-learning-9689331ba092

### Importance

1. **Non-linearity**:
    - Neural networks, without non-linear activation functions like ReLU, would behave just like a single linear transformation, which limits their ability to model complex relationships in data like image classification and object detection.
2. **Sparse Activation**:
    - ReLU naturally leads to sparse activation. In any given layer, all the negative activations are set to zero, which means that only a subset of neurons fire at a given time. This sparsity makes the network more efficient and less prone to overfitting.
3. **Computational Efficiency**:
    - ReLU is computationally very efficient, both in terms of memory requirements and computational power. The simplicity of the threshold operation (comparing a value to zero) makes it much faster to compute than other non-linear functions like sigmoid or tanh.
4. **Gradient Flow**:
    - During backpropagation, ReLU helps with healthy gradient flow in deep networks. Since the gradient of the ReLU function is either 0 (for inputs less than zero) or 1 (for inputs greater than zero), it does not suffer from the vanishing gradient problem as severely as sigmoid or tanh functions. This characteristic allows models to learn faster and achieve better performance.
5. **Model Performance**:
    - Empirically, models using ReLU tend to perform better in practice on a variety of tasks compared to those using other activation functions. Its ability to provide a non-saturating non-linearity helps models converge quicker during training.


## Performing Convolution

Convolution is a mathematical operation used primarily in signal processing and also in image processing. In the context of convolutional neural network, you are sliding a filter (matrix of weights) across an input image or previous feature map and computing the dot product of the filter and the local region it covers it.

![alt text](performing_convolution.png)

In the example above, we performed two convolutions by convolving a 3x3 filter across a 6x6 input. The first output is equaled to 0 by convolving the 3x3 filter on the green square and grabbing the dot product. Similarly to the yellow square. Another key observation is that the output image became smaller and this is due to **valid convolution** (use of no padding). 

In the picture below, it shows how we essentially convolved the 3x3 filter (middle image) to detect a verticle edge within the 6x6 input (left greyscale image). The output feature map (right image) shows the location of the verticle edge where it is marked as 30's from the example above.

![alt text](output_images.png)


### Valid Convolution

Another key observation is that the output image became smaller and this is due to **valid convolution** (use of no padding). 

The formula for the output size in a valid convolution is: `n-f/s + 1` , where

- `n` is the input size
- `f` is the (kernel) filter size
- `s` is the stride

The reduction in size of the input will depend on the kernel size and the stride.

---

### Same Convolution

In a same convolution type, padding is added to the input so that the output feature map retains the same spatial dimensions (height and width) as the input feature map. 

The formula to achieve a “same convolution” is: `Padding = f-s / 2` , where

- `f` is the filter size
- `s` is the stride
- The result you get is the padding you should use to achieve same convolution.

In valid convolution, there are problems you’ll run into such as a shrinking output and throwing away information at the edges. This is shown in the picture below.

 ![alt text](same_convolution.png)

When convolving a 3x3 filter in a 6x6 input image, the green shaded square will only be slid across once while the yellow shaded square has been convolved multiple times over. Hence, the pixels on the corners or edges are used much less during computation of the output thus throwing away information. The 6x6 input image ends up being a 4x4 input image.

However, using the padding formula, we are able to get a padding to achieve “same convolution”, allowing us to preserve the size of the original image. As shown above, the new 9x9 input image with padding is preserved as the actual 6x6 input image.


## Channel Paths

Think of each channels in the U-Net architecture being used to capture more features. For example, one channel might become specialized in detecting edges, another in detecting textures, and another in capturing color gradients. As you go deeper into the network and the number of channels increases (64, 128, 256, 512), the kinds of features that the network can detect become more complex and varied.

Even though we begin with an `in_channel` of 3 for RGB image and and `out_channel`  of 5 and it increases during encoding, it does not mean we are adding further categories, instead, it's about enhancing the network's capability to capture and represent more complex and diverse features at various levels of abstraction.

### Down sampling (encoder)

The encoder's downsampling path, enhanced by max pooling, efficiently compresses the input image into a compact feature representation, capturing essential information at multiple scales. In U-Net and similar architectures, this is achieved through two main mechanisms:

1. **Convolutional Layers with Strides**: Each convolutional layer applies filters (or kernels) to the input, creating feature maps that highlight different aspects of the input data. As we go further down the channels, the network learns more complex and abstract features.
2. **Max Pooling Layers**: These are specifically designed to downsample the feature maps when going further down the channels. (this will be done in the forward step)
    1. It significantly reduces the spatial dimensions of the feature maps at each layer. For example, with a 2x2 pooling size and a stride of 2, the height and width of each feature map are halved. 
    2. This reduction in size helps decrease the number of computations needed in subsequent layers, which is particularly important as the number of channels increases.
    3. By reducing the resolution of the feature maps, max pooling also helps in preventing overfitting. The network becomes less sensitive to small variations and noise in the input data, focusing instead on higher-level features that are more robust and generalizable.

---

### Up sampling (decoder)

Upsampling in the expansive path of U-Net involves increasing the spatial dimensions of the feature maps to ultimately match the original input size, while simultaneously reducing the number of feature channels. It typically uses the following mechanisms:

1. **Transposed Convolutional Layers**:
    - These layers perform the inverse of convolutional operations, often referred to as deconvolution. Transposed convolutions introduce zeros between pixels in the input feature maps, which are then convolved with a kernel or filter to produce a larger output. They increase the spatial dimensions of the feature maps, aiming to reverse the downsampling effect produced by pooling in the encoder path.
        - **Learning Capability**: Unlike fixed methods for upsampling, transposed convolutions can learn parameters that help in better reconstructing the image details from the compressed feature representations.
2. **Bilinear Interpolation (chose this one)**:
    - Bilinear interpolation is a fixed mathematical method used for upsampling. It calculates the pixel value at a point based on a weighted average of the pixels around it. Bilinear interpolation generally produces smoother results compared to transposed convolution and can be faster since it doesn’t involve learnable parameters.
        - **Reduction in Artifacts**: Transposed convolutions can sometimes introduce checkerboard artifacts in the output due to uneven overlapping of the convolved pixels. Bilinear interpolation, being a non-learnable method, avoids these artifacts by providing a consistent and regular method for spatial upsampling.

The choice of upsampling technique (transposed convolution vs. bilinear interpolation) in the decoder can be driven by the specific demands of the segmentation task, balancing between detail recovery and computational efficiency.

---

### Bottleneck Layer

The bottleneck serves as the critical integration and transition point within the network. Comprising of double convolution layers (could be more), it focuses on further processing the highly compressed feature maps from the encoder. The main functions of the bottleneck include:

1. **Integrating Features**: It combines various abstract features extracted through the encoder, preparing them for effective reconstruction in the decoder.
2. **Reducing Overfitting**: By compressing the feature information into a compact form, the bottleneck helps minimize the model’s tendency to overfit, ensuring it focuses on essential features.
3. **Efficient Learning**: The bottleneck focuses the network’s learning capacity on critical features, improving learning efficiency and effectiveness.

This section is pivotal for ensuring that the network not only learns to compress the input data effectively but also sets the stage for accurately reconstructing the detailed segmentation map in the decoder.


## Final 1x1 Conv Layer

The final layer often includes a 1x1 convolution that serves as the critical endpoint for producing the output that directly corresponds to the segmentation task. It essentially adjusts the depth of the output feature maps to match the number of desired output classes. For instance, if the segmentation task involves distinguishing between multiple different objects (e.g., different types of clothing in your app), the number of output channels in this layer would equal the number of classes.

This layer's functions include:

1. **Pixel-wise Classification**: Through 1x1 convolutions, the network performs a classification at each pixel, utilizing the deep features extracted and refined through the previous layers to determine the class of each pixel.
2. **Channel Reduction**: It reduces the channel depth from the potentially large number of channels in the final feature maps of the upsampling path to the number of desired segmentation classes.
3. **Spatial Size Preservation**: This layer maintains the spatial dimensions of the input, ensuring that the output segmentation map matches the size of the original image, which is essential for accurate, pixel-level segmentation.

The final 1x1 convolution layer is pivotal in bridging the complex internal representations learned by the network with the practical requirements of the segmentation task, efficiently translating feature maps into class-specific segmentation outputs.

### SoftMax Activation

Once we have convolved the final layer with a 1x1 filter, we need to apply a SoftMax activation for multiclass segmentations. It ensures that the output values across all classes for each pixel sum to 1. Each output channel’s value at a pixel will represent the probability that the pixel belongs to one of the classes. In our case, we need to have different classification for different types of clothing. 

Softmax also forces output class probabilities for a pixel to compete with each other, this means that the increasing probability of one class decreases others. This is desirable in a setting where each pixel needs to be classified into one and only one of several categories. 

The Sigmoid function maps the same transformation as Softmax, however, Sigmoid is used for **binary classification** while Softmax is for **multiclass** so it is essentially an extension of the Sigmoid function. 

![Untitled](sigmoid_vs_softmax.png)

**Learn more here:** https://towardsdatascience.com/sigmoid-and-softmax-functions-in-5-minutes-f516c80ea1f9

### Cross-Entropy Loss

When we begin training, each epoch will be applied using a loss function called Cross-Entropy Loss. This is a probabilistic loss function that discovers the difference between the predicted probability distribution and the actual distribution (ground truth). 

It can be used in both binary and multi-class classifications. In our case, we will use it for multi-class classification, which will use the categorical cross-entropy loss function. This version of cross-entropy compares the distribution of the predictions (across multiple classes) with the true distribution which is typically a **one-hot encoded vector**.

This loss function benefits our use of Softmax activation as it is often paired with it. When we apply the loss function to our raw scores (logits) from the NN as inputs, it will internally apply the Softmax activation to convert them into probabilities. After applying the Softmax, it will then compute the cross entropy loss between the predicted probabilities and the target labels which should be provided as class indices (not as one-hot encoded vectors). This can be seen in our U-Net model where the forward function is only applying the final conv layer without any Softmax activation.

Combining Softmax and cross-entropy loss into a single operation is beneficial for a couple of reasons:

- **Numerical Stability**: Computing Softmax and then applying cross-entropy can lead to numerical instability due to the manipulation of small numbers (from the exponential operations in sSoftmax). PyTorch’s **`nn.CrossEntropyLoss`** handles this internally in a way that maintains numerical stability.
- **Efficiency**: Performing these operations together is more efficient than doing them separately, reducing computational overhead and simplifying the code.


## Forward Function

The "forward step" or "forward pass" refers to the process by which input data is passed through the network from the first to the last layer to produce an output. This process is crucial as it's where the network applies all the defined operations (like convolutions, activations, pooling, etc.) on the input data to compute the output, typically used during both training and inference phases. 

Here is the full process, pretty much summarizing what we’ve talked about above:

1. **Input Data**:
    - The forward step begins when the input data (typically an image in the case of U-Net) is fed into the network. For U-Net, this input is often a multi-channel image (such as an RGB image) that needs to be segmented.
2. **Downsampling Path (Encoder)**:
    - **Convolution and Activation Layers**: The input data first passes through multiple sets of convolutional layers where filters are applied to extract features. Each convolution is usually followed by a nonlinear activation function (like ReLU) which helps the network learn complex patterns.
    - **Pooling Layers**: Between the sets of convolutional layers, pooling operations (typically max pooling) reduce the spatial dimensions of the feature maps, focusing the network on the most important features while reducing computational load.
3. **Bottleneck**:
    - This is the central part of the network where the deepest features are processed. The bottleneck typically consists of several convolutional layers without pooling, designed to process the most abstract representations of the input data.
4. **Upsampling Path (Decoder)**:
    - **Transposed Convolutions or Upsampling**: This phase involves expanding the spatial dimensions of the feature maps. Transposed convolutions or simpler upsampling techniques (like bilinear upsampling) are used to increase the size of the feature maps.
    - **Concatenation with Skip Connections**: Crucially, the upsampling path often involves concatenating the upsampled features with the correspondingly cropped features from the downsampling path. These skip connections *(the grey arrows in the architecture diagram)* help the network recover spatial details that are lost during the downsampling phase.
        - During the downsampling process, spatial resolution is reduced, which can lead to a loss of detail. Skip connections help preserve these spatial details by bypassing the deeper layers of the network and directly feeding earlier, high-resolution features to later stages.
    - **Further Convolutions**: After each upsampling and concatenation, additional convolutions are applied to refine the features and integrate the information from the skip connections.
5. **Final Convolutional Layer**:
    - The output from the last upsampling step is passed through a final 1x1 convolutional layer, which adjusts the depth of the feature maps to the number of desired output classes. This layer effectively classifies each pixel in the original input image into one of the classes.
6. **Output**:
    - The result of the forward pass is a segmented image where each pixel is classified into one of the output classes. This output can directly correspond to different segments like different types of tissues in medical images or different objects in a scene for general segmentation tasks.


## Model Evaluation Metrics
The UNet model's performance is evaluated using two primary metrics: **Accuracy** and **Dice Score**. These metrics help us understand how well our model is performing on the segmentation tasks, particularly in terms of identifying and classifying each pixel into the correct class. You can see how this is being done in `scripts/utils.py`.

### **Accuracy**

Accuracy is calculated by comparing the predicted class labels against the true labels across all pixels in the validation dataset. It provides a straightforward measure of how often the model predicts the correct class at each pixel.

### **Dice Score**

We use the Dice Score, also known as the Sørensen-Dice coefficient which is a statistical tool that measures the overlap between the predicted segmentation and the ground truth mask labels. For each class we calculate the Dice Score by finding the ratio of twice the area of overlap (intersection) between the predicted and true masks to the total number of pixels in both the predicted and true masks. 

### **Implementation Details**

During evaluation:

- We pass each batch of images through the model to obtain the raw class scores (logits).
- The `torch.argmax` function is applied to these logits to determine the predicted class for each pixel.
- Accuracy is computed by determining the percentage of pixels where the predicted class matches the true class.
- The Dice Score is calculated for each class individually and then averaged across all classes to provide a comprehensive view of the model's performance across different types of segmentation tasks.

This evaluation process is crucial for tuning our model and ensuring it performs well across various challenging segmentation scenarios.

## Training

The `train.py` script is a crucial component of the training process for the U-Net model, designed for image segmentation tasks. This script does the following:

1. **Data Loading**:
    - The script utilizes PyTorch's `DataLoader` to batch and shuffle the input data, ensuring that each training iteration receives a different subset of the dataset. This helps in generalizing the model better.
    - The data transformations (augmentations) are applied to the training dataset to introduce variability, which helps in making the model robust to different input variations.

2. **Model Training**:
    - The core of the script is the `train_fn` function, where the actual training steps occur. This function takes the model, optimizer, loss function, and data loader as inputs.
    - For each batch from the data loader, the script performs a forward pass (calculating the model's predictions), computes the loss, and then performs a backward pass to update the model's weights.

3. **Automatic Mixed Precision (AMP)**:
    - The script uses PyTorch's AMP to accelerate training while maintaining the accuracy of computations. This is handled by the `scaler` which scales the loss to prevent small gradients from flushing to zero.

4. **Loss Computation and Score Calculation**:
    - The loss function used is the `CrossEntropyLoss`, which combines softmax activation and cross-entropy loss in a single function. This is computationally efficient and numerically stable.
    - The loss is calculated by comparing the model's predictions (logits) with the true labels (masks). Specifically, the `CrossEntropyLoss` function computes the negative log likelihood of the correct class, which has been softmaxed. This computation is crucial as it directly impacts the gradients used for updating the model weights.
    - After computing the loss, the `torch.argmax` function is applied to the logits to extract the class with the highest probability for each pixel, effectively converting the logits to final class predictions. To illustrate, consider logits output `[1.2, 0.1, -0.5]` for a pixel; applying softmax gives `[0.7, 0.2, 0.1]`, indicating probabilities for each class. The `torch.argmax` function then selects the index of the maximum value, which is `0` in this case, identifying the first class as the prediction for this pixel. This step is essential for evaluating the model's performance during training and validation.

5. **Optimizer and Scheduler**:
    - An Adam optimizer is used for adjusting the weights based on the computed gradients.
    - A learning rate scheduler reduces the learning rate when the validation loss plateaus, helping in fine-tuning the model in later stages of training.

6. **Model Evaluation**:
    - The script periodically evaluates the model on a validation set to monitor its performance on unseen data.
    - Metrics such as accuracy and Dice score are computed to assess the quality of the segmentation. Accuracy is calculated by comparing the predicted class labels against the true labels across all pixels in the validation dataset. The Dice Score is calculated by finding the ratio of twice the area of overlap (intersection) between the predicted and true masks to the total number of pixels in both the predicted and true masks. This provides a measure of how well the predicted segmentation matches the ground truth.

7. **Tensor Management**:
    - All tensors are moved to the appropriate device (CPU or GPU) to ensure efficient computation.
    - Tensors representing images and masks are properly reshaped and normalized to suit the model's requirements.

### Summary

The `train.py` script is a comprehensive framework for training the U-Net model on segmentation tasks. It efficiently handles data loading, model updates, and performance evaluation, ensuring that the model learns to segment images accurately. The use of AMP and a learning rate scheduler further optimizes the training process, making it faster and more responsive to changes in training dynamics.
