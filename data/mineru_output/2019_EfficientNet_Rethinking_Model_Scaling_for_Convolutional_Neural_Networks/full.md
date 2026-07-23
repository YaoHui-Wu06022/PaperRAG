# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

Mingxing Tan <sup>1</sup> Quoc V. Le <sup>1</sup>

## Abstract

Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.

To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at https: //github.com/tensorflow/tpu/tree/ master/models/official/efficientnet.

## 1. Introduction

Scaling up ConvNets is widely used to achieve better accuracy. For example, ResNet (He et al., 2016) can be scaled up from ResNet-18 to ResNet-200 by using more layers; Recently, GPipe (Huang et al., 2018) achieved 84.3% ImageNet top-1 accuracy by scaling up a baseline model four time larger. However, the process of scaling up ConvNets has never been well understood and there are currently many ways to do it. The most common way is to scale up ConvNets by their depth (He et al., 2016) or width (Zagoruyko & Komodakis, 2016). Another less common, but increasingly popular, method is to scale up models by image resolution (Huang et al., 2018). In previous work, it is common to scale only one of the three dimensions – depth, width, and image size. Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.

![](images/68fd253d090da827f3ad35def63e9100c1cd6578171739f718c7ff45e80b6e90.jpg)  
Figure 1. Model Size vs. ImageNet Accuracy. All numbers are for single-crop, single-model. Our EfficientNets significantly outperform other ConvNets. In particular, EfficientNet-B7 achieves new state-of-the-art 84.3% top-1 accuracy but being 8.4x smaller and 6.1x faster than GPipe. EfficientNet-B1 is 7.6x smaller and 5.7x faster than ResNet-152. Details are in Table 2 and 4.

In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio. Based on this observation, we propose a simple yet effective compound scaling method. Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2 ^ { N }$ times more computational resources, then we can simply increase the network depth by $\alpha ^ { N }$ , width by $\beta ^ { N }$ , and image size by $\gamma ^ { N }$ , where $\alpha , \beta , \gamma$ are constant coefficients determined by a small grid search on the original small model. Figure 2 illustrates the difference between our scaling method and conventional methods.

![](images/48588cbfe439a101f16c0203893fc5016d5e956affed670ec01688c86c32920d.jpg)

![](images/78d33c04d085724d334cc8b839a6fe10a4470a14d7e7107c39e35282b7040899.jpg)

![](images/f28eca59b1d0183778b787950c9c646b424645613806f3f4072f99c5032a3680.jpg)  
(c) depth scaling

![](images/1042bfc4de3665fe1a917609eb3911769422c8d901a887f21b3ebfb104371530.jpg)  
(d) resolution scaling

![](images/def73617a8ddb3f894d2e7df5f0642cfe8699ca1f026443ca20a74f86720956b.jpg)  
(e) compound scaling  
Figure 2. Model Scaling. (a) is a baseline network example; (b)-(d) are conventional scaling that only increases one dimension of network width, depth, or resolution. (e) is our proposed compound scaling method that uniformly scales all three dimensions with a fixed ratio.

Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image. In fact, previous theoretical (Raghu et al., 2017; Lu et al., 2018) and empirical results (Zagoruyko & Komodakis, 2016) both show that there exists certain relationship between network width and depth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.

We demonstrate that our scaling method work well on existing MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search (Zoph & Le, 2017; Tan et al., 2019) to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets. In particular, our EfficientNet-B7 surpasses the best existing GPipe accuracy (Huang et al., 2018), but using 8.4x fewer parameters and running 6.1x faster on inference. Compared to the widely used ResNet-50 (He et al., 2016), our EfficientNet-B4 improves the top-1 accuracy from 76.3% to 83.0% (+6.7%) with similar FLOPS. Besides ImageNet, EfficientNets also transfer well and achieve stateof-the-art accuracy on 5 out of 8 widely used datasets, while reducing parameters by up to 21x than existing ConvNets.

## 2. Related Work

ConvNet Accuracy: Since AlexNet (Krizhevsky et al., 2012) won the 2012 ImageNet competition, ConvNets have become increasingly more accurate by going bigger: while the 2014 ImageNet winner GoogleNet (Szegedy et al., 2015) achieves 74.8% top-1 accuracy with about 6.8M parameters, the 2017 ImageNet winner SENet (Hu et al., 2018) achieves 82.7% top-1 accuracy with 145M parameters. Recently, GPipe (Huang et al., 2018) further pushes the state-of-the-art ImageNet top-1 validation accuracy to 84.3% using 557M parameters: it is so big that it can only be trained with a specialized pipeline parallelism library by partitioning the network and spreading each part to a different accelerator. While these models are mainly designed for ImageNet, recent studies have shown better ImageNet models also perform better across a variety of transfer learning datasets (Kornblith et al., 2019), and other computer vision tasks such as object detection (He et al., 2016; Tan et al., 2019). Although higher accuracy is critical for many applications, we have already hit the hardware memory limit, and thus further accuracy gain needs better efficiency.

ConvNet Efficiency: Deep ConvNets are often overparameterized. Model compression (Han et al., 2016; He et al., 2018; Yang et al., 2018) is a common way to reduce model size by trading accuracy for efficiency. As mobile phones become ubiquitous, it is also common to handcraft efficient mobile-size ConvNets, such as SqueezeNets (Iandola et al., 2016; Gholami et al., 2018), MobileNets (Howard et al., 2017; Sandler et al., 2018), and ShuffleNets (Zhang et al., 2018; Ma et al., 2018). Recently, neural architecture search becomes increasingly popular in designing efficient mobile-size ConvNets (Tan et al., 2019; Cai et al., 2019), and achieves even better efficiency than hand-crafted mobile ConvNets by extensively tuning the network width, depth, convolution kernel types and sizes. However, it is unclear how to apply these techniques for larger models that have much larger design space and much more expensive tuning cost. In this paper, we aim to study model efficiency for super large ConvNets that surpass state-of-the-art accuracy. To achieve this goal, we resort to model scaling.

Model Scaling: There are many ways to scale a ConvNet for different resource constraints: ResNet (He et al., 2016) can be scaled down (e.g., ResNet-18) or up (e.g., ResNet-200) by adjusting network depth (#layers), while WideResNet (Zagoruyko & Komodakis, 2016) and MobileNets (Howard et al., 2017) can be scaled by network width (#channels). It is also well-recognized that bigger input image size will help accuracy with the overhead of more FLOPS. Although prior studies (Raghu et al., 2017; Lin & Jegelka, 2018; Sharir & Shashua, 2018; Lu et al., 2018) have shown that network depth and width are both important for ConvNets’ expressive power, it still remains an open question of how to effectively scale a ConvNet to achieve better efficiency and accuracy. Our work systematically and empirically studies ConvNet scaling for all three dimensions of network width, depth, and resolutions.

## 3. Compound Model Scaling

In this section, we will formulate the scaling problem, study different approaches, and propose our new scaling method.

## 3.1. Problem Formulation

A ConvNet Layer i can be defined as a function: $Y _ { i } ~ =$ $\mathcal { F } _ { i } ( X _ { i } )$ , where ${ \mathcal { F } } _ { i }$ is the operator, $Y _ { i }$ is output tensor, $X _ { i }$ is input tensor, with tensor shape $\langle H _ { i } , W _ { i } , C _ { i } \rangle ^ { \top }$ , where $H _ { i }$ and $W _ { i }$ are spatial dimension and $C _ { i }$ is the channel dimension. A ConvNet $\mathcal { N }$ can be represented by a list of composed layers: $\mathcal { N } = \mathcal { F } _ { k } \odot . . . \odot \mathcal { F } _ { 2 } \odot \mathcal { F } _ { 1 } ( X _ { 1 } ) = \bigodot _ { i = 1 . . . k } \mathcal { F } _ { j } ( X _ { 1 } )$ . In practice, ConvNet layers are often partitioned into multiple stages and all layers in each stage share the same architecture: for example, ResNet (He et al., 2016) has five stages, and all layers in each stage has the same convolutional type except the first layer performs down-sampling. Therefore, we can define a ConvNet as:

$$
\mathcal {N} = \bigodot_ {i = 1 \dots s} \mathcal {F} _ {i} ^ {L _ {i}} \big (X _ {\langle H _ {i}, W _ {i}, C _ {i} \rangle} \big)\tag{1}
$$

where $\mathcal { F } _ { i } ^ { L _ { i } }$ denotes layer $F _ { i }$ is repeated $L _ { i }$ times in stage $i ,$ $\langle H _ { i } , W _ { i } , C _ { i } \rangle$ denotes the shape of input tensor X of layer i. Figure 2(a) illustrate a representative ConvNet, where the spatial dimension is gradually shrunk but the channel dimension is expanded over layers, for example, from initial input shape h224, 224, 3i to final output shape h7, 7, 512i.

Unlike regular ConvNet designs that mostly focus on finding the best layer architecture $\mathcal { F } _ { i } .$ , model scaling tries to expand the network length $( L _ { i } ) .$ , width $( C _ { i } )$ , and/or resolution $( H _ { i } , W _ { i } )$ without changing ${ \mathcal { F } } _ { i }$ predefined in the baseline network. By fixing $\mathcal { F } _ { i } .$ , model scaling simplifies the design problem for new resource constraints, but it still remains a large design space to explore different $L _ { i } , C _ { i } , H _ { i } , W _ { i }$ for each layer. In order to further reduce the design space, we restrict that all layers must be scaled uniformly with constant ratio. Our target is to maximize the model accuracy for any given resource constraints, which can be formulated as an optimization problem:

$$
\begin{array}{l l} \underset {d, w, r} {\max} & A c c u r a c y \big (\mathcal {N} (d, w, r) \big) \\ s. t. & \mathcal {N} (d, w, r) = \underset {i = 1 \dots s} {\bigodot} \hat {\mathcal {F}} _ {i} ^ {d \cdot \hat {L} _ {i}} \big (X _ {\langle r \cdot \hat {H} _ {i}, r \cdot \hat {W} _ {i}, w \cdot \hat {C} _ {i} \rangle} \big) \\ & \text { Memory } (\mathcal {N}) \leq \text { target\_memory } \\ & \text { FLOPS } (\mathcal {N}) \leq \text { target\_flops } \end{array}\tag{2}
$$

where $w , d , r$ are coefficients for scaling network width, depth, and resolution; $\hat { \mathcal { F } } _ { i } , \hat { L } _ { i } , \hat { H } _ { i } , \hat { W } _ { i } , \hat { C } _ { i }$ are predefined parameters in baseline network (see Table 1 as an example).

## 3.2. Scaling Dimensions

The main difficulty of problem 2 is that the optimal $d , w , r$ depend on each other and the values change under different resource constraints. Due to this difficulty, conventional methods mostly scale ConvNets in one of these dimensions:

Depth (d): Scaling network depth is the most common way used by many ConvNets (He et al., 2016; Huang et al., 2017; Szegedy et al., 2015; 2016). The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem (Zagoruyko & Komodakis, 2016). Although several techniques, such as skip connections (He et al., 2016) and batch normalization (Ioffe & Szegedy, 2015), alleviate the training problem, the accuracy gain of very deep network diminishes: for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers. Figure 3 (middle) shows our empirical study on scaling a baseline model with different depth coefficient $d ,$ further suggesting the diminishing accuracy return for very deep ConvNets.

Width (w): Scaling network width is commonly used for small size models (Howard et al., 2017; Sandler et al., 2018;

![](images/f8bdda7793f1ea7211509bf5979e98438a6db790b95879b4475f45b8fdd040d6.jpg)

![](images/f8aa6cb05f8cfc2ce22964c3e56f62170836f08c20da57a28ebf321d9a1b0a21.jpg)

![](images/1115927b1f84d9c00f442d24eaef6d3572a3d2eb96e5bfe790958bb02b222e79.jpg)  
Figure 3. Scaling Up a Baseline Model with Different Network Width (w), Depth (d), and Resolution (r) Coefficients. Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accuracy gain quickly saturate after reaching 80%, demonstrating the limitation of single dimension scaling. Baseline network is described in Table 1.

Tan et al., 2019)<sup>2</sup>. As discussed in (Zagoruyko & Komodakis, 2016), wider networks tend to be able to capture more fine-grained features and are easier to train. However, extremely wide but shallow networks tend to have difficulties in capturing higher level features. Our empirical results in Figure 3 (left) show that the accuracy quickly saturates when networks become much wider with larger w.

Resolution (r): With higher resolution input images, ConvNets can potentially capture more fine-grained patterns. Starting from 224x224 in early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331 (Zoph et al., 2018) for better accuracy. Recently, GPipe (Huang et al., 2018) achieves state-of-the-art ImageNet accuracy with 480x480 resolution. Higher resolutions, such as 600x600, are also widely used in object detection ConvNets (He et al., 2017; Lin et al., 2017). Figure 3 (right) shows the results of scaling network resolutions, where indeed higher resolutions improve accuracy, but the accuracy gain diminishes for very high resolutions (r = 1.0 denotes resolution 224x224 and r = 2.5 denotes resolution 560x560).

The above analyses lead us to the first observation:

Observation 1 – Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

## 3.3. Compound Scaling

We empirically observe that different scaling dimensions are not independent. Intuitively, for higher resolution images, we should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images. Correspondingly, we should also increase network width when resolution is higher, in order to capture more fine-grained patterns with more pixels in high resolution images. These intuitions suggest that we need to coordinate and balance different scaling dimensions rather than conventional single-dimension scaling.

![](images/0295f6a24e2ca4ed9a8591139088d71fe661231b4ef163de3c953758d6b727ec.jpg)  
Figure 4. Scaling Network Width for Different Baseline Networks. Each dot in a line denotes a model with different width coefficient (w). All baseline networks are from Table 1. The first baseline network (d=1.0, r=1.0) has 18 convolutional layers with resolution 224x224, while the last baseline (d=2.0, r=1.3) has 36 layers with resolution 299x299.

To validate our intuitions, we compare width scaling under different network depths and resolutions, as shown in Figure 4. If we only scale network width w without changing depth (d=1.0) and resolution (r=1.0), the accuracy saturates quickly. With deeper (d=2.0) and higher resolution (r=2.0), width scaling achieves much better accuracy under the same FLOPS cost. These results lead us to the second observation:

Observation 2 – In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

In fact, a few prior work (Zoph et al., 2018; Real et al., 2019) have already tried to arbitrarily balance network width and depth, but they all require tedious manual tuning.

In this paper, we propose a new compound scaling method, which use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way:

$$
\begin{array}{c} \text {depth:} d = \alpha^ {\phi} \\ \text {width:} w = \beta^ {\phi} \\ \text {resolution:} r = \gamma^ {\phi} \\ \text {s.t.} \alpha \cdot \beta^ {2} \cdot \gamma^ {2} \approx 2 \\ \alpha \geq 1, \beta \geq 1, \gamma \geq 1 \end{array}\tag{3}
$$

where $\alpha , \beta , \gamma$ are constants that can be determined by a small grid search. Intuitively, φ is a user-specified coefficient that controls how many more resources are available for model scaling, while $\alpha , \beta , \gamma$ specify how to assign these extra resources to network width, depth, and resolution respectively. Notably, the FLOPS of a regular convolution op is proportional to $d , w ^ { 2 } , r ^ { 2 }$ , i.e., doubling network depth will double FLOPS, but doubling network width or resolution will increase FLOPS by four times. Since convolution ops usually dominate the computation cost in ConvNets, scaling a ConvNet with equation 3 will approximately increase total FLOPS by $\left( \alpha \cdot \beta ^ { 2 } \cdot \gamma ^ { 2 } \right) ^ { \phi }$ . In this paper, we constraint α · $\beta ^ { 2 } \cdot \gamma ^ { 2 } \approx 2$ such that for any new φ, the total FLOPS will approximately<sup>3</sup> increase by $2 ^ { \phi }$

## 4. EfficientNet Architecture

Since model scaling does not change layer operators $\hat { \mathcal { F } } _ { i }$ in baseline network, having a good baseline network is also critical. We will evaluate our scaling method using existing ConvNets, but in order to better demonstrate the effectiveness of our scaling method, we have also developed a new mobile-size baseline, called EfficientNet.

Inspired by (Tan et al., 2019), we develop our baseline network by leveraging a multi-objective neural architecture search that optimizes both accuracy and FLOPS. Specifically, we use the same search space as (Tan et al., 2019), and use $A C C ( m ) \times [ F L O P S ( m ) / T ] ^ { w }$ as the optimization goal, where $A C C ( m )$ and $F L O P S ( m )$ denote the accuracy and FLOPS of model m, T is the target FLOPS and $w { = } { - } 0 . 0 7$ is a hyperparameter for controlling the trade-off between accuracy and FLOPS. Unlike (Tan et al., 2019; Cai et al., 2019), here we optimize FLOPS rather than latency since we are not targeting any specific hardware device. Our search produces an efficient network, which we name EfficientNet-B0. Since we use the same search space as (Tan et al., 2019), the architecture is similar to Mnas-

Table 1. EfficientNet-B0 baseline network – Each row describes a stage i with $\hat { L } _ { i }$ layers, with input resolution $\langle \hat { H } _ { i } , \hat { W } _ { i } \rangle$ and output channels $\hat { C } _ { i }$ . Notations are adopted from equation 2.

<table><tr><td>Stage i</td><td>Operator $\hat{\mathcal{F}}_i$ </td><td>Resolution $\hat{H}_i \times \hat{W}_i$ </td><td>#Channels $\hat{C}_i$ </td><td>#Layers $\hat{L}_i$ </td></tr><tr><td>1</td><td>Conv3x3</td><td>224 × 224</td><td>32</td><td>1</td></tr><tr><td>2</td><td>MBConv1, k3x3</td><td>112 × 112</td><td>16</td><td>1</td></tr><tr><td>3</td><td>MBConv6, k3x3</td><td>112 × 112</td><td>24</td><td>2</td></tr><tr><td>4</td><td>MBConv6, k5x5</td><td>56 × 56</td><td>40</td><td>2</td></tr><tr><td>5</td><td>MBConv6, k3x3</td><td>28 × 28</td><td>80</td><td>3</td></tr><tr><td>6</td><td>MBConv6, k5x5</td><td>14 × 14</td><td>112</td><td>3</td></tr><tr><td>7</td><td>MBConv6, k5x5</td><td>14 × 14</td><td>192</td><td>4</td></tr><tr><td>8</td><td>MBConv6, k3x3</td><td>7 × 7</td><td>320</td><td>1</td></tr><tr><td>9</td><td>Conv1x1 &amp; Pooling &amp; FC</td><td>7 × 7</td><td>1280</td><td>1</td></tr></table>

Net, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation optimization (Hu et al., 2018).

Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:

• STEP 1: we first fix $\phi = 1$ , assuming twice more resources available, and do a small grid search of $\alpha , \beta , \gamma$ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are $\alpha = 1 . 2 , \beta =$ $1 . 1 , \gamma = 1 . 1 5$ , under constraint of $\alpha \cdot \beta ^ { 2 } \cdot \gamma ^ { 2 } \approx 2$

• STEP 2: we then fix $\alpha , \beta , \gamma$ as constants and scale up baseline network with different φ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2).

Notably, it is possible to achieve even better performance by searching for $\alpha , \beta ,$ γ directly around a large model, but the search cost becomes prohibitively more expensive on larger models. Our method solves this issue by only doing search once on the small baseline network (step 1), and then use the same scaling coefficients for all other models (step 2).

## 5. Experiments

In this section, we will first evaluate our scaling method on existing ConvNets and the new proposed EfficientNets.

## 5.1. Scaling Up MobileNets and ResNets

As a proof of concept, we first apply our scaling method to the widely-used MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Table 3 shows the ImageNet results of scaling them in different ways. Compared to other single-dimension scaling methods, our compound scaling method improves the accuracy on all these models, suggesting the effectiveness of our proposed scaling method for general existing ConvNets.

Table 2. EfficientNet Performance Results on ImageNet (Russakovsky et al., 2015). All EfficientNet models are scaled from our baseline EfficientNet-B0 using different compound coefficient φ in Equation 3. ConvNets with similar top-1/top-5 accuracy are grouped together for efficiency comparison. Our scaled EfficientNet models consistently reduce parameters and FLOPS by an order of magnitude (up to 8.4x parameter reduction and up to 16x FLOPS reduction) than existing ConvNets.

<table><tr><td>Model</td><td>Top-1 Acc.</td><td>Top-5 Acc.</td><td>#Params</td><td>Ratio-to-EfficientNet</td><td>#FLOPs</td><td>Ratio-to-EfficientNet</td></tr><tr><td>EfficientNet-B0</td><td>77.1%</td><td>93.3%</td><td>5.3M</td><td>1x</td><td>0.39B</td><td>1x</td></tr><tr><td>ResNet-50 (He et al., 2016)</td><td>76.0%</td><td>93.0%</td><td>26M</td><td>4.9x</td><td>4.1B</td><td>11x</td></tr><tr><td>DenseNet-169 (Huang et al., 2017)</td><td>76.2%</td><td>93.2%</td><td>14M</td><td>2.6x</td><td>3.5B</td><td>8.9x</td></tr><tr><td>EfficientNet-B1</td><td>79.1%</td><td>94.4%</td><td>7.8M</td><td>1x</td><td>0.70B</td><td>1x</td></tr><tr><td>ResNet-152 (He et al., 2016)</td><td>77.8%</td><td>93.8%</td><td>60M</td><td>7.6x</td><td>11B</td><td>16x</td></tr><tr><td>DenseNet-264 (Huang et al., 2017)</td><td>77.9%</td><td>93.9%</td><td>34M</td><td>4.3x</td><td>6.0B</td><td>8.6x</td></tr><tr><td>Inception-v3 (Szegedy et al., 2016)</td><td>78.8%</td><td>94.4%</td><td>24M</td><td>3.0x</td><td>5.7B</td><td>8.1x</td></tr><tr><td>Xception (Chollet, 2017)</td><td>79.0%</td><td>94.5%</td><td>23M</td><td>3.0x</td><td>8.4B</td><td>12x</td></tr><tr><td>EfficientNet-B2</td><td>80.1%</td><td>94.9%</td><td>9.2M</td><td>1x</td><td>1.0B</td><td>1x</td></tr><tr><td>Inception-v4 (Szegedy et al., 2017)</td><td>80.0%</td><td>95.0%</td><td>48M</td><td>5.2x</td><td>13B</td><td>13x</td></tr><tr><td>Inception-resnet-v2 (Szegedy et al., 2017)</td><td>80.1%</td><td>95.1%</td><td>56M</td><td>6.1x</td><td>13B</td><td>13x</td></tr><tr><td>EfficientNet-B3</td><td>81.6%</td><td>95.7%</td><td>12M</td><td>1x</td><td>1.8B</td><td>1x</td></tr><tr><td>ResNeXt-101 (Xie et al., 2017)</td><td>80.9%</td><td>95.6%</td><td>84M</td><td>7.0x</td><td>32B</td><td>18x</td></tr><tr><td>PolyNet (Zhang et al., 2017)</td><td>81.3%</td><td>95.8%</td><td>92M</td><td>7.7x</td><td>35B</td><td>19x</td></tr><tr><td>EfficientNet-B4</td><td>82.9%</td><td>96.4%</td><td>19M</td><td>1x</td><td>4.2B</td><td>1x</td></tr><tr><td>SENet (Hu et al., 2018)</td><td>82.7%</td><td>96.2%</td><td>146M</td><td>7.7x</td><td>42B</td><td>10x</td></tr><tr><td>NASNet-A (Zoph et al., 2018)</td><td>82.7%</td><td>96.2%</td><td>89M</td><td>4.7x</td><td>24B</td><td>5.7x</td></tr><tr><td>AmoebaNet-A (Real et al., 2019)</td><td>82.8%</td><td>96.1%</td><td>87M</td><td>4.6x</td><td>23B</td><td>5.5x</td></tr><tr><td>PNASNet (Liu et al., 2018)</td><td>82.9%</td><td>96.2%</td><td>86M</td><td>4.5x</td><td>23B</td><td>6.0x</td></tr><tr><td>EfficientNet-B5</td><td>83.6%</td><td>96.7%</td><td>30M</td><td>1x</td><td>9.9B</td><td>1x</td></tr><tr><td>AmoebaNet-C (Cubuk et al., 2019)</td><td>83.5%</td><td>96.5%</td><td>155M</td><td>5.2x</td><td>41B</td><td>4.1x</td></tr><tr><td>EfficientNet-B6</td><td>84.0%</td><td>96.8%</td><td>43M</td><td>1x</td><td>19B</td><td>1x</td></tr><tr><td>EfficientNet-B7</td><td>84.3%</td><td>97.0%</td><td>66M</td><td>1x</td><td>37B</td><td>1x</td></tr><tr><td>GPipe (Huang et al., 2018)</td><td>84.3%</td><td>97.0%</td><td>557M</td><td>8.4x</td><td>-</td><td>-</td></tr></table>

We omit ensemble and multi-crop models (Hu et al., 2018), or models pretrained on 3.5B Instagram images (Mahajan et al., 2018).

Table 3. Scaling Up MobileNets and ResNet.

<table><tr><td>Model</td><td>FLOPS</td><td>Top-1 Acc.</td></tr><tr><td>Baseline MobileNetV1 (Howard et al., 2017)</td><td>0.6B</td><td>70.6%</td></tr><tr><td>Scale MobileNetV1 by width ( $w=2$ )</td><td>2.2B</td><td>74.2%</td></tr><tr><td>Scale MobileNetV1 by resolution ( $r=2$ )</td><td>2.2B</td><td>72.7%</td></tr><tr><td>compound scale ( $d=1.4, w=1.2, r=1.3$ )</td><td>2.3B</td><td>75.6%</td></tr><tr><td>Baseline MobileNetV2 (Sandler et al., 2018)</td><td>0.3B</td><td>72.0%</td></tr><tr><td>Scale MobileNetV2 by depth ( $d=4$ )</td><td>1.2B</td><td>76.8%</td></tr><tr><td>Scale MobileNetV2 by width ( $w=2$ )</td><td>1.1B</td><td>76.4%</td></tr><tr><td>Scale MobileNetV2 by resolution ( $r=2$ )</td><td>1.2B</td><td>74.8%</td></tr><tr><td>MobileNetV2 compound scale</td><td>1.3B</td><td>77.4%</td></tr><tr><td>Baseline ResNet-50 (He et al., 2016)</td><td>4.1B</td><td>76.0%</td></tr><tr><td>Scale ResNet-50 by depth ( $d=4$ )</td><td>16.2B</td><td>78.1%</td></tr><tr><td>Scale ResNet-50 by width ( $w=2$ )</td><td>14.7B</td><td>77.7%</td></tr><tr><td>Scale ResNet-50 by resolution ( $r=2$ )</td><td>16.4B</td><td>77.5%</td></tr><tr><td>ResNet-50 compound scale</td><td>16.7B</td><td>78.8%</td></tr></table>

Table 4. Inference Latency Comparison – Latency is measured with batch size 1 on a single core of Intel Xeon CPU E5-2690.

<table><tr><td></td><td>Acc. @ Latency</td><td></td><td>Acc. @ Latency</td></tr><tr><td>ResNet-152</td><td>77.8% @ 0.554s</td><td>GPipe</td><td>84.3% @ 19.0s</td></tr><tr><td>EfficientNet-B1</td><td>78.8% @ 0.098s</td><td>EfficientNet-B7</td><td>84.4% @ 3.1s</td></tr><tr><td>Speedup</td><td>5.7x</td><td>Speedup</td><td>6.1x</td></tr></table>

![](images/529d717380f76da8312085bf2496e6e0e8ae2e05234a726c89c19a78a5ca172c.jpg)  
Figure 5. FLOPS vs. ImageNet Accuracy – Similar to Figure 1 except it compares FLOPS rather than model size.

## 5.2. ImageNet Results for EfficientNet

We train our EfficientNet models on ImageNet using similar settings as (Tan et al., 2019): RMSProp optimizer with decay 0.9 and momentum 0.9; batch norm momentum 0.99;

Table 5. EfficientNet Performance Results on Transfer Learning Datasets. Our scaled EfficientNet models achieve new state-of-theart accuracy for 5 out of 8 datasets, with 9.6x fewer parameters on average.

<table><tr><td rowspan="2"></td><td colspan="6">Comparison to best public-available results</td><td colspan="6">Comparison to best reported results</td></tr><tr><td>Model</td><td>Acc.</td><td>#Param</td><td>Our Model</td><td>Acc.</td><td>#Param(ratio)</td><td>Model</td><td>Acc.</td><td>#Param</td><td>Our Model</td><td>Acc.</td><td>#Param(ratio)</td></tr><tr><td>CIFAR-10</td><td>NASNet-A</td><td>98.0%</td><td>85M</td><td>EfficientNet-B0</td><td>98.1%</td><td>4M (21x)</td><td> $^\dagger$ Gpipe</td><td>99.0%</td><td>556M</td><td>EfficientNet-B7</td><td>98.9%</td><td>64M (8.7x)</td></tr><tr><td>CIFAR-100</td><td>NASNet-A</td><td>87.5%</td><td>85M</td><td>EfficientNet-B0</td><td>88.1%</td><td>4M (21x)</td><td>Gpipe</td><td>91.3%</td><td>556M</td><td>EfficientNet-B7</td><td>91.7%</td><td>64M (8.7x)</td></tr><tr><td>Birdsnap</td><td>Inception-v4</td><td>81.8%</td><td>41M</td><td>EfficientNet-B5</td><td>82.0%</td><td>28M (1.5x)</td><td>GPipe</td><td>83.6%</td><td>556M</td><td>EfficientNet-B7</td><td>84.3%</td><td>64M (8.7x)</td></tr><tr><td>Stanford Cars</td><td>Inception-v4</td><td>93.4%</td><td>41M</td><td>EfficientNet-B3</td><td>93.6%</td><td>10M (4.1x)</td><td> $^\ddagger$ DAT</td><td>94.8%</td><td>-</td><td>EfficientNet-B7</td><td>94.7%</td><td>-</td></tr><tr><td>Flowers</td><td>Inception-v4</td><td>98.5%</td><td>41M</td><td>EfficientNet-B5</td><td>98.5%</td><td>28M (1.5x)</td><td>DAT</td><td>97.7%</td><td>-</td><td>EfficientNet-B7</td><td>98.8%</td><td>-</td></tr><tr><td>FGVC Aircraft</td><td>Inception-v4</td><td>90.9%</td><td>41M</td><td>EfficientNet-B3</td><td>90.7%</td><td>10M (4.1x)</td><td>DAT</td><td>92.9%</td><td>-</td><td>EfficientNet-B7</td><td>92.9%</td><td>-</td></tr><tr><td>Oxford-IIIT Pets</td><td>ResNet-152</td><td>94.5%</td><td>58M</td><td>EfficientNet-B4</td><td>94.8%</td><td>17M (5.6x)</td><td>GPipe</td><td>95.9%</td><td>556M</td><td>EfficientNet-B6</td><td>95.4%</td><td>41M (14x)</td></tr><tr><td>Food-101</td><td>Inception-v4</td><td>90.8%</td><td>41M</td><td>EfficientNet-B4</td><td>91.5%</td><td>17M (2.4x)</td><td>GPipe</td><td>93.0%</td><td>556M</td><td>EfficientNet-B7</td><td>93.0%</td><td>64M (8.7x)</td></tr><tr><td>Geo-Mean</td><td></td><td></td><td></td><td></td><td></td><td>(4.7x)</td><td></td><td></td><td></td><td></td><td></td><td>(9.6x)</td></tr></table>

<sup>†</sup>GPipe (Huang et al., 2018) trains giant models with specialized pipeline parallelism library.  
<sup>‡</sup>DAT denotes domain adaptive transfer learning (Ngiam et al., 2018). Here we only compare ImageNet-based transfer learning results.  
Transfer accuracy and #params for NASNet (Zoph et al., 2018), Inception-v4 (Szegedy et al., 2017), ResNet-152 (He et al., 2016) are from (Kornblith et al., 2019).

![](images/c4c7b69f75bb806b6ebfec64662bfa03a452227d05a058e350370b7511d86b14.jpg)  
Figure 6. Model Parameters vs. Transfer Learning Accuracy – All models are pretrained on ImageNet and finetuned on new datasets.

weight decay 1e-5; initial learning rate 0.256 that decays by 0.97 every 2.4 epochs. We also use SiLU (Swish-1) activation (Ramachandran et al., 2018; Elfwing et al., 2018; Hendrycks & Gimpel, 2016), AutoAugment (Cubuk et al., 2019), and stochastic depth (Huang et al., 2016) with survival probability 0.8. As commonly known that bigger models need more regularization, we linearly increase dropout (Srivastava et al., 2014) ratio from 0.2 for EfficientNet-B0 to 0.5 for B7. We reserve 25K randomly picked images from the training set as a minival set, and perform early stopping on this minival; we then evaluate the earlystopped checkpoint on the original validation set to report the final validation accuracy.

Table 2 shows the performance of all EfficientNet models that are scaled from the same baseline EfficientNet-B0. Our EfficientNet models generally use an order of magnitude fewer parameters and FLOPS than other ConvNets with similar accuracy. In particular, our EfficientNet-B7 achieves 84.3% top1 accuracy with 66M parameters and 37B FLOPS, being more accurate but 8.4x smaller than the previous best GPipe (Huang et al., 2018). These gains come from both better architectures, better scaling, and better training settings that are customized for EfficientNet.

Figure 1 and Figure 5 illustrates the parameters-accuracy and FLOPS-accuracy curve for representative ConvNets, where our scaled EfficientNet models achieve better accuracy with much fewer parameters and FLOPS than other ConvNets. Notably, our EfficientNet models are not only small, but also computational cheaper. For example, our EfficientNet-B3 achieves higher accuracy than ResNeXt-101 (Xie et al., 2017) using 18x fewer FLOPS.

To validate the latency, we have also measured the inference latency for a few representative CovNets on a real CPU as shown in Table 4, where we report average latency over 20 runs. Our EfficientNet-B1 runs 5.7x faster than the widely used ResNet-152, while EfficientNet-B7 runs about 6.1x faster than GPipe (Huang et al., 2018), suggesting our EfficientNets are indeed fast on real hardware.

![](images/03ee21fdf7f79ac42f9bb4862f75db044a89386e77e57bd949f639bbface9740.jpg)  
Figure 7. Class Activation Map (CAM) (Zhou et al., 2016) for Models with different scaling methods- Our compound scaling method allows the scaled model (last column) to focus on more relevant regions with more object details. Model details are in Table 7.

Table 6. Transfer Learning Datasets.

<table><tr><td>Dataset</td><td>Train Size</td><td>Test Size</td><td>#Classes</td></tr><tr><td>CIFAR-10 (Krizhevsky &amp; Hinton, 2009)</td><td>50,000</td><td>10,000</td><td>10</td></tr><tr><td>CIFAR-100 (Krizhevsky &amp; Hinton, 2009)</td><td>50,000</td><td>10,000</td><td>100</td></tr><tr><td>Birdsnap (Berg et al., 2014)</td><td>47,386</td><td>2,443</td><td>500</td></tr><tr><td>Stanford Cars (Krause et al., 2013)</td><td>8,144</td><td>8,041</td><td>196</td></tr><tr><td>Flowers (Nilsback &amp; Zisserman, 2008)</td><td>2,040</td><td>6,149</td><td>102</td></tr><tr><td>FGVC Aircraft (Maji et al., 2013)</td><td>6,667</td><td>3,333</td><td>100</td></tr><tr><td>Oxford-IIIT Pets (Parkhi et al., 2012)</td><td>3,680</td><td>3,369</td><td>37</td></tr><tr><td>Food-101 (Bossard et al., 2014)</td><td>75,750</td><td>25,250</td><td>101</td></tr></table>

## 5.3. Transfer Learning Results for EfficientNet

We have also evaluated our EfficientNet on a list of commonly used transfer learning datasets, as shown in Table 6. We borrow the same training settings from (Kornblith et al., 2019) and (Huang et al., 2018), which take ImageNet pretrained checkpoints and finetune on new datasets.

Table 5 shows the transfer learning performance: (1) Compared to public available models, such as NASNet-A (Zoph et al., 2018) and Inception-v4 (Szegedy et al., 2017), our EfficientNet models achieve better accuracy with 4.7x average (up to 21x) parameter reduction. (2) Compared to stateof-the-art models, including DAT (Ngiam et al., 2018) that dynamically synthesizes training data and GPipe (Huang et al., 2018) that is trained with specialized pipeline parallelism, our EfficientNet models still surpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters

Figure 6 compares the accuracy-parameters curve for a variety of models. In general, our EfficientNets consistently achieve better accuracy with an order of magnitude fewer parameters than existing models, including ResNet (He et al., 2016), DenseNet (Huang et al., 2017), Inception (Szegedy et al., 2017), and NASNet (Zoph et al., 2018).

## 6. Discussion

To disentangle the contribution of our proposed scaling method from the EfficientNet architecture, Figure 8 compares the ImageNet performance of different scaling methods for the same EfficientNet-B0 baseline network. In general, all scaling methods improve accuracy with the cost of more FLOPS, but our compound scaling method can further improve accuracy, by up to 2.5%, than other singledimension scaling methods, suggesting the importance of our proposed compound scaling.

![](images/7ce5426824b9745c20d040ea88461307aec953b81687aeb051390803377f472c.jpg)  
Figure 8. Scaling Up EfficientNet-B0 with Different Methods.

Table 7. Scaled Models Used in Figure 7.

<table><tr><td>Model</td><td>FLOPS</td><td>Top-1 Acc.</td></tr><tr><td>Baseline model (EfficientNet-B0)</td><td>0.4B</td><td>77.3%</td></tr><tr><td>Scale model by depth (d=4)</td><td>1.8B</td><td>79.0%</td></tr><tr><td>Scale model by width (w=2)</td><td>1.8B</td><td>78.9%</td></tr><tr><td>Scale model by resolution (r=2)</td><td>1.9B</td><td>79.1%</td></tr><tr><td>Compound Scale (d=1.4, w=1.2, r=1.3)</td><td>1.8B</td><td>81.1%</td></tr></table>

In order to further understand why our compound scaling method is better than others, Figure 7 compares the class activation map (Zhou et al., 2016) for a few representative models with different scaling methods. All these models are scaled from the same baseline, and their statistics are shown in Table 7. Images are randomly picked from ImageNet validation set. As shown in the figure, the model with compound scaling tends to focus on more relevant regions with more object details, while other models are either lack of object details or unable to capture all objects in the images.

## 7. Conclusion

In this paper, we systematically study ConvNet scaling and identify that carefully balancing network width, depth, and resolution is an important but missing piece, preventing us from better accuracy and efficiency. To address this issue, we propose a simple and highly effective compound scaling method, which enables us to easily scale up a baseline ConvNet to any target resource constraints in a more principled way, while maintaining model efficiency. Powered by this compound scaling method, we demonstrate that a mobilesize EfficientNet model can be scaled up very effectively, surpassing state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS, on both ImageNet and five commonly used transfer learning datasets.

## Acknowledgements

We thank Ruoming Pang, Vijay Vasudevan, Alok Aggarwal, Barret Zoph, Hongkun Yu, Jonathon Shlens, Raphael Gontijo Lopes, Yifeng Lu, Daiyi Peng, Xiaodan Song, Samy Bengio, Jeff Dean, and the Google Brain team for their help.

## Appendix

Since 2017, most research papers only report and compare ImageNet validation accuracy; this paper also follows this convention for better comparison. In addition, we have also verified the test accuracy by submitting our predictions on the 100k test set images to http://image-net.org; results are in Table 8. As expected, the test accuracy is very close to the validation accuracy.

Table 8. ImageNet Validation vs. Test Top-1/5 Accuracy.

<table><tr><td></td><td>B0</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>B5</td><td>B6</td><td>B7</td></tr><tr><td>Val top1</td><td>77.11</td><td>79.13</td><td>80.07</td><td>81.59</td><td>82.89</td><td>83.60</td><td>83.95</td><td>84.26</td></tr><tr><td>Test top1</td><td>77.23</td><td>79.17</td><td>80.16</td><td>81.72</td><td>82.94</td><td>83.69</td><td>84.04</td><td>84.33</td></tr><tr><td>Val top5</td><td>93.35</td><td>94.47</td><td>94.90</td><td>95.67</td><td>96.37</td><td>96.71</td><td>96.76</td><td>96.97</td></tr><tr><td>Test top5</td><td>93.45</td><td>94.43</td><td>94.98</td><td>95.70</td><td>96.27</td><td>96.64</td><td>96.86</td><td>96.94</td></tr></table>

## References

Berg, T., Liu, J., Woo Lee, S., Alexander, M. L., Jacobs, D. W., and Belhumeur, P. N. Birdsnap: Large-scale fine-grained visual categorization of birds. CVPR, pp. 2011–2018, 2014.

Bossard, L., Guillaumin, M., and Van Gool, L. Food-101– mining discriminative components with random forests. ECCV, pp. 446–461, 2014.

Cai, H., Zhu, L., and Han, S. Proxylessnas: Direct neural architecture search on target task and hardware. ICLR, 2019.

Chollet, F. Xception: Deep learning with depthwise separable convolutions. CVPR, pp. 1610–02357, 2017.

Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., and Le, Q. V. Autoaugment: Learning augmentation policies from data. CVPR, 2019.

Elfwing, S., Uchibe, E., and Doya, K. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018.

Gholami, A., Kwon, K., Wu, B., Tai, Z., Yue, X., Jin, P., Zhao, S., and Keutzer, K. Squeezenext: Hardware-aware neural network design. ECV Workshop at CVPR’18, 2018.

Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. ICLR, 2016.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. CVPR, pp. 770–778, 2016.

He, K., Gkioxari, G., Dollar, P., and Girshick, R. Mask ´ r-cnn. ICCV, pp. 2980–2988, 2017.

He, Y., Lin, J., Liu, Z., Wang, H., Li, L.-J., and Han, S. Amc: Automl for model compression and acceleration on mobile devices. ECCV, 2018.

Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Hu, J., Shen, L., and Sun, G. Squeeze-and-excitation networks. CVPR, 2018.

Huang, G., Sun, Y., Liu, Z., Sedra, D., and Weinberger, K. Q. Deep networks with stochastic depth. ECCV, pp. 646–661, 2016.

Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. CVPR, 2017.

Huang, Y., Cheng, Y., Chen, D., Lee, H., Ngiam, J., Le, Q. V., and Chen, Z. Gpipe: Efficient training of giant neural networks using pipeline parallelism. arXiv preprint arXiv:1808.07233, 2018.

Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., and Keutzer, K. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <0.5 mb model size. arXiv preprint arXiv:1602.07360, 2016.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML, pp. 448–456, 2015.

Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet models transfer better? CVPR, 2019.

Krause, J., Deng, J., Stark, M., and Fei-Fei, L. Collecting a large-scale dataset of fine-grained cars. Second Workshop on Fine-Grained Visual Categorizatio, 2013.

Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. Technical Report, 2009.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In NIPS, pp. 1097–1105, 2012.

Lin, H. and Jegelka, S. Resnet with one-neuron hidden layers is a universal approximator. NeurIPS, pp. 6172– 6181, 2018.

Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B.,´ and Belongie, S. Feature pyramid networks for object detection. CVPR, 2017.

Liu, C., Zoph, B., Shlens, J., Hua, W., Li, L.-J., Fei-Fei, L., Yuille, A., Huang, J., and Murphy, K. Progressive neural architecture search. ECCV, 2018.

Lu, Z., Pu, H., Wang, F., Hu, Z., and Wang, L. The expressive power of neural networks: A view from the width. NeurIPS, 2018.

Ma, N., Zhang, X., Zheng, H.-T., and Sun, J. Shufflenet v2: Practical guidelines for efficient cnn architecture design. ECCV, 2018.

Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri, M., Li, Y., Bharambe, A., and van der Maaten, L. Exploring the limits of weakly supervised pretraining. arXiv preprint arXiv:1805.00932, 2018.

Maji, S., Rahtu, E., Kannala, J., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

Ngiam, J., Peng, D., Vasudevan, V., Kornblith, S., Le, Q. V., and Pang, R. Domain adaptive transfer learning with specialist models. arXiv preprint arXiv:1811.07056, 2018.

Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. ICVGIP, pp. 722–729, 2008.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. Cats and dogs. CVPR, pp. 3498–3505, 2012.

Raghu, M., Poole, B., Kleinberg, J., Ganguli, S., and Sohl-Dickstein, J. On the expressive power of deep neural networks. ICML, 2017.

Ramachandran, P., Zoph, B., and Le, Q. V. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2018.

Real, E., Aggarwal, A., Huang, Y., and Le, Q. V. Regularized evolution for image classifier architecture search. AAAI, 2019.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3): 211–252, 2015.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. CVPR, 2018.

Sharir, O. and Shashua, A. On the expressive power of overlapping architectures of deep learning. ICLR, 2018.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. CVPR, pp. 1–9, 2015.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. CVPR, pp. 2818–2826, 2016.

Szegedy, C., Ioffe, S., Vanhoucke, V., and Alemi, A. A. Inception-v4, inception-resnet and the impact of residual connections on learning. AAAI, 4:12, 2017.

Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., and Le, Q. V. MnasNet: Platform-aware neural architecture search for mobile. CVPR, 2019.

Xie, S., Girshick, R., Dollar, P., Tu, Z., and He, K. Aggre-´ gated residual transformations for deep neural networks. CVPR, pp. 5987–5995, 2017.

Yang, T.-J., Howard, A., Chen, B., Zhang, X., Go, A., Sze, V., and Adam, H. Netadapt: Platform-aware neural network adaptation for mobile applications. ECCV, 2018.

Zagoruyko, S. and Komodakis, N. Wide residual networks. BMVC, 2016.

Zhang, X., Li, Z., Loy, C. C., and Lin, D. Polynet: A pursuit of structural diversity in very deep networks. CVPR, pp. 3900–3908, 2017.

Zhang, X., Zhou, X., Lin, M., and Sun, J. Shufflenet: An extremely efficient convolutional neural network for mobile devices. CVPR, 2018.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A. Learning deep features for discriminative localization. CVPR, pp. 2921–2929, 2016.

Zoph, B. and Le, Q. V. Neural architecture search with reinforcement learning. ICLR, 2017.

Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning transferable architectures for scalable image recognition. CVPR, 2018.