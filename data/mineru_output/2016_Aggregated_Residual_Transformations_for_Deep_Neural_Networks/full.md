# Aggregated Residual Transformations for Deep Neural Networks

Saining Xie<sup>1</sup> Ross Girshick<sup>2</sup> Piotr Dollar´ <sup>2</sup> Zhuowen Tu<sup>1</sup> Kaiming He<sup>2</sup> <sup>1</sup>UC San Diego <sup>2</sup>Facebook AI Research {s9xie,ztu}@ucsd.edu {rbg,pdollar,kaiminghe}@fb.com

## Abstract

We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online<sup>1</sup>.

## 1. Introduction

Research on visual recognition is undergoing a transition from “feature engineering” to “network engineering” [25, 24, 44, 34, 36, 38, 14]. In contrast to traditional handdesigned features (e.g., SIFT [29] and HOG [5]), features learned by neural networks from large-scale data [33] require minimal human involvement during training, and can be transferred to a variety of recognition tasks [7, 10, 28]. Nevertheless, human effort has been shifted to designing better network architectures for learning representations.

Designing architectures becomes increasingly difficult with the growing number of hyper-parameters (width<sup>2</sup>, filter sizes, strides, etc.), especially when there are many layers. The VGG-nets [36] exhibit a simple yet effective strategy of constructing very deep networks: stacking building blocks of the same shape. This strategy is inherited by ResNets [14] which stack modules of the same topology. This simple rule reduces the free choices of hyperparameters, and depth is exposed as an essential dimension in neural networks. Moreover, we argue that the simplicity of this rule may reduce the risk of over-adapting the hyperparameters to a specific dataset. The robustness of VGGnets and ResNets has been proven by various visual recognition tasks [7, 10, 9, 28, 31, 14] and by non-visual tasks involving speech [42, 30] and language [4, 41, 20].

![](images/55a2c1c5e4c7cec29fc5d2995fe7ea179525f65cc5c7fa57c94c99bc3d6b5a75.jpg)  
Figure 1. Left: A block of ResNet [14]. Right: A block of ResNeXt with cardinality <sup>=</sup> <sup>32</sup>, with roughly the same complexity. A layer is shown as (# in channels, filter size, # out channels).

Unlike VGG-nets, the family of Inception models [38, 17, 39, 37] have demonstrated that carefully designed topologies are able to achieve compelling accuracy with low theoretical complexity. The Inception models have evolved over time [38, 39], but an important common property is a split-transform-merge strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by 1×1 convolutions), transformed by a set of specialized filters (3×3, 5×5, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (e.g., 5×5) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity.

Despite good accuracy, the realization of Inception models has been accompanied with a series of complicating factors — the filter numbers and sizes are tailored for each individual transformation, and the modules are customized stage-by-stage. Although careful combinations of these components yield excellent neural network recipes, it is in general unclear how to adapt the Inception architectures to new datasets/tasks, especially when there are many factors and hyper-parameters to be designed.

In this paper, we present a simple architecture which adopts VGG/ResNets’ strategy of repeating layers, while exploiting the split-transform-merge strategy in an easy, extensible way. A module in our network performs a set of transformations, each on a low-dimensional embedding, whose outputs are aggregated by summation. We pursuit a simple realization of this idea — the transformations to be aggregated are all of the same topology (e.g., Fig. 1 (right)). This design allows us to extend to any large number of transformations without specialized designs.

Interestingly, under this simplified situation we show that our model has two other equivalent forms (Fig. 3). The reformulation in Fig. 3(b) appears similar to the Inception-ResNet module [37] in that it concatenates multiple paths; but our module differs from all existing Inception modules in that all our paths share the same topology and thus the number of paths can be easily isolated as a factor to be investigated. In a more succinct reformulation, our module can be reshaped by Krizhevsky et al.’s grouped convolutions [24] (Fig. 3(c)), which, however, had been developed as an engineering compromise.

We empirically demonstrate that our aggregated transformations outperform the original ResNet module, even under the restricted condition of maintaining computational complexity and model size — e.g., Fig. 1(right) is designed to keep the FLOPs complexity and number of parameters of Fig. 1(left). We emphasize that while it is relatively easy to increase accuracy by increasing capacity (going deeper or wider), methods that increase accuracy while maintaining (or reducing) complexity are rare in the literature.

Our method indicates that cardinality (the size of the set of transformations) is a concrete, measurable dimension that is of central importance, in addition to the dimensions of width and depth. Experiments demonstrate that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider, especially when depth and width starts to give diminishing returns for existing models.

Our neural networks, named ResNeXt (suggesting the next dimension), outperform ResNet-101/152 [14], ResNet-200 [15], Inception-v3 [39], and Inception-ResNet-v2 [37] on the ImageNet classification dataset. In particular, a 101-layer ResNeXt is able to achieve better accuracy than ResNet-200 [15] but has only 50% complexity. Moreover, ResNeXt exhibits considerably simpler designs than all Inception models. ResNeXt was the foundation of our submission to the ILSVRC 2016 classification task, in which we secured second place. This paper further evaluates ResNeXt on a larger ImageNet-5K set and the COCO object detection dataset [27], showing consistently better accuracy than its ResNet counterparts. We expect that ResNeXt will also generalize well to other visual (and non-visual) recognition tasks.

## 2. Related Work

Multi-branch convolutional networks. The Inception models [38, 17, 39, 37] are successful multi-branch architectures where each branch is carefully customized. ResNets [14] can be thought of as two-branch networks where one branch is the identity mapping. Deep neural decision forests [22] are tree-patterned multi-branch networks with learned splitting functions.

Grouped convolutions. The use of grouped convolutions dates back to the AlexNet paper [24], if not earlier. The motivation given by Krizhevsky et al. [24] is for distributing the model over two GPUs. Grouped convolutions are supported by Caffe [19], Torch [3], and other libraries, mainly for compatibility of AlexNet. To the best of our knowledge, there has been little evidence on exploiting grouped convolutions to improve accuracy. A special case of grouped convolutions is channel-wise convolutions in which the number of groups is equal to the number of channels. Channel-wise convolutions are part of the separable convolutions in [35].

Compressing convolutional networks. Decomposition (at spatial [6, 18] and/or channel [6, 21, 16] level) is a widely adopted technique to reduce redundancy of deep convolutional networks and accelerate/compress them. Ioannou et al. [16] present a “root”-patterned network for reducing computation, and branches in the root are realized by grouped convolutions. These methods [6, 18, 21, 16] have shown elegant compromise of accuracy with lower complexity and smaller model sizes. Instead of compression, our method is an architecture that empirically shows stronger representational power.

Ensembling. Averaging a set of independently trained networks is an effective solution to improving accuracy [24], widely adopted in recognition competitions [33]. Veit et al. [40] interpret a single ResNet as an ensemble of shallower networks, which results from ResNet’s additive behaviors [15]. Our method harnesses additions to aggregate a set of transformations. But we argue that it is imprecise to view our method as ensembling, because the members to be aggregated are trained jointly, not independently.

## 3. Method

## 3.1. Template

We adopt a highly modularized design following VGG/ResNets. Our network consists of a stack of residual blocks. These blocks have the same topology, and are subject to two simple rules inspired by VGG/ResNets: (i) if producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes), and (ii) each time when the spatial map is downsampled by a factor of 2, the width of the blocks is multiplied by a factor of 2. The second rule ensures that the computational complexity, in terms of FLOPs (floating-point operations, in # of multiply-adds), is roughly the same for all blocks.

<table><tr><td>stage</td><td>output</td><td>ResNet-50</td><td>ResNeXt-50 (32×4d)</td></tr><tr><td>conv1</td><td>112×112</td><td>7×7, 64, stride 2</td><td>7×7, 64, stride 2</td></tr><tr><td rowspan="2">conv2</td><td rowspan="2">56×56</td><td>3×3 max pool, stride 2</td><td>3×3 max pool, stride 2</td></tr><tr><td> $\begin{bmatrix} 1 \times 1, 64 \\ 3 \times 3, 64 \\ 1 \times 1, 256 \end{bmatrix} \times 3$ </td><td> $\begin{bmatrix} 1 \times 1, 128 \\ 3 \times 3, 128, C=32 \\ 1 \times 1, 256 \end{bmatrix} \times 3$ </td></tr><tr><td>conv3</td><td>28×28</td><td> $\begin{bmatrix} 1 \times 1, 128 \\ 3 \times 3, 128 \\ 1 \times 1, 512 \end{bmatrix} \times 4$ </td><td> $\begin{bmatrix} 1 \times 1, 256 \\ 3 \times 3, 256, C=32 \\ 1 \times 1, 512 \end{bmatrix} \times 4$ </td></tr><tr><td>conv4</td><td>14×14</td><td> $\begin{bmatrix} 1 \times 1, 256 \\ 3 \times 3, 256 \\ 1 \times 1, 1024 \end{bmatrix} \times 6$ </td><td> $\begin{bmatrix} 1 \times 1, 512 \\ 3 \times 3, 512, C=32 \\ 1 \times 1, 1024 \end{bmatrix} \times 6$ </td></tr><tr><td>conv5</td><td>7×7</td><td> $\begin{bmatrix} 1 \times 1, 512 \\ 3 \times 3, 512 \\ 1 \times 1, 2048 \end{bmatrix} \times 3$ </td><td> $\begin{bmatrix} 1 \times 1, 1024 \\ 3 \times 3, 1024, C=32 \\ 1 \times 1, 2048 \end{bmatrix} \times 3$ </td></tr><tr><td></td><td>1×1</td><td>global average pool1000-d fc, softmax</td><td>global average pool1000-d fc, softmax</td></tr><tr><td colspan="2"># params.</td><td> $\mathbf{25.5\times 10^{6}}$ </td><td> $\mathbf{25.0\times 10^{6}}$ </td></tr><tr><td colspan="2">FLOPs</td><td> $\mathbf{4.1\times 10^{9}}$ </td><td> $\mathbf{4.2\times 10^{9}}$ </td></tr></table>

Table 1. (Left) ResNet-50. (Right) ResNeXt-50 with a 32×4d template (using the reformulation in Fig. 3(c)). Inside the brackets are the shape of a residual block, and outside the brackets is the number of stacked blocks on a stage. ${ } ^ {  } C = 3 2 ^ { \circ }$ suggests grouped convolutions [24] with 32 groups. The numbers of parameters and FLOPs are similar between these two models.

With these two rules, we only need to design a template module, and all modules in a network can be determined accordingly. So these two rules greatly narrow down the design space and allow us to focus on a few key factors. The networks constructed by these rules are in Table 1.

## 3.2. Revisiting Simple Neurons

The simplest neurons in artificial neural networks perform inner product (weighted sum), which is the elementary transformation done by fully-connected and convolutional layers. Inner product can be thought of as a form of aggregating transformation:

$$
\sum_ {i = 1} ^ {D} w _ {i} x _ {i},\tag{1}
$$

where $\mathbf { x } = [ x _ { 1 } , x _ { 2 } , . . . , x _ { D } ]$ is a D-channel input vector to the neuron and $w _ { i }$ is a filter’s weight for the i-th channel. This operation (usually including some output nonlinearity) is referred to as a “neuron”. See Fig. 2.

![](images/44de029ad0a59742284f4591c1f748dd79929fcbd571a16259145aab2be5b02f.jpg)  
Figure 2. A simple neuron that performs inner product.

The above operation can be recast as a combination of splitting, transforming, and aggregating. (i) Splitting: the vector x is sliced as a low-dimensional embedding, and in the above, it is a single-dimension subspace $x _ { i }$ . (ii) Transforming: the low-dimensional representation is transformed, and in the above, it is simply scaled: $w _ { i } x _ { i }$ . (iii) Aggregating: the transformations in all embeddings are aggregated by $\Sigma _ { i = 1 } ^ { D }$

## 3.3. Aggregated Transformations

Given the above analysis of a simple neuron, we consider replacing the elementary transformation $( w _ { i } x _ { i } )$ with a more generic function, which in itself can also be a network. In contrast to “Network-in-Network” [26] that turns out to increase the dimension of depth, we show that our “Network-in-Neuron” expands along a new dimension.

Formally, we present aggregated transformations as:

$$
\mathcal {F} (\mathbf {x}) = \sum_ {i = 1} ^ {C} \mathcal {T} _ {i} (\mathbf {x}),\tag{2}
$$

where $\mathcal { T } _ { i } ( \mathbf { x } )$ can be an arbitrary function. Analogous to a simple neuron, $\mathcal { T } _ { i }$ should project x into an (optionally lowdimensional) embedding and then transform it.

In Eqn.(2), C is the size of the set of transformations to be aggregated. We refer to C as cardinality [2]. In Eqn.(2) $C$ is in a position similar to D in Eqn.(1), but $C$ need not equal D and can be an arbitrary number. While the dimension of width is related to the number of simple transformations (inner product), we argue that the dimension of cardinality controls the number of more complex transformations. We show by experiments that cardinality is an essential dimension and can be more effective than the dimensions of width and depth.

In this paper, we consider a simple way of designing the transformation functions: all $\mathcal { T } _ { i } ^ { \cdot } \mathrm { s }$ have the same topology. This extends the VGG-style strategy of repeating layers of the same shape, which is helpful for isolating a few factors and extending to any large number of transformations. We set the individual transformation $\mathcal { T } _ { i }$ to be the bottleneckshaped architecture [14], as illustrated in Fig. 1 (right). In this case, the first 1×1 layer in each $\mathcal { T } _ { i }$ produces the lowdimensional embedding.

![](images/514e6a5e8a42d8b0e33c4dc68bd3de5a880534d12869a41c021d071b6e950bef.jpg)  
Figure 3. Equivalent building blocks of ResNeXt. (a): Aggregated residual transformations, the same as Fig. 1 right. (b): A block equivalent to (a), implemented as early concatenation. (c): A block equivalent to (a,b), implemented as grouped convolutions [24]. Notations in bold text highlight the reformulation changes. A layer is denoted as (# input channels, filter size, # output channels).

The aggregated transformation in Eqn.(2) serves as the residual function [14] (Fig. 1 right):

$$
\mathbf {y} = \mathbf {x} + \sum_ {i = 1} ^ {C} \mathcal {T} _ {i} (\mathbf {x}),\tag{3}
$$

where y is the output.

Relation to Inception-ResNet. Some tensor manipulations show that the module in Fig. 1(right) (also shown in Fig. 3(a)) is equivalent to Fig. 3(b).<sup>3</sup> Fig. 3(b) appears similar to the Inception-ResNet [37] block in that it involves branching and concatenating in the residual function. But unlike all Inception or Inception-ResNet modules, we share the same topology among the multiple paths. Our module requires minimal extra effort designing each path.

Relation to Grouped Convolutions. The above module becomes more succinct using the notation of grouped convolutions [24].<sup>4</sup> This reformulation is illustrated in Fig. 3(c). All the low-dimensional embeddings (the first 1×1 layers) can be replaced by a single, wider layer (e.g., 1×1, 128-d in Fig 3(c)). Splitting is essentially done by the grouped convolutional layer when it divides its input channels into groups. The grouped convolutional layer in Fig. 3(c) performs 32 groups of convolutions whose input and output channels are 4-dimensional. The grouped convolutional layer concatenates them as the outputs of the layer. The block in Fig. 3(c) looks like the original bottleneck residual block in Fig. 1(left), except that Fig. 3(c) is a wider but sparsely connected module.

![](images/dd58786d87e95fec6a0f105efe494ea41fe5d5f5f4956f70c12d472a7daf9e69.jpg)  
Figure 4. (Left): Aggregating transformations of depth <sup>=</sup> <sup>2</sup>. (Right): An equivalent block, which is trivially wider.

We note that the reformulations produce nontrivial topologies only when the block has depth ≥3. If the block has depth = 2 (e.g., the basic block in [14]), the reformulations lead to trivially a wide, dense module. See the illustration in Fig. 4.

Discussion. We note that although we present reformulations that exhibit concatenation (Fig. 3(b)) or grouped convolutions (Fig. 3(c)), such reformulations are not always applicable for the general form of Eqn.(3), e.g., if the transformation Ti takes arbitrary forms and are heterogenous. We choose to use homogenous forms in this paper because they are simpler and extensible. Under this simplified case, grouped convolutions in the form of Fig. 3(c) are helpful for easing implementation.

## 3.4. Model Capacity

Our experiments in the next section will show that our models improve accuracy when maintaining the model complexity and number of parameters. This is not only interesting in practice, but more importantly, the complexity and number of parameters represent inherent capacity of models and thus are often investigated as fundamental properties of deep networks [8].

When we evaluate different cardinalities C while preserving complexity, we want to minimize the modification of other hyper-parameters. We choose to adjust the width of the bottleneck (e.g., 4-d in Fig 1(right)), because it can be isolated from the input and output of the block. This strategy introduces no change to other hyper-parameters (depth or input/output width of blocks), so is helpful for us to focus on the impact of cardinality.

<table><tr><td>cardinality C</td><td>1</td><td>2</td><td>4</td><td>8</td><td>32</td></tr><tr><td>width of bottleneck d</td><td>64</td><td>40</td><td>24</td><td>14</td><td>4</td></tr><tr><td>width of group conv.</td><td>64</td><td>80</td><td>96</td><td>112</td><td>128</td></tr></table>

Table 2. Relations between cardinality and width (for the template of conv2), with roughly preserved complexity on a residual block. The number of parameters is ∼70k for the template of conv2. The number of FLOPs is ∼0.22 billion (# params×56×56 for conv2).

In Fig. 1(left), the original ResNet bottleneck block [14] has 256 · 64 + 3 · 3 · 64 · 64 + 64 · 256 ≈ 70k parameters and proportional FLOPs (on the same feature map size). With bottleneck width d, our template in Fig. 1(right) has:

$$
C \cdot (2 5 6 \cdot d + 3 \cdot 3 \cdot d \cdot d + d \cdot 2 5 6)\tag{4}
$$

parameters and proportional FLOPs. When C = 32 and d = 4, Eqn.(4) ≈ 70k. Table 2 shows the relationship between cardinality C and bottleneck width d.

Because we adopt the two rules in Sec. 3.1, the above approximate equality is valid between a ResNet bottleneck block and our ResNeXt on all stages (except for the subsampling layers where the feature maps size changes). Table 1 compares the original ResNet-50 and our ResNeXt-50 that is of similar capacity.<sup>5</sup> We note that the complexity can only be preserved approximately, but the difference of the complexity is minor and does not bias our results.

## 4. Implementation details

Our implementation follows [14] and the publicly available code of fb.resnet.torch [11]. On the ImageNet dataset, the input image is 224×224 randomly cropped from a resized image using the scale and aspect ratio augmentation of [38] implemented by [11]. The shortcuts are identity connections except for those increasing dimensions which are projections (type B in [14]). Downsampling of conv3, 4, and 5 is done by stride-2 convolutions in the 3×3 layer of the first block in each stage, as suggested in [11]. We use SGD with a mini-batch size of 256 on 8 GPUs (32 per GPU). The weight decay is 0.0001 and the momentum is 0.9. We start from a learning rate of 0.1, and divide it by 10 for three times using the schedule in [11]. We adopt the weight initialization of [13]. In all ablation comparisons, we evaluate the error on the single 224×224 center crop from an image whose shorter side is 256.

Our models are realized by the form of Fig. 3(c). We perform batch normalization (BN) [17] right after the convolutions in Fig. 3(c).<sup>6</sup> ReLU is performed right after each BN, expect for the output of the block where ReLU is performed after the adding to the shortcut, following [14].

We note that the three forms in Fig. 3 are strictly equivalent, when BN and ReLU are appropriately addressed as mentioned above. We have trained all three forms and obtained the same results. We choose to implement by Fig. 3(c) because it is more succinct and faster than the other two forms.

## 5. Experiments

## 5.1. Experiments on ImageNet-1K

We conduct ablation experiments on the 1000-class ImageNet classification task [33]. We follow [14] to construct 50-layer and 101-layer residual networks. We simply replace all blocks in ResNet-50/101 with our blocks.

Notations. Because we adopt the two rules in Sec. 3.1, it is sufficient for us to refer to an architecture by the template. For example, Table 1 shows a ResNeXt-50 constructed by a template with cardinality = 32 and bottleneck width = 4d (Fig. 3). This network is denoted as ResNeXt-50 (32×4d) for simplicity. We note that the input/output width of the template is fixed as 256-d (Fig. 3), and all widths are doubled each time when the feature map is subsampled (see Table 1).

Cardinality <sup>vs</sup>. Width. We first evaluate the trade-off between cardinality C and bottleneck width, under preserved complexity as listed in Table 2. Table 3 shows the results and Fig. 5 shows the curves of error vs. epochs. Comparing with ResNet-50 (Table 3 top and Fig. 5 left), the 32×4d ResNeXt-50 has a validation error of 22.2%, which is 1.7% lower than the ResNet baseline’s 23.9%. With cardinality C increasing from 1 to 32 while keeping complexity, the error rate keeps reducing. Furthermore, the 32×4d ResNeXt also has a much lower training error than the ResNet counterpart, suggesting that the gains are not from regularization but from stronger representations.

Similar trends are observed in the case of ResNet-101 (Fig. 5 right, Table 3 bottom), where the 32×4d ResNeXt-101 outperforms the ResNet-101 counterpart by 0.8%. Although this improvement of validation error is smaller than that of the 50-layer case, the improvement of training error is still big (20% for ResNet-101 and 16% for 32×4d ResNeXt-101, Fig. 5 right). In fact, more training data will enlarge the gap of validation error, as we show on an ImageNet-5K set in the next subsection.

Table 3 also suggests that with complexity preserved, increasing cardinality at the price of reducing width starts to show saturating accuracy when the bottleneck width is small. We argue that it is not worthwhile to keep reducing width in such a trade-off. So we adopt a bottleneck width no smaller than 4d in the following.

![](images/08602c2502d492f04bea659c2deac72ad09b355e6ac20b3f8530255ea1266a0d.jpg)

![](images/925cc4e7a13403aa231e754fdb2186555fc0a83d34b53603c79e42eba1c78124.jpg)  
Figure 5. Training curves on ImageNet-1K. (Left): ResNet/ResNeXt-50 with preserved complexity (∼4.1 billion $\mathrm { F L O P s } , \sim 2 5$ million parameters); (Right): ResNet/ResNeXt-101 with preserved complexity (∼7.8 billion FLOPs, ∼44 million parameters).

<table><tr><td></td><td>setting</td><td>top-1 error (%)</td></tr><tr><td>ResNet-50</td><td> $1 \times 64d$ </td><td>23.9</td></tr><tr><td>ResNeXt-50</td><td> $2 \times 40d$ </td><td>23.0</td></tr><tr><td>ResNeXt-50</td><td> $4 \times 24d$ </td><td>22.6</td></tr><tr><td>ResNeXt-50</td><td> $8 \times 14d$ </td><td>22.3</td></tr><tr><td>ResNeXt-50</td><td> $32 \times 4d$ </td><td>22.2</td></tr><tr><td>ResNet-101</td><td> $1 \times 64d$ </td><td>22.0</td></tr><tr><td>ResNeXt-101</td><td> $2 \times 40d$ </td><td>21.7</td></tr><tr><td>ResNeXt-101</td><td> $4 \times 24d$ </td><td>21.4</td></tr><tr><td>ResNeXt-101</td><td> $8 \times 14d$ </td><td>21.3</td></tr><tr><td>ResNeXt-101</td><td> $32 \times 4d$ </td><td>21.2</td></tr></table>

Table 3. Ablation experiments on ImageNet-1K. (Top): ResNet-50 with preserved complexity (∼4.1 billion FLOPs); (Bottom): ResNet-101 with preserved complexity (∼7.8 billion FLOPs). The error rate is evaluated on the single crop of 224×224 pixels.

Increasing Cardinality <sup>vs</sup>. Deeper/Wider. Next we investigate increasing complexity by increasing cardinality C or increasing depth or width. The following comparison can also be viewed as with reference to 2× FLOPs of the ResNet-101 baseline. We compare the following variants that have ∼15 billion FLOPs. (i) Going deeper to 200 layers. We adopt the ResNet-200 [15] implemented in [11]. (ii) Going wider by increasing the bottleneck width. (iii) Increasing cardinality by doubling C.

Table 4 shows that increasing complexity by 2× consistently reduces error vs. the ResNet-101 baseline (22.0%). But the improvement is small when going deeper (ResNet-200, by 0.3%) or wider (wider ResNet-101, by 0.7%).

On the contrary, increasing cardinality C shows much better results than going deeper or wider. The 2×64d ResNeXt-101 (i.e., doubling C on 1×64d ResNet-101 baseline and keeping the width) reduces the top-1 error by 1.3% to 20.7%. The 64×4d ResNeXt-101 $( i . e . ,$ doubling C on 32×4d ResNeXt-101 and keeping the width) reduces the top-1 error to 20.4%.

<table><tr><td></td><td>setting</td><td>top-1 err (%)</td><td>top-5 err (%)</td></tr><tr><td colspan="4">1× complexity references:</td></tr><tr><td>ResNet-101</td><td>1 × 64d</td><td>22.0</td><td>6.0</td></tr><tr><td>ResNeXt-101</td><td>32 × 4d</td><td>21.2</td><td>5.6</td></tr><tr><td colspan="4">2× complexity models follow:</td></tr><tr><td>ResNet-200 [15]</td><td>1 × 64d</td><td>21.7</td><td>5.8</td></tr><tr><td>ResNet-101, wider</td><td>1 × 100d</td><td>21.3</td><td>5.7</td></tr><tr><td>ResNeXt-101</td><td>2 × 64d</td><td>20.7</td><td>5.5</td></tr><tr><td>ResNeXt-101</td><td>64 × 4d</td><td>20.4</td><td>5.3</td></tr></table>

Table 4. Comparisons on ImageNet-1K when the number of FLOPs is increased to 2× of ResNet-101’s. The error rate is evaluated on the single crop of 224×224 pixels. The highlighted factors are the factors that increase complexity.

We also note that 32×4d ResNet-101 (21.2%) performs better than the deeper ResNet-200 and the wider ResNet-101, even though it has only ∼50% complexity. This again shows that cardinality is a more effective dimension than the dimensions of depth and width.

Residual connections. The following table shows the effects of the residual (shortcut) connections:

<table><tr><td></td><td>setting</td><td>w/ residual</td><td>w/o residual</td></tr><tr><td>ResNet-50</td><td>1 × 64d</td><td>23.9</td><td>31.2</td></tr><tr><td>ResNeXt-50</td><td>32 × 4d</td><td>22.2</td><td>26.1</td></tr></table>

Removing shortcuts from the ResNeXt-50 increases the error by 3.9 points to 26.1%. Removing shortcuts from its

ResNet-50 counterpart is much worse (31.2%). These comparisons suggest that the residual connections are helpful for optimization, whereas aggregated transformations are stronger representations, as shown by the fact that they perform consistently better than their counterparts with or without residual connections.

Performance. For simplicity we use Torch’s built-in grouped convolution implementation, without special optimization. We note that this implementation was brute-force and not parallelization-friendly. On 8 GPUs of NVIDIA M40, training 32×4d ResNeXt-101 in Table 3 takes 0.95s per mini-batch, vs. 0.70s of ResNet-101 baseline that has similar FLOPs. We argue that this is a reasonable overhead. We expect carefully engineered lower-level implementation (e.g., in CUDA) will reduce this overhead. We also expect that the inference time on CPUs will present less overhead. Training the 2×complexity model (64×4d ResNeXt-101) takes 1.7s per mini-batch and 10 days total on 8 GPUs.

Comparisons with state-of-the-art results. Table 5 shows more results of single-crop testing on the ImageNet validation set. In addition to testing a 224×224 crop, we also evaluate a 320×320 crop following [15]. Our results compare favorably with ResNet, Inception-v3/v4, and Inception-ResNet-v2, achieving a single-crop top-5 error rate of 4.4%. In addition, our architecture design is much simpler than all Inception models, and requires considerably fewer hyper-parameters to be set by hand.

ResNeXt is the foundation of our entries to the ILSVRC 2016 classification task, in which we achieved 2<sup>nd</sup> place. We note that many models (including ours) start to get saturated on this dataset after using multi-scale and/or multicrop testing. We had a single-model top-1/top-5 error rates of 17.7%/3.7% using the multi-scale dense testing in [14], on par with Inception-ResNet-v2’s single-model results of 17.8%/3.7% that adopts multi-scale, multi-crop testing. We had an ensemble result of 3.03% top-5 error on the test set, on par with the winner’s 2.99% and Inception-v4/Inception-ResNet-v2’s 3.08% [37].

<table><tr><td rowspan="2"></td><td colspan="2">224×224</td><td colspan="2">320×320 / 299×299</td></tr><tr><td>top-1 err</td><td>top-5 err</td><td>top-1 err</td><td>top-5 err</td></tr><tr><td>ResNet-101 [14]</td><td>22.0</td><td>6.0</td><td>-</td><td>-</td></tr><tr><td>ResNet-200 [15]</td><td>21.7</td><td>5.8</td><td>20.1</td><td>4.8</td></tr><tr><td>Inception-v3 [39]</td><td>-</td><td>-</td><td>21.2</td><td>5.6</td></tr><tr><td>Inception-v4 [37]</td><td>-</td><td>-</td><td>20.0</td><td>5.0</td></tr><tr><td>Inception-ResNet-v2 [37]</td><td>-</td><td>-</td><td>19.9</td><td>4.9</td></tr><tr><td>ResNeXt-101 (64 × 4d)</td><td>20.4</td><td>5.3</td><td>19.1</td><td>4.4</td></tr></table>

Table 5. State-of-the-art models on the ImageNet-1K validation set (single-crop testing). The test size of ResNet/ResNeXt is 224×224 and 320×320 as in [15] and of the Inception models is 299×299.

![](images/3f2d60d839c9d4f7bb31df2b02b0c34f340f11476e87ad23f493e989361a2de6.jpg)  
Figure 6. ImageNet-5K experiments. Models are trained on the 5K set and evaluated on the original 1K validation set, plotted as a 1K-way classification task. ResNeXt and its ResNet counterpart have similar complexity.

<table><tr><td rowspan="2"></td><td rowspan="2">setting</td><td colspan="2">5K-way classification</td><td colspan="2">1K-way classification</td></tr><tr><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td></tr><tr><td>ResNet-50</td><td>1 × 64d</td><td>45.5</td><td>19.4</td><td>27.1</td><td>8.2</td></tr><tr><td>ResNeXt-50</td><td>32 × 4d</td><td>42.3</td><td>16.8</td><td>24.4</td><td>6.6</td></tr><tr><td>ResNet-101</td><td>1 × 64d</td><td>42.4</td><td>16.9</td><td>24.2</td><td>6.8</td></tr><tr><td>ResNeXt-101</td><td>32 × 4d</td><td>40.1</td><td>15.1</td><td>22.2</td><td>5.7</td></tr></table>

Table 6. Error (%) on ImageNet-5K. The models are trained on ImageNet-5K and tested on the ImageNet-1K val set, treated as a 5K-way classification task or a 1K-way classification task at test time. ResNeXt and its ResNet counterpart have similar complexity. The error is evaluated on the single crop of 224×224 pixels.

## 5.2. Experiments on ImageNet-5K

The performance on ImageNet-1K appears to saturate. But we argue that this is not because of the capability of the models but because of the complexity of the dataset. Next we evaluate our models on a larger ImageNet subset that has 5000 categories.

Our 5K dataset is a subset of the full ImageNet-22K set [33]. The 5000 categories consist of the original ImageNet-1K categories and additional 4000 categories that have the largest number of images in the full ImageNet set. The 5K set has 6.8 million images, about 5× of the 1K set. There is no official train/val split available, so we opt to evaluate on the original ImageNet-1K validation set. On this 1K-class val set, the models can be evaluated as a 5K-way classification task (all labels predicted to be the other 4K classes are automatically erroneous) or as a 1K-way classification task (softmax is applied only on the 1K classes) at test time.

The implementation details are the same as in Sec. 4. The 5K-training models are all trained from scratch, and are trained for the same number of mini-batches as the 1Ktraining models (so 1/5× epochs). Table 6 and Fig. 6 show the comparisons under preserved complexity. ResNeXt-50 reduces the 5K-way top-1 error by 3.2% comparing with ResNet-50, and ResNetXt-101 reduces the 5K-way top-1 error by 2.3% comparing with ResNet-101. Similar gaps are observed on the 1K-way error. These demonstrate the stronger representational power of ResNeXt.

<table><tr><td></td><td># params</td><td>CIFAR-10</td><td>CIFAR-100</td></tr><tr><td>Wide ResNet [43]</td><td>36.5M</td><td>4.17</td><td>20.50</td></tr><tr><td>ResNeXt-29, 8×64d</td><td>34.4M</td><td>3.65</td><td>17.77</td></tr><tr><td>ResNeXt-29, 16×64d</td><td>68.1M</td><td>3.58</td><td>17.31</td></tr></table>

Table 7. Test error (%) and model size on CIFAR. Our results are the average of 10 runs.

Moreover, we find that the models trained on the 5K set (with 1K-way error 22.2%/5.7% in Table 6) perform competitively comparing with those trained on the 1K set (21.2%/5.6% in Table 3), evaluated on the same 1K-way classification task on the validation set. This result is achieved without increasing the training time (due to the same number of mini-batches) and without fine-tuning. We argue that this is a promising result, given that the training task of classifying 5K categories is a more challenging one.

## 5.3. Experiments on CIFAR

We conduct more experiments on CIFAR-10 and 100 datasets [23]. We use the architectures as in [14] and replace the basic residual block by the bottleneck template of $\left\lceil \begin{array} { l } { 1 \times 1 , 6 4 } \\ { 3 \times 3 , 6 4 } \\ { 1 \times 1 , 2 5 6 } \end{array} \right\rceil$ . Our networks start with a single 3×3 conv layer, followed by 3 stages each having 3 residual blocks, and end with average pooling and a fully-connected classifier (total 29-layer deep), following [14]. We adopt the same translation and flipping data augmentation as [14]. Implementation details are in the appendix.

We compare two cases of increasing complexity based on the above baseline: (i) increase cardinality and fix all widths, or (ii) increase width of the bottleneck and fix cardinality = 1. We train and evaluate a series of networks under these changes. Fig. 7 shows the comparisons of test error rates vs. model sizes. We find that increasing cardinality is more effective than increasing width, consistent to what we have observed on ImageNet-1K. Table 7 shows the results and model sizes, comparing with the Wide ResNet [43] which is the best published record. Our model with a similar model size (34.4M) shows results better than Wide ResNet. Our larger method achieves 3.58% test error (average of 10 runs) on CIFAR-10 and 17.31% on CIFAR-100. To the best of our knowledge, these are the state-of-the-art results (with similar data augmentation) in the literature including unpublished technical reports.

![](images/f15db1bf1b1c22411bf544540eaf62b35f9b7fd7a057c8ee73555d060fad1754.jpg)  
Figure 7. Test error vs. model size on CIFAR-10. The results are computed with 10 runs, shown with standard error bars. The labels show the settings of the templates.

<table><tr><td></td><td>setting</td><td>AP@0.5</td><td>AP</td></tr><tr><td>ResNet-50</td><td>1 × 64d</td><td>47.6</td><td>26.5</td></tr><tr><td>ResNeXt-50</td><td>32 × 4d</td><td>49.7</td><td>27.5</td></tr><tr><td>ResNet-101</td><td>1 × 64d</td><td>51.1</td><td>29.8</td></tr><tr><td>ResNeXt-101</td><td>32 × 4d</td><td>51.9</td><td>30.0</td></tr></table>

Table 8. Object detection results on the COCO minival set. ResNeXt and its ResNet counterpart have similar complexity.

## 5.4. Experiments on COCO object detection

Next we evaluate the generalizability on the COCO object detection set [27]. We train the models on the 80k training set plus a 35k val subset and evaluate on a 5k val subset (called minival), following [1]. We evaluate the COCOstyle Average Precision (AP) as well as AP@IoU=0.5 [27]. We adopt the basic Faster R-CNN [32] and follow [14] to plug ResNet/ResNeXt into it. The models are pre-trained on ImageNet-1K and fine-tuned on the detection set. Implementation details are in the appendix.

Table 8 shows the comparisons. On the 50-layer baseline, ResNeXt improves AP@0.5 by 2.1% and AP by 1.0%, without increasing complexity. ResNeXt shows smaller improvements on the 101-layer baseline. We conjecture that more training data will lead to a larger gap, as observed on the ImageNet-5K set.

It is also worth noting that recently ResNeXt has been adopted in Mask R-CNN [12] that achieves state-of-the-art results on COCO instance segmentation and object detection tasks.

## Acknowledgment

S.X. and Z.T.’s research was partly supported by NSF IIS-1618477. The authors would like to thank Tsung-Yi Lin and Priya Goyal for valuable discussions.

## References

[1] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Insideoutside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.

[2] G. Cantor. Uber unendliche, lineare punktmannich- <sup>¨</sup> faltigkeiten, arbeiten zur mengenlehre aus den jahren 1872-1884. 1884.

[3] R. Collobert, S. Bengio, and J. Mariethoz. Torch: a mod-´ ular machine learning software library. Technical report, Idiap, 2002.

[4] A. Conneau, H. Schwenk, L. Barrault, and Y. Lecun. Very deep convolutional networks for natural language processing. arXiv:1606.01781, 2016.

[5] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005.

[6] E. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. In NIPS, 2014.

[7] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In ICML, 2014.

[8] D. Eigen, J. Rolfe, R. Fergus, and Y. LeCun. Understanding deep architectures using a recursive convolutional network. arXiv:1312.1847, 2013.

[9] R. Girshick. Fast R-CNN. In ICCV, 2015.

[10] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[11] S. Gross and M. Wilber. Training and investigating Residual Nets. https://github.com/ facebook/fb.resnet.torch, 2016.

[12] K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask´ R-CNN. arXiv:1703.06870, 2017.

[13] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.

[14] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[15] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[16] Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi. Deep roots: Improving cnn efficiency with hierarchical filter groups. arXiv:1605.06489, 2016.

[17] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[18] M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. In BMVC, 2014.

[19] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv:1408.5093, 2014.

[20] N. Kalchbrenner, L. Espeholt, K. Simonyan, A. v. d. Oord, A. Graves, and K. Kavukcuoglu. Neural machine translation in linear time. arXiv:1610.10099, 2016.

[21] Y.-D. Kim, E. Park, S. Yoo, T. Choi, L. Yang, and D. Shin. Compression of deep convolutional neural networks for fast and low power mobile applications. In ICLR, 2016.

[22] P. Kontschieder, M. Fiterau, A. Criminisi, and S. R. Bulo.\` Deep convolutional neural decision forests. In ICCV, 2015.

[23] A. Krizhevsky. Learning multiple layers of features from tiny images. Tech Report, 2009.

[24] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

[25] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[26] M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR, 2014.

[27] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft´ COCO: Common objects in context. In ECCV. 2014.

[28] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[29] D. G. Lowe. Distinctive image features from scaleinvariant keypoints. IJCV, 2004.

[30] A. Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv:1609.03499, 2016.

[31] P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to segment object candidates. In NIPS, 2015.

[32] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[33] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

[34] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.

[35] L. Sifre and S. Mallat. Rigid-motion scattering for texture classification. arXiv:1403.1687, 2014.

[36] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[37] C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. In ICLR Workshop, 2016.

[38] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[39] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[40] A. Veit, M. Wilber, and S. Belongie. Residual networks behave like ensembles of relatively shallow network. In NIPS, 2016.

[41] Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144, 2016.

[42] W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, and G. Zweig. The Microsoft 2016 Conversational Speech Recognition System. arXiv:1609.03528, 2016.

[43] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.

[44] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.