# Supervised Contrastive Learning

Prannay Khosla Google Research Yonglong Tian <sup>†</sup> MIT

Piotr Teterwak <sup>∗</sup> <sup>†</sup> Chen Wang † Boston University Snap Inc. Phillip Isola <sup>†</sup> Aaron Maschinot MIT Google Research Dilip Krishnan Google Research

Aaron Sarna ‡ Google Research Ce Liu Google Research

## Abstract

Contrastive learning applied to self-supervised representation learning has seen a resurgence in recent years, leading to state of the art performance in the unsupervised training of deep image models. Modern batch contrastive approaches subsume or significantly outperform traditional contrastive losses such as triplet, max-margin and the N-pairs loss. In this work, we extend the self-supervised batch contrastive approach to the fully-supervised setting, allowing us to effectively leverage label information. Clusters of points belonging to the same class are pulled together in embedding space, while simultaneously pushing apart clusters of samples from different classes. We analyze two possible versions of the supervised contrastive (SupCon) loss, identifying the best-performing formulation of the loss. On ResNet-200, we achieve top-1 accuracy of 81.4% on the ImageNet dataset, which is 0.8% above the best number reported for this architecture. We show consistent outperformance over cross-entropy on other datasets and two ResNet variants. The loss shows benefits for robustness to natural corruptions, and is more stable to hyperparameter settings such as optimizers and data augmentations. Our loss function is simple to implement and reference TensorFlow code is released at https://t.ly/supcon <sup>1</sup>.

## 1 Introduction

The cross-entropy loss is the most widely used loss function for supervised learning of deep classification models. A number of works have explored shortcomings of this loss, such as lack of robustness to noisy labels [63, 46] and the possibility of poor margins [10, 31], leading to reduced generalization performance. However, in practice, most proposed alternatives have not worked better for large-scale datasets, such as ImageNet [7], as evidenced by the continued use of cross-entropy to achieve state of the art results [5, 6, 55, 25].

In recent years, a resurgence of work in contrastive learning has led to major advances in self-supervised representation learning [54, 18, 38, 48, 22, 3, 15]. The common idea in these works is the following: pull together an anchor and a “positive” sample in embedding space, and push apart the anchor from many “negative” samples. Since no labels are available, a positive pair often consists of data augmentations of the sample, and negative pairs are formed by the anchor and randomly chosen samples from the minibatch. This is depicted in Fig. 2 (left). In [38, 48], connections are made of the contrastive loss to maximization of mutual information between different views of the data.

![](images/83314114f4f4bbf97fc77db8e1bf0ccc5aab2c8541be96a6943b66742cfc240c.jpg)  
Figure 1: Our SupCon loss consistently outperforms cross-entropy with standard data augmentations. We show top-1 accuracy for the ImageNet dataset, on ResNet-50, ResNet-101 and ResNet-200, and compare against AutoAugment [5], RandAugment [6] and CutMix [59].

![](images/62965b9daa862c518c7e3c668a34611cb4278df500c855a12efa3bfb6a4efb82.jpg)  
Figure 2: Supervised vs. self-supervised contrastive losses: The self-supervised contrastive loss (left, Eq. 1) contrasts a single positive for each anchor (i.e., an augmented version of the same image) against a set of negatives consisting of the entire remainder of the batch. The supervised contrastive loss (right) considered in this paper (Eq. 2), however, contrasts the set of all samples from the same class as positives against the negatives from the remainder of the batch. As demonstrated by the photo of the black and white puppy, taking class label information into account results in an embedding space where elements of the same class are more closely aligned than in the self-supervised case.

In this work, we propose a loss for supervised learning that builds on the contrastive self-supervised literature by leveraging label information. Normalized embeddings from the same class are pulled closer together than embeddings from different classes. Our technical novelty in this work is to consider many positives per anchor in addition to many negatives (as opposed to self-supervised contrastive learning which uses only a single positive). These positives are drawn from samples of the same class as the anchor, rather than being data augmentations of the anchor, as done in self-supervised learning. While this is a simple extension to the self-supervised setup, it is nonobvious how to setup the loss function correctly, and we analyze two alternatives. Fig. 2 (right) and Fig. 1 (Supplementary) provide a visual explanation of our proposed loss. Our loss can be seen as a generalization of both the triplet [52] and N-pair losses [45]; the former uses only one positive and one negative sample per anchor, and the latter uses one positive and many negatives. The use of many positives and many negatives for each anchor allows us to achieve state of the art performance without the need for hard negative mining, which can be difficult to tune properly. To the best of our knowledge, this is the first contrastive loss to consistently perform better than cross-entropy on large-scale classification problems. Furthermore, it provides a unifying loss function that can be used for either self-supervised or supervised learning.

Our resulting loss, SupCon, is simple to implement and stable to train, as our empirical results show. It achieves excellent top-1 accuracy on the ImageNet dataset on the ResNet-50 and ResNet-200 architectures [17]. On ResNet-200 [5], we achieve a top-1 accuracy of 81.4%, which is a 0.8% improvement over the state of the art [30] cross-entropy loss on the same architecture (see Fig. 1). The gain in top-1 accuracy is accompanied by increased robustness as measured on the ImageNet-C dataset [19]. Our main contributions are summarized below:

1. We propose a novel extension to the contrastive loss function that allows for multiple positives per anchor, thus adapting contrastive learning to the fully supervised setting. Analytically and empirically, we show that a na¨ıve extension performs much worse than our proposed version.

2. We show that our loss provides consistent boosts in top-1 accuracy for a number of datasets. It is also more robust to natural corruptions.

3. We demonstrate analytically that the gradient of our loss function encourages learning from hard positives and hard negatives.

4. We show empirically that our loss is less sensitive than cross-entropy to a range of hyperparameters.

## 2 Related Work

Our work draws on existing literature in self-supervised representation learning, metric learning and supervised learning. Here we focus on the most relevant papers. The cross-entropy loss was introduced as a powerful loss function to train deep networks [40, 1, 29]. The key idea is simple and intuitive: each class is assigned a target (usually 1-hot) vector. However, it is unclear why these target labels should be the optimal ones and some work has tried to identify better target label vectors, e.g. [56]. A number of papers have studied other drawbacks of the cross-entropy loss, such as sensitivity to noisy labels [63, 46], presence of adversarial examples [10, 36], and poor margins [2]. Alternative losses have been proposed, but the most effective ideas in practice have been approaches that change the reference label distribution, such as label smoothing [47, 35], data augmentations such as Mixup [60] and CutMix [59], and knowledge distillation [21].

Powerful self-supervised representation learning approaches based on deep learning models have recently been developed in the natural language domain [8, 57, 33]. In the image domain, pixelpredictive approaches have also been used to learn embeddings [9, 61, 62, 37]. These methods try to predict missing parts of the input signal. However, a more effective approach has been to replace a dense per-pixel predictive loss, with a loss in lower-dimensional representation space. The state of the art family of models for self-supervised representation learning using this paradigm are collected under the umbrella of contrastive learning [54, 18, 22, 48, 43, 3, 50]. In these works, the losses are inspired by noise contrastive estimation [13, 34] or N-pair losses [45]. Typically, the loss is applied at the last layer of a deep network. At test time, the embeddings from a previous layer are utilized for downstream transfer tasks, fine tuning or direct retrieval tasks. [15] introduces the approximation of only back-propagating through part of the loss, and also the approximation of using stale representations in the form of a memory bank.

Closely related to contrastive learning is the family of losses based on metric distance learning or triplets [4, 52, 42]. These losses have been used to learn powerful representations, often in supervised settings, where labels are used to guide the choice of positive and negative pairs. The key distinction between triplet losses and contrastive losses is the number of positive and negative pairs per data point; triplet losses use exactly one positive and one negative pair per anchor. In the supervised metric learning setting, the positive pair is chosen from the same class and the negative pair is chosen from other classes, nearly always requiring hard-negative mining for good performance [42]. Self-supervised contrastive losses similarly use just one positive pair for each anchor sample, selected using either co-occurrence [18, 22, 48] or data augmentation [3]. The major difference is that many negative pairs are used for each anchor. These are usually chosen uniformly at random using some form of weak knowledge, such as patches from other images, or frames from other randomly chosen videos, relying on the assumption that this approach yields a very low probability of false negatives.

Resembling our supervised contrastive approach is the soft-nearest neighbors loss introduced in [41] and used in [53]. Like [53], we improve upon [41] by normalizing the embeddings and replacing euclidean distance with inner products. We further improve on [53] by the increased use of data augmentation, a disposable contrastive head and two-stage training (contrastive followed by crossentropy), and crucially, changing the form of the loss function to significantly improve results (see Section 3). [12] also uses a closely related loss formulation to ours to entangle representations at intermediate layers by maximizing the loss. Most similar to our method is the Compact Clustering via Label Propagation (CCLP) regularizer in Kamnitsas et. al. [24]. While CCLP focuses mostly on the semi-supervised case, in the fully supervised case the regularizer reduces to almost exactly our loss formulation. Important practical differences include our normalization of the contrastive embedding onto the unit sphere, tuning of a temperature parameter in the contrastive objective, and stronger augmentation. Additionally, Kamnitsas et. al. use the contrastive embedding as an input to a classification head, which is trained jointly with the CCLP regularizer, while SupCon employs a two stage training and discards the contrastive head. Lastly, the scale of experiments in Kamnitsas et. al. is much smaller than in this work. Merging the findings of our paper and CCLP is a promising direction for semi-supervised learning research.

## 3 Method

Our method is structurally similar to that used in [48, 3] for self-supervised contrastive learning, with modifications for supervised classification. Given an input batch of data, we first apply data augmentation twice to obtain two copies of the batch. Both copies are forward propagated through the encoder network to obtain a 2048-dimensional normalized embedding. During training, this representation is further propagated through a projection network that is discarded at inference time. The supervised contrastive loss is computed on the outputs of the projection network. To use the trained model for classification, we train a linear classifier on top of the frozen representations using a cross-entropy loss. Fig. 1 in the Supplementary material provides a visual explanation.

## 3.1 Representation Learning Framework

The main components of our framework are:

• Data Augmentation module, $A u g ( \cdot )$ . For each input sample, x, we generate two random augmentations, $\tilde { \pmb { x } } = A u g ( \pmb { x } )$ , each of which represents a different view of the data and contains some subset of the information in the original sample. Sec. 4 gives details of the augmentations.

• Encoder Network, $E n c ( \cdot )$ , which maps x to a representation vector, ${ \pmb r } = E { \pmb n } c ( { \pmb x } ) \in \mathcal { R } ^ { D _ { E } }$ . Both augmented samples are separately input to the same encoder, resulting in a pair of representation vectors. r is normalized to the unit hypersphere in $\mathcal { R } ^ { D _ { E } } \left( D _ { E } = 2 0 \bar { 4 } 8 \right.$ in all our experiments in the paper). Consistent with the findings of [42, 51], our analysis and experiments show that this normalization improves top-1 accuracy.

• Projection Network, $P r o j ( \cdot )$ , which maps r to a vector $z = P r o j ( r ) \in \mathcal { R } ^ { D _ { P } }$ . We instantiate ${ P r o j ( \cdot ) }$ as either a multi-layer perceptron [14] with a single hidden layer of size 2048 and output vector of size $D _ { P } = 1 2 8$ or just a single linear layer of size $D _ { P } = 1 \dot { 2 } 8 ;$ we leave to future work the investigation of optimal $P r o j ( \cdot )$ architectures. We again normalize the output of this network to lie on the unit hypersphere, which enables using an inner product to measure distances in the projection space. As in self-supervised contrastive learning [48, 3], we discard ${ P r o j ( \cdot ) }$ at the end of contrastive training. As a result, our inference-time models contain exactly the same number of parameters as a cross-entropy model using the same encoder, $E n c ( \cdot )$

## 3.2 Contrastive Loss Functions

Given this framework, we now look at the family of contrastive losses, starting from the selfsupervised domain and analyzing the options for adapting it to the supervised domain, showing that one formulation is superior. For a set of N randomly sampled sample/label pairs, $\{ \pmb { x } _ { k } , \pmb { y } _ { k } \} _ { k = 1 \dots N } .$ the corresponding batch used for training consists of 2N pairs, $\big \{ \tilde { { \pmb x } } _ { \ell } , \tilde { { \pmb y } } _ { \ell } \big \} _ { \ell = 1 \dots 2 N }$ , where ${ \tilde { \pmb x } } _ { 2 k }$ and $\tilde { x } _ { 2 k - 1 }$ are two random augmentations (a.k.a., “views”) of ${ \pmb x } _ { k } ( k = 1 . . . N )$ and $\tilde { { \pmb y } } _ { 2 k - 1 } = \tilde { { \pmb y } } _ { 2 k } = { \pmb y } _ { k }$ For the remainder of this paper, we will refer to a set of N samples as a “batch” and the set of $2 N$ augmented samples as a “multiviewed batch”.

## 3.2.1 Self-Supervised Contrastive Loss

Within a multiviewed batch, let $i \in I \equiv \left\{ 1 . . . 2 N \right\}$ be the index of an arbitrary augmented sample, and let $j ( i )$ be the index of the other augmented sample originating from the same source sample. In self-supervised contrastive learning (e.g., [3, 48, 18, 22]), the loss takes the following form.

$$
\mathcal {L} ^ {\text { self }} = \sum_ {i \in I} \mathcal {L} _ {i} ^ {\text { self }} = - \sum_ {i \in I} \log \frac {\exp \left(\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {j (i)} / \tau\right)}{\sum_ {a \in A (i)} \exp \left(\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {a} / \tau\right)}\tag{1}
$$

Here, $z _ { \ell } = P r o j ( E n c ( \tilde { \pmb { x } } _ { \ell } ) ) \in \mathcal { R } ^ { D _ { P } }$ , the • symbol denotes the inner (dot) product, $\tau \in \mathcal { R } ^ { + }$ is a scalar temperature parameter, and $A ( i ) \equiv I \dot { \setminus } \{ i \}$ . The index i is called the anchor, index $j ( i )$ is called the positive, and the other $2 ( N - 1 )$ indices $( \{ k \in A ( i ) \setminus \{ j ( i ) \} )$ are called the negatives.

Note that for each anchor i, there is 1 positive pair and $2 N - 2$ negative pairs. The denominator has a total of 2N − 1 terms (the positive and negatives).

## 3.2.2 Supervised Contrastive Losses

For supervised learning, the contrastive loss in Eq. 1 is incapable of handling the case where, due to the presence of labels, more than one sample is known to belong to the same class. Generalization to an arbitrary numbers of positives, though, leads to a choice between multiple possible functions. Eqs. 2 and 3 present the two most straightforward ways to generalize Eq. 1 to incorporate supervision.

$$
\mathcal {L} _ {o u t} ^ {s u p} = \sum_ {i \in I} \mathcal {L} _ {o u t, i} ^ {s u p} = \sum_ {i \in I} \frac {- 1}{| P (i) |} \sum_ {p \in P (i)} \log \frac {\exp (\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {p} / \tau)}{\sum_ {a \in A (i)} \exp (\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {a} / \tau)}\tag{2}
$$

$$
\mathcal {L} _ {i n} ^ {s u p} = \sum_ {i \in I} \mathcal {L} _ {i n, i} ^ {s u p} = \sum_ {i \in I} - \log \left\{\frac {1}{| P (i) |} \sum_ {p \in P (i)} \frac {\exp \left(\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {p} / \tau\right)}{\sum_ {a \in A (i)} \exp \left(\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {a} / \tau\right)} \right\}\tag{3}
$$

Here, $P ( i ) \equiv \{ p \in A ( i ) : \tilde { \pmb { y } } _ { p } = \tilde { \pmb { y } } _ { i } \}$ is the set of indices of all positives in the multiviewed batch distinct from i, and $| P ( i ) |$ | is its cardinality. In Eq. 2, the summation over positives is located outside of the log $( \mathcal { L } _ { o u t } ^ { s u p } )$ while in Eq. 3, the summation is located inside of the log $( \mathcal { L } _ { i n } ^ { s u p } )$ ). Both losses have the following desirable properties:

• Generalization to an arbitrary number of positives. The major structural change of Eqs. 2 and 3 over Eq. 1 is that now, for any anchor, all positives in a multiviewed batch (i.e., the augmentation-based sample as well as any of the remaining samples with the same label) contribute to the numerator. For randomly-generated batches whose size is large with respect to the number of classes, multiple additional terms will be present (on average, $\bar { N } / C$ , where C is the number of classes). The supervised losses encourage the encoder to give closely aligned representations to all entries from the same class, resulting in a more robust clustering of the representation space than that generated from Eq. 1, as is supported by our experiments in Sec. 4.

• Contrastive power increases with more negatives. Eqs. 2 and 3 both preserve the summation over negatives in the contrastive denominator of Eq. 1. This form is largely motivated by noise contrastive estimation and N-pair losses [13, 45], wherein the ability to discriminate between signal and noise (negatives) is improved by adding more examples of negatives. This property is important for representation learning via self-supervised contrastive learning, with many papers showing increased performance with increasing number of negatives [18, 15, 48, 3].

• Intrinsic ability to perform hard positive/negative mining. When used with normalized representations, the loss in Eq. 1 induces a gradient structure that gives rise to implicit hard positive/negative mining. The gradient contributions from hard positives/negatives (i.e., ones against which continuing to contrast the anchor greatly benefits the encoder) are large while those for easy positives/negatives (i.e., ones against which continuing to contrast the anchor only weakly benefits the encoder) are small. Furthermore, for hard positives, the effect increases (asymptotically) as the number of negatives does. Eqs. 2 and 3 both preserve this useful property and generalize it to all positives. This implicit property allows the contrastive loss to sidestep the need for explicit hard mining, which is a delicate but critical part of many losses, such as triplet loss [42]. We note that this implicit property applies to both supervised and self-supervised contrastive losses, but our derivation is the first to clearly show this property. We provide a full derivation of this property from the loss gradient in the Supplementary material.

The two loss formulations are not, however, equivalent. Because log is a concave function, Jensen’s Inequality [23] implies that $\mathcal { L } _ { o u t . } ^ { s u p } \le \mathcal { L } _ { i n } ^ { s u p }$ . One might thus be tempted to conclude that $\mathcal { L } _ { i n } ^ { s u p }$ is the superior supervised loss function (since it bounds $\mathcal { L } _ { o u t } ^ { s u p } )$ . However, this conclusion is not supported analytically. Table 1 compares the ImageNet [7] top-1 classification accuracy using $\mathcal { L } _ { o u t } ^ { s u p }$ and $\mathcal { L } _ { i n } ^ { s u p }$ for different batch sizes (N) on the ResNet-50 [17] architecture. The $\mathcal { L } _ { o u t } ^ { s u p }$ supervised

<table><tr><td>Loss</td><td>Top-1</td></tr><tr><td> $\mathcal{L}_{out}^{sup}$ </td><td>78.7%</td></tr><tr><td> $\mathcal{L}_{in}^{sup}$ </td><td>67.4%</td></tr></table>

Table 1: ImageNet Top-1 classification accuracy for supervised contrastive losses on ResNet-50 for a batch size of 6144.

loss achieves significantly higher performance than $\mathcal { L } _ { i n } ^ { \dot { s } u p }$ . We conjecture that this is due to the gradient of $\mathcal { L } _ { i n } ^ { s u p }$ having structure less optimal for training than that of $\mathcal { L } _ { o u t } ^ { s u p }$ . For $\mathcal { L } _ { o u t } ^ { s u p }$ , the positives normalization factor $( \mathrm { i } . \mathrm { e } . , 1 / | P ( i ) | )$ serves to remove bias present in the positives in a multiviewed batch contributing to the loss. However, though $\mathcal { L } _ { i n } ^ { s u p }$ also contains the same normalization factor, it is located inside of the log. It thus contributes only an additive constant to the overall loss, which does not affect the gradient. Without any normalization effects, the gradients of $\mathcal { L } _ { i n } ^ { s u p }$ are more susceptible to bias in the positives, leading to sub-optimal training.

An analysis of the gradients themselves supports this conclusion. As shown in the Supplementary, the gradient for either $\mathcal { L } _ { o u t , i } ^ { s u p }$ or $\mathcal { L } _ { i n , i } ^ { s u p }$ with respect to the embedding $z _ { i }$ has the following form.

$$
\frac {\partial \mathcal {L} _ {i} ^ {s u p}}{\partial \boldsymbol {z} _ {i}} = \frac {1}{\tau} \left\{\sum_ {p \in P (i)} \boldsymbol {z} _ {p} \left(P _ {i p} - X _ {i p}\right) + \sum_ {n \in N (i)} \boldsymbol {z} _ {n} P _ {i n} \right\}\tag{4}
$$

Here, $N ( i ) \equiv \{ n \in A ( i ) : \tilde { \pmb { y } } _ { n } \neq \tilde { \pmb { y } } _ { i } \}$ is the set of indices of all negatives in the multiviewed batch, and $\begin{array} { r } { P _ { i x } \equiv \exp \left( z _ { i } \cdot z _ { x } / \tau \right) / \sum _ { a \in A ( i ) } \exp \left( z _ { i } \cdot z _ { a } / \tau \right) } \end{array}$ . The difference between the gradients for the two losses is in $X _ { i p }$

$$
X _ {i p} = \left\{ \begin{array}{c c} \frac {\exp (\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {p} / \tau)}{\sum_ {p ^ {\prime} \in P (i)} \exp \bigl (\boldsymbol {z} _ {i} \cdot \boldsymbol {z} _ {p ^ {\prime}} / \tau \bigr)} & , \quad \text {if} \mathcal {L} _ {i} ^ {s u p} = \mathcal {L} _ {i n, i} ^ {s u p} \\ \frac {1}{| P (i) |} & , \quad \text {if} \mathcal {L} _ {i} ^ {s u p} = \mathcal {L} _ {o u t, i} ^ {s u p} \end{array} \right.\tag{5}
$$

If each $z _ { p }$ is set to the (less biased) mean positive representation vector, $\overline { { z } } , X _ { i p } ^ { i n }$ reduces to $X _ { i p } ^ { o u t }$

$$
\left. X _ {i p} ^ {i n} \right| _ {\boldsymbol {z} _ {p} = \overline {{\boldsymbol {z}}}} = \frac {\exp \left(\boldsymbol {z} _ {i} \cdot \overline {{\boldsymbol {z}}} / \tau\right)}{\sum_ {p ^ {\prime} \in P (i)} \exp \left(\boldsymbol {z} _ {i} \cdot \overline {{\boldsymbol {z}}} / \tau\right)} = \frac {\exp \left(\boldsymbol {z} _ {i} \cdot \overline {{\boldsymbol {z}}} / \tau\right)}{| P (i) | \cdot \exp \left(\boldsymbol {z} _ {i} \cdot \overline {{\boldsymbol {z}}} / \tau\right)} = \frac {1}{| P (i) |} = X _ {i p} ^ {o u t}\tag{6}
$$

From the form of ${ \partial \mathcal { L } _ { i } ^ { s u p } } / { \partial z _ { i } }$ , we conclude that the stabilization due to using the mean of positives benefits training. Throughout the rest of the paper, we consider only $\mathcal { L } _ { o u t } ^ { s u p }$

## 3.2.3 Connection to Triplet Loss and N-pairs Loss

Supervised contrastive learning is closely related to the triplet loss [52], one of the widely-used loss functions for supervised learning. In the Supplementary, we show that the triplet loss is a special case of the contrastive loss when one positive and one negative are used. When more than one negative is used, we show that the SupCon loss becomes equivalent to the N-pairs loss [45].

## 4 Experiments

We evaluate our SupCon loss $( \mathcal { L } _ { o u t } ^ { s u p } , \mathrm { E q . } 2 )$ by measuring classification accuracy on a number of common image classification benchmarks including CIFAR-10 and CIFAR-100 [27] and ImageNet [7]. We also benchmark our ImageNet models on robustness to common image corruptions [19] and show how performance varies with changes to hyperparameters and reduced data. For the encoder network $( E n c ( \cdot ) )$ we experimented with three commonly used encoder architectures: ResNet-50, ResNet-101, and ResNet-200 [17]. The normalized activations of the final pooling layer $( D _ { E } = 2 0 4 8 )$ are used as the representation vector. We experimented with four different implementations of the $A u g ( \cdot )$ data augmentation module: AutoAugment [5]; RandAugment [6]; SimAugment [3], and Stacked RandAugment [49] (see details of our SimAugment and Stacked RandAugment implementations in the Supplementary). AutoAugment outperforms all other data augmentation strategies on ResNet-50 for both SupCon and cross-entropy. Stacked RandAugment performed best for ResNet-200 for both loss functions. We provide more details in the Supplementary.

## 4.1 Classification Accuracy

Table 2 shows that SupCon generalizes better than cross-entropy, margin classifiers (with use of labels) and unsupervised contrastive learning techniques on CIFAR-10, CIFAR-100 and ImageNet datasets. Table 3 shows results for ResNet-50 and ResNet-200 (we use ResNet-v1 [17]) for ImageNet. We achieve a new state of the art accuracy of 78.7% on ResNet-50 with AutoAugment (for comparison, a number of the other top-performing methods are shown in Fig. 1). Note that we also achieve a slight improvement over CutMix [59], which is considered to be a state of the art data augmentation strategy. Incorporating data augmentation strategies such as CutMix [59] and MixUp [60] into contrastive learning could potentially improve results further.

<table><tr><td>Dataset</td><td>SimCLR[3]</td><td>Cross-Entropy</td><td>Max-Margin [32]</td><td>SupCon</td></tr><tr><td>CIFAR10</td><td>93.6</td><td>95.0</td><td>92.4</td><td>96.0</td></tr><tr><td>CIFAR100</td><td>70.7</td><td>75.3</td><td>70.5</td><td>76.5</td></tr><tr><td>ImageNet</td><td>70.2</td><td>78.2</td><td>78.0</td><td>78.7</td></tr></table>

Table 2: Top-1 classification accuracy on ResNet-50 [17] for various datasets. We compare cross-entropy training, unsupervised representation learning (SimCLR [3]), max-margin classifiers [32] and SupCon (ours). We re-implemented and tuned hyperparameters for all baseline numbers except margin classifiers where we report published results. Note that the CIFAR-10 and CIFAR-100 results are from our PyTorch implementation and ImageNet from our TensorFlow implementation.

<table><tr><td>Loss</td><td>Architecture</td><td>Augmentation</td><td>Top-1</td><td>Top-5</td></tr><tr><td>Cross-Entropy (baseline)</td><td>ResNet-50</td><td>MixUp [60]</td><td>77.4</td><td>93.6</td></tr><tr><td>Cross-Entropy (baseline)</td><td>ResNet-50</td><td>CutMix [59]</td><td>78.6</td><td>94.1</td></tr><tr><td>Cross-Entropy (baseline)</td><td>ResNet-50</td><td>AutoAugment [5]</td><td>78.2</td><td>92.9</td></tr><tr><td>Cross-Entropy (our impl.)</td><td>ResNet-50</td><td>AutoAugment [30]</td><td>77.6</td><td>95.3</td></tr><tr><td>SupCon</td><td>ResNet-50</td><td>AutoAugment [5]</td><td>78.7</td><td>94.3</td></tr><tr><td>Cross-Entropy (baseline)</td><td>ResNet-200</td><td>AutoAugment [5]</td><td>80.6</td><td>95.3</td></tr><tr><td>Cross-Entropy (our impl.)</td><td>ResNet-200</td><td>Stacked RandAugment [49]</td><td>80.9</td><td>95.2</td></tr><tr><td>SupCon</td><td>ResNet-200</td><td>Stacked RandAugment [49]</td><td>81.4</td><td>95.9</td></tr><tr><td>SupCon</td><td>ResNet-101</td><td>Stacked RandAugment [49]</td><td>80.2</td><td>94.7</td></tr></table>

Table 3: Top-1/Top-5 accuracy results on ImageNet for AutoAugment [5] with ResNet-50 and for Stacked RandAugment [49] with ResNet-101 and ResNet-200. The baseline numbers are taken from the referenced papers, and we also re-implement cross-entropy.

We also experimented with memory based alternatives [15]. On ImageNet, with a memory size of 8192 (requiring only the storage of 128-dimensional vectors), a batch size of 256, and SGD optimizer, running on 8 Nvidia V100 GPUs, SupCon is able to achieve 79.1% top-1 accuracy on ResNet-50. This is in fact slightly better than the 78.7% accuracy with 6144 batch size (and no memory); and with significantly reduced compute and memory footprint.

Since SupCon uses 2 views per sample, its batch sizes are effectively twice the cross-entropy equivalent. We therefore also experimented with the cross-entropy ResNet-50 baselines using a batch size of 12,288. These only achieved 77.5% top-1 accuracy. We additionally experimented with increasing the number of training epochs for cross-entropy all the way to 1400, but this actually decreased accuracy (77.0%).

We tested the N-pairs loss [45] in our framework with a batch size of 6144. N-pairs achieves only 57.4% top-1 accuracy on ImageNet. We believe this is due to multiple factors missing from N-pairs loss compared to supervised contrastive: the use of multiple views; lower temperature; and many more positives. We show some results of the impact of the number of positives per anchor in the Supplementary (Sec. 6), and the N-pairs result is inline with them. We also note that the original N-pairs paper [45] has already shown the outperformance of N-pairs loss to triplet loss.

## 4.2 Robustness to Image Corruptions and Reduced Training Data

Deep neural networks lack robustness to out of distribution data or natural corruptions such as noise, blur and JPEG compression. The benchmark ImageNet-C dataset [19] is used to measure trained model performance on such corruptions. In Fig. 3(left), we compare the supervised contrastive models to cross-entropy using the Mean Corruption Error (mCE) and Relative Mean Corruption Error metrics [19]. Both metrics measure average degradation in performance compared to ImageNet test set, averaged over all possible corruptions and severity levels. Relative mCE is a better metric when we compare models with different Top-1 accuracy, while mCE is a better measure of absolute robustness to corruptions. The SupCon models have lower mCE values across different corruptions, showing increased robustness. We also see from Fig. 3(right) that SupCon models demonstrate lesser degradation in accuracy with increasing corruption severity.

<table><tr><td>Loss</td><td>Architecture</td><td>rel. mCE</td><td>mCE</td></tr><tr><td rowspan="3">Cross-Entropy (baselines)</td><td>AlexNet [28]</td><td>100.0</td><td>100.0</td></tr><tr><td>VGG-19+BN [44]</td><td>122.9</td><td>81.6</td></tr><tr><td>ResNet-18 [17]</td><td>103.9</td><td>84.7</td></tr><tr><td rowspan="2">Cross-Entropy (our implementation)</td><td>ResNet-50</td><td>96.2</td><td>68.6</td></tr><tr><td>ResNet-200</td><td>69.1</td><td>52.4</td></tr><tr><td rowspan="2">Supervised Contrastive</td><td>ResNet-50</td><td>94.6</td><td>67.2</td></tr><tr><td>ResNet-200</td><td>66.5</td><td>50.6</td></tr></table>

![](images/874ad673405bc796fe6a313ce140c86beb4467ef8163dfcb5f9565ba1e289a3c.jpg)  
Figure 3: Training with supervised contrastive loss makes models more robust to corruptions in images. Left: Robustness as measured by Mean Corruption Error (mCE) and relative mCE over the ImageNet-C dataset [19] (lower is better). Right: Mean Accuracy as a function of corruption severity averaged over all various corruptions. (higher is better).

![](images/9bd6be7efbd68b160e30e6ef9608b0b985a1123755b9e7fb7fec27a2199df7ad.jpg)

![](images/29fd72953f1bcba95ebaf66470446fc98a064ab91b133907c1ab9cdc16872f3c.jpg)

![](images/700c392bbede39cf9f3e002b4f289b1d125914547c43ebaa7c071a5066d9b412.jpg)

![](images/b45a8c90f78d700c3667bb8ac8a6e41ae350cea1777d529fd4c626418a86437c.jpg)  
Figure 4: Accuracy of cross-entropy and supervised contrastive loss as a function of hyperparameters and training data size, all measured on ImageNet with a ResNet-50 encoder. (From left to right) (a): Standard boxplot showing Top-1 accuracy vs changes in augmentation, optimizer and learning rates. (b): Top-1 accuracy as a function of batch size shows both losses benefit from larger batch sizes while Supervised Contrastive has higher Top-1 accuracy even when trained with smaller batch sizes. (c): Top-1 accuracy as a function of SupCon pretraining epochs. (d): Top-1 accuracy as a function of temperature during pretraining stage for SupCon.

<table><tr><td></td><td>Food</td><td>CIFAR10</td><td>CIFAR100</td><td>Birdsnap</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>VOC2007</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td><td>Mean</td></tr><tr><td>SimCLR-50 [3]</td><td>88.20</td><td>97.70</td><td>85.90</td><td>75.90</td><td>63.50</td><td>91.30</td><td>88.10</td><td>84.10</td><td>73.20</td><td>89.20</td><td>92.10</td><td>97.00</td><td>84.81</td></tr><tr><td>Xent-50</td><td>87.38</td><td>96.50</td><td>84.93</td><td>74.70</td><td>63.15</td><td>89.57</td><td>80.80</td><td>85.36</td><td>76.86</td><td>92.35</td><td>92.34</td><td>96.93</td><td>84.67</td></tr><tr><td>SupCon-50</td><td>87.23</td><td>97.42</td><td>84.27</td><td>75.15</td><td>58.04</td><td>91.69</td><td>84.09</td><td>85.17</td><td>74.60</td><td>93.47</td><td>91.04</td><td>96.0</td><td>84.27</td></tr><tr><td>Xent-200</td><td>89.36</td><td>97.96</td><td>86.49</td><td>76.50</td><td>64.36</td><td>90.01</td><td>84.22</td><td>86.27</td><td>76.76</td><td>93.48</td><td>93.84</td><td>97.20</td><td>85.77</td></tr><tr><td>SupCon-200</td><td>88.62</td><td>98.28</td><td>87.28</td><td>76.26</td><td>60.46</td><td>91.78</td><td>88.68</td><td>85.18</td><td>74.26</td><td>93.12</td><td>94.91</td><td>96.97</td><td>85.67</td></tr></table>

Table 4: Transfer learning results. Numbers are mAP for VOC2007 [11]; mean-per-class accuracy for Aircraft, Pets, Caltech, and Flowers; and top-1 accuracy for all other datasets.

## 4.3 Hyperparameter Stability

We experimented with hyperparameter stability by changing augmentations, optimizers and learning rates one at a time from the best combination for each of the methodologies. In Fig. 4(a), we compare the top-1 accuracy of SupCon loss against cross-entropy across changes in augmentations (RandAugment [6], AutoAugment [5], SimAugment [3], Stacked RandAugment [49]); optimizers (LARS, SGD with Momentum and RMSProp); and learning rates. We observe significantly lower variance in the output of the contrastive loss. Note that batch sizes for cross-entropy and supervised contrastive are the same, thus ruling out any batch-size effects. In Fig. 4(b), sweeping batch size and holding all other hyperparameters constant results in consistently better top-1 accuracy of the supervised contrastive loss.

## 4.4 Transfer Learning

We evaluate the learned representation for fine-tuning on 12 natural image datasets, following the protocol in Chen et.al. [3]. SupCon is on par with cross-entropy and self-supervised contrastive loss on transfer learning performance when trained on the same architecture (Table 4). Our results are consistent with the findings in [16] and [26]: while better ImageNet models are correlated with better transfer performance, the dominant factor is architecture. Understanding the connection between training objective and transfer performance is left to future work.

## 4.5 Training Details

The SupCon loss was trained for 700 epochs during pretraining for ResNet-200 and 350 epochs for smaller models. Fig. 4(c) shows accuracy as a function of SupCon training epochs for a ResNet50, demonstrating that even 200 epochs is likely sufficient for most purposes.

An (optional) additional step of training a linear classifier is used to compute top-1 accuracy. This is not needed if the purpose is to use representations for transfer learning tasks or retrieval. The second stage needs as few as 10 epochs of additional training. Note that in practice the linear classifier can be trained jointly with the encoder and projection networks by blocking gradient propagation from the linear classifier back to the encoder, and achieve roughly the same results without requiring two-stage training. We chose not to do that here to help isolate the effects of the SupCon loss.

We trained our models with batch sizes of up to 6144, although batch sizes of 2048 suffice for most purposes for both SupCon and cross-entropy losses (as shown in Fig. 4(b)). We associate some of the performance increase with batch size to the effect on the gradient due to hard positives increasing with an increasing number of negatives (see the Supplementary for details). We report metrics for experiments with batch size 6144 for ResNet-50 and batch size 4096 for ResNet-200 (due to the larger network size, a smaller batch size is necessary). We observed that for a fixed batch size it was possible to train with SupCon using larger learning rates than what was required by cross-entropy to achieve similar performance.

All our results used a temperature of τ = 0.1. Smaller temperature benefits training more than higher ones, but extremely low temperatures are harder to train due to numerical instability. Fig. 4(d) shows the effect of temperature on Top-1 performance of supervised contrastive learning. As we can see from Eq. 4, the gradient scales inversely with choice of temperature τ ; therefore we rescale the loss by τ during training for stability.

We experimented with standard optimizers such as LARS [58], RMSProp [20] and SGD with momentum [39] in different permutations for the initial pre-training step and training of the dense layer. While SGD with momentum works best for training ResNets with cross-entropy, we get the best performance for SupCon on ImageNet by using LARS for pre-training and RMSProp to training the linear layer. For CIFAR10 and CIFAR100 SGD with momentum performed best. Additional results for combinations of optimizers are provided in the Supplementary. Reference code is released at https://t.ly/supcon.

## Broader Impact

This work provides a technical advancement in the field of supervised classification, which already has tremendous impact throughout industry. Whether or not they realize it, most people experience the results of this type of classifier many times a day.

As we have shown, supervised contrastive learning can improve both the accuracy and robustness of classifiers, which for most applications should strictly be an improvement. For example, an autonomous car that makes a classification error due to data distribution shift can result in catastrophic results. Thus decreasing this class of error undoubtedly promotes safety. Human driver error is a massive source of fatalities around the world, so improving the safety of autonomous cars furthers the efforts of replacing human drivers. The flip side of that progress is of course the potential for loss of employment in fields like trucking and taxi driving. Similar two-sided coins can be considered for assessing the impact of any application of classification.

An additional potential impact of our work in particular is showing the value of training with large batch sizes. Generally, large batch size training comes at the cost of substantial energy consumption, which unfortunately today requires the burning of fossil fuels, which in turn warms our planet. We are proud to say that the model training that was done in the course of this research was entirely carbon-neutral, where all power consumed was either green to start with, or offset by purchases of green energy. There is unfortunately no way to guarantee that once this research is publicly available that all practitioners of it will choose, or even have the ability to choose, to limit the environmental impact of their model training.

## Acknowledgments and Disclosure of Funding

Additional revenues related to this work: In the past 36 months, Phillip Isola has had employment at MIT, Google, and OpenAI; honorarium for lecturing at the ACDL summer school in Italy; honorarium for speaking at GIST AI Day in South Korea. P.I.’s lab at MIT has been supported by grants from Facebook, IBM, and the US Air Force; start up funding from iFlyTech via MIT; gifts from Adobe and Google; compute credit donations from Google Cloud.

## References

[1] Eric B Baum and Frank Wilczek. Supervised learning of probability distributions by neural networks. In Neural information processing systems, pages 52–61, 1988. 3

[2] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with label-distribution-aware margin loss. In Advances in Neural Information Processing Systems, pages 1565–1576, 2019. 3

[3] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709, 2020. 2, 3, 4, 5, 6, 7, 8

[4] Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 1, pages 539–546. IEEE, 2005. 3

[5] Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. Autoaugment: Learning augmentation strategies from data. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 113–123, 2019. 1, 2, 6, 7, 8

[6] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical data augmentation with no separate search. arXiv preprint arXiv:1909.13719, 2019. 1, 6, 8

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, 2009. 1, 5, 6

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 3

[9] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In Proceedings of the IEEE International Conference on Computer Vision, pages 1422–1430, 2015. 3

[10] Gamaleldin Elsayed, Dilip Krishnan, Hossein Mobahi, Kevin Regan, and Samy Bengio. Large margin deep networks for classification. In Advances in neural information processing systems, pages 842–852, 2018. 1, 3

[11] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results. http://www.pascalnetwork.org/challenges/VOC/voc2007/workshop/index.html. 8

[12] Nicholas Frosst, Nicolas Papernot, and Geoffrey E. Hinton. Analyzing and improving representations with the soft nearest neighbor loss. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 2012–2020. PMLR, 2019. 3

[13] Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation prin-¨ ciple for unnormalized statistical models. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pages 297–304, 2010. 3, 5

[14] Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning. Springer Series in Statistics. Springer New York Inc., New York, NY, USA, 2001. 4

[15] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. arXiv preprint arXiv:1911.05722, 2019. 2, 3, 5, 7

[16] Kaiming He, Ross Girshick, and Piotr Dollar. Rethinking imagenet pre-training. In ´ Proceedings of the IEEE International Conference on Computer Vision, pages 4918–4927, 2019. 8

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2, 5, 6, 7, 8

[18] Olivier J Henaff, Ali Razavi, Carl Doersch, SM Eslami, and Aaron van den Oord.´ Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272, 2019. 2, 3, 4, 5

[19] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019. 2, 6, 7, 8

[20] Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. Cited on, 14(8), 2012. 9

[21] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 3

[22] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In International Conference on Learning Representations, 2019. 2, 3, 4

[23] J. L. W. V. Jensen. Sur les fonctions convexes et les inegalites entre les valeurs moyennes. Acta Math, 1906. 5

[24] Konstantinos Kamnitsas, Daniel C. Castro, Lo¨ıc Le Folgoc, Ian Walker, Ryutaro Tanno, Daniel Rueckert, Ben Glocker, Antonio Criminisi, and Aditya V. Nori. Semi-supervised learning via compact latent space clustering. In International Conference on Machine Learning, 2018. 3

[25] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Large scale learning of general visual representations for transfer. arXiv preprint arXiv:1912.11370, 2019. 1

[26] Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better imagenet models transfer better? In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2661–2671, 2019. 8

[27] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 6

[28] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. 8

[29] Esther Levin and Michael Fleisher. Accelerated learning in layered neural networks. Complex systems, 2:625–640, 1988. 3

[30] Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, and Sungwoong Kim. Fast autoaugment. arXiv preprint arXiv:1905.00397, 2019. 2, 7

[31] Weiyang Liu, Yandong Wen, Zhiding Yu, and Meng Yang. Large-margin softmax loss for convolutional neural networks. In ICML, volume 2, page 7, 2016. 1

[32] Weiyang Liu, Yandong Wen, Zhiding Yu, and Meng Yang. Large-margin softmax loss for convolutional neural networks, 2016. 7

[33] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013. 3

[34] Andriy Mnih and Koray Kavukcuoglu. Learning word embeddings efficiently with noisecontrastive estimation. In Advances in neural information processing systems, pages 2265– 2273, 2013. 3

[35] Rafael Muller, Simon Kornblith, and Geoffrey E Hinton. When does label smoothing help? In¨ Advances in Neural Information Processing Systems, pages 4696–4705, 2019. 3

[36] Kamil Nar, Orhan Ocal, S Shankar Sastry, and Kannan Ramchandran. Cross-entropy loss and low-rank features have responsibility for adversarial examples. arXiv preprint arXiv:1901.08360, 2019. 3

[37] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European Conference on Computer Vision, pages 69–84. Springer, 2016. 3

[38] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 2

[39] Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016. 9

[40] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986. 3

[41] Ruslan Salakhutdinov and Geoff Hinton. Learning a nonlinear embedding by preserving class neighbourhood structure. In Artificial Intelligence and Statistics, pages 412–419, 2007. 3

[42] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015. 3, 4, 5

[43] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In 2018 IEEE International Conference on Robotics and Automation (ICRA), 2018. 3

[44] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015. 8

[45] Kihyuk Sohn. Improved deep metric learning with multi-class n-pair loss objective. In Advances in neural information processing systems, pages 1857–1865, 2016. 2, 3, 5, 6, 7

[46] Sainbayar Sukhbaatar, Joan Bruna, Manohar Paluri, Lubomir Bourdev, and Rob Fergus. Training convolutional networks with noisy labels. arXiv preprint arXiv:1406.2080, 2014. 1, 3

[47] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016. 3

[48] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849, 2019. 2, 3, 4, 5

[49] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning. arXiv preprint arXiv:2005.10243, 2019. 6, 7, 8

[50] Michael Tschannen, Josip Djolonga, Paul K Rubenstein, Sylvain Gelly, and Mario Lucic. On mutual information maximization for representation learning. arXiv preprint arXiv:1907.13625, 2019. 3

[51] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. arXiv preprint arXiv:2005.10242, 2020. 4

[52] Kilian Q Weinberger and Lawrence K Saul. Distance metric learning for large margin nearest neighbor classification. Journal of Machine Learning Research, 10(Feb):207–244, 2009. 2, 3, 6

[53] Zhirong Wu, Alexei A Efros, and Stella Yu. Improving generalization via scalable neighborhood component analysis. In European Conference on Computer Vision (ECCV) 2018, 2018. 3

[54] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3733–3742, 2018. 2, 3

[55] Qizhe Xie, Eduard Hovy, Minh-Thang Luong, and Quoc V Le. Self-training with noisy student improves imagenet classification. arXiv preprint arXiv:1911.04252, 2019. 1

[56] Shuo Yang, Ping Luo, Chen Change Loy, Kenneth W Shum, and Xiaoou Tang. Deep representation learning with target coding. In Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015. 3

[57] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems, pages 5754–5764, 2019. 3

[58] Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017. 9

[59] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE International Conference on Computer Vision, pages 6023–6032, 2019. 1, 3, 7

[60] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 3, 7

[61] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In European conference on computer vision, pages 649–666. Springer, 2016. 3

[62] Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1058–1067, 2017. 3

[63] Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In Advances in neural information processing systems, pages 8778– 8788, 2018. 1, 3