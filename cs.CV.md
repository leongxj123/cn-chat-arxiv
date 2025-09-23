# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faster and Accurate Neural Networks with Semantic Inference.](http://arxiv.org/abs/2310.01259) | 本研究提出了一种名为语义推理（SINF）的新框架，在减少计算负载的同时，通过聚类语义相似的类来提取子图，从而减少深度神经网络的计算负担，并在性能上有限损失。 |
| [^2] | [Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations.](http://arxiv.org/abs/2303.10523) | 本文提出了一种无监督的方法，通过对CNN进行转换，从而更好地解释中间层的表示，提取了一个可解释性欠完备基础，并证明该方法在各种网络结构和训练数据集上都很有效。 |

# 详细

[^1]: 使用语义推理实现更快更准确的神经网络

    Faster and Accurate Neural Networks with Semantic Inference. (arXiv:2310.01259v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.01259](http://arxiv.org/abs/2310.01259)

    本研究提出了一种名为语义推理（SINF）的新框架，在减少计算负载的同时，通过聚类语义相似的类来提取子图，从而减少深度神经网络的计算负担，并在性能上有限损失。

    

    深度神经网络通常具有显著的计算负担。虽然提出了结构化剪枝和专门用于移动设备的神经网络的方法，但它们会导致明显的准确率损失。在本文中，我们利用潜在表示中的内在冗余来减少计算负载，并在性能上有限损失。我们证明，语义上相似的输入共享许多滤波器，尤其是在较早的层次上。因此，可以对语义上相似的类进行聚类，以创建特定于聚类的子图。为此，我们提出了一个名为语义推理（SINF）的新框架。简而言之，SINF（i）使用一个小的附加分类器来识别对象属于的语义聚类，并（ii）执行与该语义聚类相关的基本DNN提取的子图进行推理。为了提取每个特定于聚类的子图，我们提出了一个名为区分能力得分（DCS）的新方法，用于找到具有区分能力的子图。

    Deep neural networks (DNN) usually come with a significant computational burden. While approaches such as structured pruning and mobile-specific DNNs have been proposed, they incur drastic accuracy loss. In this paper we leverage the intrinsic redundancy in latent representations to reduce the computational load with limited loss in performance. We show that semantically similar inputs share many filters, especially in the earlier layers. Thus, semantically similar classes can be clustered to create cluster-specific subgraphs. To this end, we propose a new framework called Semantic Inference (SINF). In short, SINF (i) identifies the semantic cluster the object belongs to using a small additional classifier and (ii) executes the subgraph extracted from the base DNN related to that semantic cluster for inference. To extract each cluster-specific subgraph, we propose a new approach named Discriminative Capability Score (DCS) that finds the subgraph with the capability to discriminate amon
    
[^2]: 无监督解释性基础抽取用于基于概念的视觉解释

    Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations. (arXiv:2303.10523v1 [cs.CV])

    [http://arxiv.org/abs/2303.10523](http://arxiv.org/abs/2303.10523)

    本文提出了一种无监督的方法，通过对CNN进行转换，从而更好地解释中间层的表示，提取了一个可解释性欠完备基础，并证明该方法在各种网络结构和训练数据集上都很有效。

    

    研究人员尝试用人类可以理解的概念来解释CNN图像分类器预测和中间层表示。本文提出了一种无监督后处理方法，通过查找解释像素激活的稀疏二值化转换表示的特征空间旋转来提取解释性欠完备基础。我们对现有的流行CNN进行了实验，并证明了我们方法在网络架构和训练数据集上提取解释性基础的有效性。最后，我们扩展了文献中的基础可解释性度量，并表明，当中间层表示被转换为我们方法提取的基础时，它们变得更易解释。

    An important line of research attempts to explain CNN image classifier predictions and intermediate layer representations in terms of human understandable concepts. In this work, we expand on previous works in the literature that use annotated concept datasets to extract interpretable feature space directions and propose an unsupervised post-hoc method to extract a disentangling interpretable basis by looking for the rotation of the feature space that explains sparse one-hot thresholded transformed representations of pixel activations. We do experimentation with existing popular CNNs and demonstrate the effectiveness of our method in extracting an interpretable basis across network architectures and training datasets. We make extensions to the existing basis interpretability metrics found in the literature and show that, intermediate layer representations become more interpretable when transformed to the bases extracted with our method. Finally, using the basis interpretability metrics
    

