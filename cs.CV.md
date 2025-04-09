# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Supervised Siamese Autoencoders.](http://arxiv.org/abs/2304.02549) | 本论文提出了一种新的自监督方法，名为SidAE，它结合了孪生架构和去噪自编码器的优点，可以更好地提取输入数据的特征，以在多个下游任务中获得更好的性能。 |

# 详细

[^1]: 自监督的孪生自编码器

    Self-Supervised Siamese Autoencoders. (arXiv:2304.02549v1 [cs.LG])

    [http://arxiv.org/abs/2304.02549](http://arxiv.org/abs/2304.02549)

    本论文提出了一种新的自监督方法，名为SidAE，它结合了孪生架构和去噪自编码器的优点，可以更好地提取输入数据的特征，以在多个下游任务中获得更好的性能。

    

    完全监督的模型通常需要大量的标记训练数据，这往往是昂贵且难以获得的。相反，自监督表示学习减少了实现相同或更高下游性能所需的标记数据量。目标是在自监督任务上预先训练深度神经网络，以便网络能够从原始输入数据中提取有意义的特征。然后，将这些特征用作下游任务（如图像分类）中的输入。在先前的研究中，自编码器和孪生网络（如SimSiam）已成功应用于这些任务中。然而，仍然存在一些挑战，例如将特征的特性（例如，细节级别）与给定的任务和数据集匹配。在本文中，我们提出了一种结合了孪生架构和去噪自编码器优势的新自监督方法。我们展示了我们的模型，名为SidAE（孪生去噪自编码器），在多个下游任务上胜过了两个自监督最新基准。

    Fully supervised models often require large amounts of labeled training data, which tends to be costly and hard to acquire. In contrast, self-supervised representation learning reduces the amount of labeled data needed for achieving the same or even higher downstream performance. The goal is to pre-train deep neural networks on a self-supervised task such that afterwards the networks are able to extract meaningful features from raw input data. These features are then used as inputs in downstream tasks, such as image classification. Previously, autoencoders and Siamese networks such as SimSiam have been successfully employed in those tasks. Yet, challenges remain, such as matching characteristics of the features (e.g., level of detail) to the given task and data set. In this paper, we present a new self-supervised method that combines the benefits of Siamese architectures and denoising autoencoders. We show that our model, called SidAE (Siamese denoising autoencoder), outperforms two se
    

