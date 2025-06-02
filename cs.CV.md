# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space](https://arxiv.org/abs/2303.14537) | 深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。 |
| [^2] | [On the Design Fundamentals of Diffusion Models: A Survey.](http://arxiv.org/abs/2306.04542) | 本文综述了扩散模型的设计基础，即其三个关键组件：正向过程、逆向过程和采样过程，为未来的研究提供了有益的细粒度透视。 |

# 详细

[^1]: 深度增强：在激活空间中使用自监督学习进行数据增强

    Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space

    [https://arxiv.org/abs/2303.14537](https://arxiv.org/abs/2303.14537)

    深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。

    

    我们提出了一种称为深度增强的方法，通过使用辍学或PCA来转换神经网络中的目标层，以提高性能和泛化能力。我们通过在自然语言处理、计算机视觉和图学习中的对比学习任务上进行大量实验来展示深度增强。 我们观察到在对比学习的基础模型中，如Transformers、ResNets和图神经网络上深度增强能够带来显著的性能提升，但在相应的监督问题上观察到相反的效果。 我们的分析表明，深度增强减轻了层之间的相互适应，即"崩溃"形式的问题。 我们利用这一观察结果制定了一种选择目标层的方法；特别是，我们的实验表明，用深度增强定位更深层次的层要优于增强输入数据。 这种方法的简单网络和模态无关性使其

    arXiv:2303.14537v2 Announce Type: replace-cross  Abstract: We introduce Deep Augmentation, an approach to implicit data augmentation using dropout or PCA to transform a targeted layer within a neural network to improve performance and generalization. We demonstrate Deep Augmentation through extensive experiments on contrastive learning tasks in NLP, computer vision, and graph learning. We observe substantial performance gains with Transformers, ResNets, and Graph Neural Networks as the underlying models in contrastive learning, but observe inverse effects on the corresponding supervised problems. Our analysis suggests that Deep Augmentation alleviates co-adaption between layers, a form of "collapse." We use this observation to formulate a method for selecting which layer to target; in particular, our experimentation reveals that targeting deeper layers with Deep Augmentation outperforms augmenting the input data. The simple network- and modality-agnostic nature of this approach enables
    
[^2]: 关于扩散模型的设计基础：综述

    On the Design Fundamentals of Diffusion Models: A Survey. (arXiv:2306.04542v1 [cs.LG])

    [http://arxiv.org/abs/2306.04542](http://arxiv.org/abs/2306.04542)

    本文综述了扩散模型的设计基础，即其三个关键组件：正向过程、逆向过程和采样过程，为未来的研究提供了有益的细粒度透视。

    

    扩散模型是一种生成模型，通过逐渐添加和删除噪声来学习训练数据的潜在分布以生成数据。扩散模型的组成部分已经受到了广泛的关注，许多设计选择被提出。现有的评论主要关注高层次的解决方案，对组件的设计基础覆盖较少。本研究旨在通过提供一个全面而连贯的综述，针对扩散模型的组件设计选择进行分析。具体来说，我们将这个综述按照三个关键组件进行组织，即正向过程、逆向过程和采样过程。这使得我们可以提供扩散模型的细粒度透视，有助于未来研究分析个体组件、设计选择的适用性以及扩散模型的实现。

    Diffusion models are generative models, which gradually add and remove noise to learn the underlying distribution of training data for data generation. The components of diffusion models have gained significant attention with many design choices proposed. Existing reviews have primarily focused on higher-level solutions, thereby covering less on the design fundamentals of components. This study seeks to address this gap by providing a comprehensive and coherent review on component-wise design choices in diffusion models. Specifically, we organize this review according to their three key components, namely the forward process, the reverse process, and the sampling procedure. This allows us to provide a fine-grained perspective of diffusion models, benefiting future studies in the analysis of individual components, the applicability of design choices, and the implementation of diffusion models.
    

