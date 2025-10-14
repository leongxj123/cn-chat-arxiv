# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning.](http://arxiv.org/abs/2401.06344) | 本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。 |
| [^2] | [Camouflaged Image Synthesis Is All You Need to Boost Camouflaged Detection.](http://arxiv.org/abs/2308.06701) | 该研究提出了一个用于合成伪装数据以改善对自然场景中伪装物体检测的框架，该方法利用生成模型生成逼真的伪装图像，并在三个数据集上取得了优于目前最先进方法的结果。 |

# 详细

[^1]: 超级-STTN：社交群体感知的时空转换网络用于人体轨迹预测与超图推理

    Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning. (arXiv:2401.06344v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2401.06344](http://arxiv.org/abs/2401.06344)

    本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    

    在各种现实世界的应用中，包括服务机器人和自动驾驶汽车，预测拥挤的意图和轨迹是至关重要的。理解环境动态是具有挑战性的，不仅因为对建模成对的空间和时间相互作用的复杂性，还因为群体间相互作用的多样性。为了解码拥挤场景中全面的成对和群体间相互作用，我们引入了Hyper-STTN，这是一种基于超图的时空转换网络，用于人群轨迹预测。在Hyper-STTN中，通过一组多尺度超图构建了拥挤的群体间相关性，这些超图具有不同的群体大小，通过基于随机游走概率的超图谱卷积进行捕捉。此外，还采用了空间-时间转换器来捕捉行人在空间-时间维度上的对照相互作用。然后，这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    Predicting crowded intents and trajectories is crucial in varouls real-world applications, including service robots and autonomous vehicles. Understanding environmental dynamics is challenging, not only due to the complexities of modeling pair-wise spatial and temporal interactions but also the diverse influence of group-wise interactions. To decode the comprehensive pair-wise and group-wise interactions in crowded scenarios, we introduce Hyper-STTN, a Hypergraph-based Spatial-Temporal Transformer Network for crowd trajectory prediction. In Hyper-STTN, crowded group-wise correlations are constructed using a set of multi-scale hypergraphs with varying group sizes, captured through random-walk robability-based hypergraph spectral convolution. Additionally, a spatial-temporal transformer is adapted to capture pedestrians' pair-wise latent interactions in spatial-temporal dimensions. These heterogeneous group-wise and pair-wise are then fused and aligned though a multimodal transformer net
    
[^2]: 伪装图像合成是提高伪装物体检测的关键

    Camouflaged Image Synthesis Is All You Need to Boost Camouflaged Detection. (arXiv:2308.06701v1 [cs.CV])

    [http://arxiv.org/abs/2308.06701](http://arxiv.org/abs/2308.06701)

    该研究提出了一个用于合成伪装数据以改善对自然场景中伪装物体检测的框架，该方法利用生成模型生成逼真的伪装图像，并在三个数据集上取得了优于目前最先进方法的结果。

    

    融入自然场景的伪装物体给深度学习模型检测和合成带来了重大挑战。伪装物体检测是计算机视觉中一个关键任务，具有广泛的实际应用，然而由于数据有限，该研究课题一直受到限制。我们提出了一个用于合成伪装数据以增强对自然场景中伪装物体检测的框架。我们的方法利用生成模型生成逼真的伪装图像，这些图像可以用来训练现有的物体检测模型。具体而言，我们使用伪装环境生成器，由伪装分布分类器进行监督，合成伪装图像，然后将其输入我们的生成器以扩展数据集。我们的框架在三个数据集（COD10k、CAMO和CHAMELEON）上的效果超过了目前最先进的方法，证明了它在改善伪装物体检测方面的有效性。

    Camouflaged objects that blend into natural scenes pose significant challenges for deep-learning models to detect and synthesize. While camouflaged object detection is a crucial task in computer vision with diverse real-world applications, this research topic has been constrained by limited data availability. We propose a framework for synthesizing camouflage data to enhance the detection of camouflaged objects in natural scenes. Our approach employs a generative model to produce realistic camouflage images, which can be used to train existing object detection models. Specifically, we use a camouflage environment generator supervised by a camouflage distribution classifier to synthesize the camouflage images, which are then fed into our generator to expand the dataset. Our framework outperforms the current state-of-the-art method on three datasets (COD10k, CAMO, and CHAMELEON), demonstrating its effectiveness in improving camouflaged object detection. This approach can serve as a plug-
    

