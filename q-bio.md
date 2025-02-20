# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FakET: Simulating Cryo-Electron Tomograms with Neural Style Transfer.](http://arxiv.org/abs/2304.02011) | 本文提出了一种使用加性噪声和神经风格迁移技术来模拟电子显微镜正向算子，以解决深度学习方法需要大量训练数据集的问题。该方法在粒子定位和分类任务上表现良好。 |

# 详细

[^1]: FakET: 利用神经风格迁移模拟冷冻电子断层图像

    FakET: Simulating Cryo-Electron Tomograms with Neural Style Transfer. (arXiv:2304.02011v1 [cs.LG])

    [http://arxiv.org/abs/2304.02011](http://arxiv.org/abs/2304.02011)

    本文提出了一种使用加性噪声和神经风格迁移技术来模拟电子显微镜正向算子，以解决深度学习方法需要大量训练数据集的问题。该方法在粒子定位和分类任务上表现良好。

    

    粒子定位和分类是计算显微学中最基本的问题之一。近年来，深度学习方法在这些任务中取得了巨大成功。这些监督式学习方法的一个关键缺点是它们需要大量的训练数据集，通常是与模拟透射电子显微镜物理的复杂数值正向模型中的粒子模型结合生成的。这些模型的计算机实现非常耗费计算资源，限制了它们的适用范围。本文提出了一种基于加性噪声和神经风格迁移技术模拟电子显微镜正向算子的简单方法。我们使用目前最先进的已经建立的状态之一对定位和分类任务进行评估，显示出与基准测试相当的性能。与以前的方法不同，我们的方法加速了运算，显著减少了计算成本。

    Particle localization and -classification constitute two of the most fundamental problems in computational microscopy. In recent years, deep learning based approaches have been introduced for these tasks with great success. A key shortcoming of these supervised learning methods is their need for large training data sets, typically generated from particle models in conjunction with complex numerical forward models simulating the physics of transmission electron microscopes. Computer implementations of such forward models are computationally extremely demanding and limit the scope of their applicability. In this paper we propose a simple method for simulating the forward operator of an electron microscope based on additive noise and Neural Style Transfer techniques. We evaluate the method on localization and classification tasks using one of the established state-of-the-art architectures showing performance on par with the benchmark. In contrast to previous approaches, our method acceler
    

