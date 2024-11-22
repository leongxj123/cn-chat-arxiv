# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PDE-CNNs: Axiomatic Derivations and Applications](https://arxiv.org/abs/2403.15182) | PDE-CNNs通过利用几何意义的演化PDE的求解器替代传统的组件，提供了更少的参数、固有的等变性、更好的性能、数据效率和几何可解释性。 |
| [^2] | [RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning](https://arxiv.org/abs/2402.08823) | RanDumb是一种简单的方法，通过固定的随机变换嵌入原始像素并学习简单的线性分类器，质疑了持续表示学习的效果。 实验结果显示，RanDumb在众多持续学习基准测试中明显优于使用深度网络进行持续学习的表示学习。 |
| [^3] | [Smart Pressure e-Mat for Human Sleeping Posture and Dynamic Activity Recognition.](http://arxiv.org/abs/2305.11367) | 本文介绍了一种基于Velostat的智能压力电子垫系统，可用于识别人体姿势和运动，具有高精度。 |

# 详细

[^1]: PDE-CNNs：公理推导与应用

    PDE-CNNs: Axiomatic Derivations and Applications

    [https://arxiv.org/abs/2403.15182](https://arxiv.org/abs/2403.15182)

    PDE-CNNs通过利用几何意义的演化PDE的求解器替代传统的组件，提供了更少的参数、固有的等变性、更好的性能、数据效率和几何可解释性。

    

    基于偏微分方程组卷积神经网络（PDE-G-CNNs）利用具有几何意义的演化偏微分方程的求解器替代G-CNNs中常规组件。PDE-G-CNNs同时提供了几个关键优势：更少的参数、固有等变性、更好的性能、数据效率和几何可解释性。本文重点研究特征图在整个网络中为二维的欧几里德等变PDE-G-CNNs。我们将这个框架的变体称为PDE-CNN。我们列出了几个在实践中令人满意的公理，并从中推导出应在PDE-CNN中使用哪些PDE。在这里，我们通过经典线性和形态尺度空间理论的公理受启发，通过引入半域值信号对其进行推广。此外，我们通过实验证实，相对于小型网络，PDE-CNN提供了更少的参数、更好的性能和数据效率。

    arXiv:2403.15182v1 Announce Type: new  Abstract: PDE-based Group Convolutional Neural Networks (PDE-G-CNNs) utilize solvers of geometrically meaningful evolution PDEs as substitutes for the conventional components in G-CNNs. PDE-G-CNNs offer several key benefits all at once: fewer parameters, inherent equivariance, better performance, data efficiency, and geometric interpretability. In this article we focus on Euclidean equivariant PDE-G-CNNs where the feature maps are two dimensional throughout. We call this variant of the framework a PDE-CNN. We list several practically desirable axioms and derive from these which PDEs should be used in a PDE-CNN. Here our approach to geometric learning via PDEs is inspired by the axioms of classical linear and morphological scale-space theory, which we generalize by introducing semifield-valued signals. Furthermore, we experimentally confirm for small networks that PDE-CNNs offer fewer parameters, better performance, and data efficiency in compariso
    
[^2]: RanDumb: 一种质疑持续表示学习效果的简单方法

    RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning

    [https://arxiv.org/abs/2402.08823](https://arxiv.org/abs/2402.08823)

    RanDumb是一种简单的方法，通过固定的随机变换嵌入原始像素并学习简单的线性分类器，质疑了持续表示学习的效果。 实验结果显示，RanDumb在众多持续学习基准测试中明显优于使用深度网络进行持续学习的表示学习。

    

    我们提出了RanDumb来检验持续表示学习的效果。RanDumb将原始像素使用一个固定的随机变换嵌入，这个变换近似了RBF-Kernel，在看到任何数据之前初始化，并学习一个简单的线性分类器。我们提出了一个令人惊讶且一致的发现：在众多持续学习基准测试中，RanDumb在性能上明显优于使用深度网络进行持续学习的表示学习，这表明在这些情景下表示学习的性能较差。RanDumb不存储样本，并在数据上进行单次遍历，一次处理一个样本。它与GDumb相辅相成，在GDumb性能特别差的低样本情况下运行。当将RanDumb扩展到使用预训练模型替换随机变换的情景时，我们得出相同一致的结论。我们的调查结果既令人惊讶又令人担忧，因为表示学习在这些情况下表现糟糕。

    arXiv:2402.08823v1 Announce Type: cross Abstract: We propose RanDumb to examine the efficacy of continual representation learning. RanDumb embeds raw pixels using a fixed random transform which approximates an RBF-Kernel, initialized before seeing any data, and learns a simple linear classifier on top. We present a surprising and consistent finding: RanDumb significantly outperforms the continually learned representations using deep networks across numerous continual learning benchmarks, demonstrating the poor performance of representation learning in these scenarios. RanDumb stores no exemplars and performs a single pass over the data, processing one sample at a time. It complements GDumb, operating in a low-exemplar regime where GDumb has especially poor performance. We reach the same consistent conclusions when RanDumb is extended to scenarios with pretrained models replacing the random transform with pretrained feature extractor. Our investigation is both surprising and alarming as
    
[^3]: 智能压力电子垫用于人类睡眠姿势和动态活动识别

    Smart Pressure e-Mat for Human Sleeping Posture and Dynamic Activity Recognition. (arXiv:2305.11367v1 [cs.CV])

    [http://arxiv.org/abs/2305.11367](http://arxiv.org/abs/2305.11367)

    本文介绍了一种基于Velostat的智能压力电子垫系统，可用于识别人体姿势和运动，具有高精度。

    

    在强调医疗保健、早期教育和健身方面，越来越多的非侵入式测量和识别方法受到关注。压力感应由于其简单的结构、易于访问、可视化应用和无害性而得到广泛研究。本文介绍了一种基于压敏材料Velostat的智能压力电子垫(SP e-Mat)系统，用于人体监测应用，包括睡眠姿势、运动和瑜伽识别。在子系统扫描电子垫读数并处理信号后，它生成一个压力图像流。采用深度神经网络(DNNs)来拟合和训练压力图像流，并识别相应的人类行为。四种睡眠姿势和受Nintendo Switch Ring Fit Adventure(RFA)启发的五种动态活动被用作拟议的SPeM系统的初步验证。SPeM系统在两种应用中均达到了较高的准确性，这证明了其高精度和。

    With the emphasis on healthcare, early childhood education, and fitness, non-invasive measurement and recognition methods have received more attention. Pressure sensing has been extensively studied due to its advantages of simple structure, easy access, visualization application, and harmlessness. This paper introduces a smart pressure e-mat (SPeM) system based on a piezoresistive material Velostat for human monitoring applications, including sleeping postures, sports, and yoga recognition. After a subsystem scans e-mat readings and processes the signal, it generates a pressure image stream. Deep neural networks (DNNs) are used to fit and train the pressure image stream and recognize the corresponding human behavior. Four sleeping postures and five dynamic activities inspired by Nintendo Switch Ring Fit Adventure (RFA) are used as a preliminary validation of the proposed SPeM system. The SPeM system achieves high accuracies on both applications, which demonstrates the high accuracy and
    

