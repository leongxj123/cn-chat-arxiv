# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Expressive Power of a Variant of the Looped Transformer](https://arxiv.org/abs/2402.13572) | 设计了一种新型Transformer块AlgoFormer，相比标准Transformer和Looped Transformer，AlgoFormer在相同参数数量下能够实现更高的算法表达能力 |
| [^2] | [Optimal Transport-inspired Deep Learning Framework for Slow-Decaying Problems: Exploiting Sinkhorn Loss and Wasserstein Kernel.](http://arxiv.org/abs/2308.13840) | 本论文提出了一种将最优传输理论与神经网络结合的新的减小模型（ROM）框架。通过利用Sinkhorn算法进行训练，该框架可以捕捉数据的几何结构，从而实现精确学习减少的解决方案流形。 |

# 详细

[^1]: 论一种变种Looped Transformer的表达能力

    On the Expressive Power of a Variant of the Looped Transformer

    [https://arxiv.org/abs/2402.13572](https://arxiv.org/abs/2402.13572)

    设计了一种新型Transformer块AlgoFormer，相比标准Transformer和Looped Transformer，AlgoFormer在相同参数数量下能够实现更高的算法表达能力

    

    除了自然语言处理，在解决更广泛的应用程序（包括科学计算和计算机视觉）方面，Transformer展现出卓越的性能。先前的工作试图从表达能力和功能性角度解释，标准的Transformer能够执行一些算法。为了赋予Transformer算法能力，并受到最近提出的Looped Transformer的启发，我们设计了一种新颖的Transformer块，名为Algorithm Transformer（简称AlgoFormer）。与标准Transformer和纯粹的Looped Transformer相比，所提出的AlgoFormer在使用相同数量的参数时可以实现更高的算法表示表达能力。特别是，受人类设计的学习算法结构的启发，我们的Transformer块包括一个负责进行ta

    arXiv:2402.13572v1 Announce Type: cross  Abstract: Besides natural language processing, transformers exhibit extraordinary performance in solving broader applications, including scientific computing and computer vision. Previous works try to explain this from the expressive power and capability perspectives that standard transformers are capable of performing some algorithms. To empower transformers with algorithmic capabilities and motivated by the recently proposed looped transformer (Yang et al., 2024; Giannou et al., 2023), we design a novel transformer block, dubbed Algorithm Transformer (abbreviated as AlgoFormer). Compared with the standard transformer and vanilla looped transformer, the proposed AlgoFormer can achieve significantly higher expressiveness in algorithm representation when using the same number of parameters. In particular, inspired by the structure of human-designed learning algorithms, our transformer block consists of a pre-transformer that is responsible for ta
    
[^2]: 受最优传输启发的慢衰减问题的深度学习框架：利用Sinkhorn损失和Wasserstein核

    Optimal Transport-inspired Deep Learning Framework for Slow-Decaying Problems: Exploiting Sinkhorn Loss and Wasserstein Kernel. (arXiv:2308.13840v1 [math.NA])

    [http://arxiv.org/abs/2308.13840](http://arxiv.org/abs/2308.13840)

    本论文提出了一种将最优传输理论与神经网络结合的新的减小模型（ROM）框架。通过利用Sinkhorn算法进行训练，该框架可以捕捉数据的几何结构，从而实现精确学习减少的解决方案流形。

    

    减小模型（ROMs）被广泛用于科学计算中以处理高维系统。然而，传统的ROM方法可能只能部分捕捉到数据的固有几何特征。这些特征包括底层结构、关系和对精确建模至关重要的基本特征。为了克服这个局限性，我们提出了一个将最优传输（OT）理论和基于神经网络的方法相结合的新型ROM框架。具体而言，我们研究了以Wasserstein距离为自定义核的核Proper正交分解（kPOD）方法，并利用Sinkhorn算法高效地训练得到的神经网络（NN）。通过利用基于OT的非线性降维，所提出的框架能够捕捉数据的几何结构，这对于准确学习减少的解决方案流形至关重要。与传统的均方误差或交叉熵等度量标准相比，

    Reduced order models (ROMs) are widely used in scientific computing to tackle high-dimensional systems. However, traditional ROM methods may only partially capture the intrinsic geometric characteristics of the data. These characteristics encompass the underlying structure, relationships, and essential features crucial for accurate modeling.  To overcome this limitation, we propose a novel ROM framework that integrates optimal transport (OT) theory and neural network-based methods. Specifically, we investigate the Kernel Proper Orthogonal Decomposition (kPOD) method exploiting the Wasserstein distance as the custom kernel, and we efficiently train the resulting neural network (NN) employing the Sinkhorn algorithm. By leveraging an OT-based nonlinear reduction, the presented framework can capture the geometric structure of the data, which is crucial for accurate learning of the reduced solution manifold. When compared with traditional metrics such as mean squared error or cross-entropy,
    

