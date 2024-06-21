# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation](https://rss.arxiv.org/abs/2402.01542) | 本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。 |
| [^2] | [RFold: RNA Secondary Structure Prediction with Decoupled Optimization.](http://arxiv.org/abs/2212.14041) | 所提出的RFold方法采用解耦优化过程和注意力机制进行简单又有效的RNA二级结构预测，具有较高的准确性和速度。 |

# 详细

[^1]: 使用标记数据增强的测地插值方法学习蛋白质折叠的集体变量

    Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation

    [https://rss.arxiv.org/abs/2402.01542](https://rss.arxiv.org/abs/2402.01542)

    本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。

    

    在分子动力学（MD）模拟中，通常通过增强采样技术来研究蛋白质折叠等罕见事件，其中大部分依赖于沿着加速发生的集体变量（CV）的定义。获得富有表达力的CV至关重要，但往往受到关于特定事件的信息不足的阻碍，例如从未折叠到折叠构象的转变。我们提出了一种模拟无关的数据增强策略，利用受物理启发的度量来生成类似蛋白质折叠转变的测地插值，从而提高采样效率，而无需真实的过渡态样本。通过利用插值进度参数，我们引入了基于回归的学习方案来构建CV模型，当过渡态数据有限且嘈杂时，该方法表现优于基于分类器的方法。

    In molecular dynamics (MD) simulations, rare events, such as protein folding, are typically studied by means of enhanced sampling techniques, most of which rely on the definition of a collective variable (CV) along which the acceleration occurs. Obtaining an expressive CV is crucial, but often hindered by the lack of information about the particular event, e.g., the transition from unfolded to folded conformation. We propose a simulation-free data augmentation strategy using physics-inspired metrics to generate geodesic interpolations resembling protein folding transitions, thereby improving sampling efficiency without true transition state samples. Leveraging interpolation progress parameters, we introduce a regression-based learning scheme for CV models, which outperforms classifier-based methods when transition state data is limited and noisy
    
[^2]: RFold：基于解耦优化方法的RNA二级结构预测

    RFold: RNA Secondary Structure Prediction with Decoupled Optimization. (arXiv:2212.14041v2 [q-bio.BM] UPDATED)

    [http://arxiv.org/abs/2212.14041](http://arxiv.org/abs/2212.14041)

    所提出的RFold方法采用解耦优化过程和注意力机制进行简单又有效的RNA二级结构预测，具有较高的准确性和速度。

    

    核糖核酸（RNA）的二级结构比三级结构更稳定和更易于在细胞中访问，因此对于功能预测至关重要。尽管深度学习在这个领域中显示出了很好的结果，但当前的方法存在泛化性差和复杂性高的问题。在这项工作中，我们提出了一种简单而有效的RNA二级结构预测方法RFold。RFold引入了一种解耦优化的过程，将传统的约束满足问题分解为逐行和逐列优化，简化了求解过程，同时保证了输出的有效性。此外，RFold采用注意力地图作为信息表示，而不是设计手工特征。广泛的实验表明，RFold具有竞争性能，并且比现有最先进的方法具有约8倍的推理效率。代码和Colab演示可在\href{this http URL}{this http UR}上找到。

    The secondary structure of ribonucleic acid (RNA) is more stable and accessible in the cell than its tertiary structure, making it essential for functional prediction. Although deep learning has shown promising results in this field, current methods suffer from poor generalization and high complexity. In this work, we present RFold, a simple yet effective RNA secondary structure prediction in an end-to-end manner. RFold introduces a decoupled optimization process that decomposes the vanilla constraint satisfaction problem into row-wise and column-wise optimization, simplifying the solving process while guaranteeing the validity of the output. Moreover, RFold adopts attention maps as informative representations instead of designing hand-crafted features. Extensive experiments demonstrate that RFold achieves competitive performance and about eight times faster inference efficiency than the state-of-the-art method. The code and Colab demo are available in \href{this http URL}{this http UR
    

