# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation](https://rss.arxiv.org/abs/2402.01542) | 本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。 |

# 详细

[^1]: 使用标记数据增强的测地插值方法学习蛋白质折叠的集体变量

    Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation

    [https://rss.arxiv.org/abs/2402.01542](https://rss.arxiv.org/abs/2402.01542)

    本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。

    

    在分子动力学（MD）模拟中，通常通过增强采样技术来研究蛋白质折叠等罕见事件，其中大部分依赖于沿着加速发生的集体变量（CV）的定义。获得富有表达力的CV至关重要，但往往受到关于特定事件的信息不足的阻碍，例如从未折叠到折叠构象的转变。我们提出了一种模拟无关的数据增强策略，利用受物理启发的度量来生成类似蛋白质折叠转变的测地插值，从而提高采样效率，而无需真实的过渡态样本。通过利用插值进度参数，我们引入了基于回归的学习方案来构建CV模型，当过渡态数据有限且嘈杂时，该方法表现优于基于分类器的方法。

    In molecular dynamics (MD) simulations, rare events, such as protein folding, are typically studied by means of enhanced sampling techniques, most of which rely on the definition of a collective variable (CV) along which the acceleration occurs. Obtaining an expressive CV is crucial, but often hindered by the lack of information about the particular event, e.g., the transition from unfolded to folded conformation. We propose a simulation-free data augmentation strategy using physics-inspired metrics to generate geodesic interpolations resembling protein folding transitions, thereby improving sampling efficiency without true transition state samples. Leveraging interpolation progress parameters, we introduce a regression-based learning scheme for CV models, which outperforms classifier-based methods when transition state data is limited and noisy
    

