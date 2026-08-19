# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analyzing Male Domestic Violence through Exploratory Data Analysis and Explainable Machine Learning Insights](https://arxiv.org/abs/2403.15594) | 该研究是关于在孟加拉国背景下对男性家庭暴力进行开创性探索，揭示了男性受害者的存在、模式和潜在因素，填补了现有文献对男性受害者研究空白的重要性。 |
| [^2] | [Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket.](http://arxiv.org/abs/2401.02020) | Spikformer V2是一种基于SNNs和自注意机制的脉冲神经网络，通过提出脉冲自注意机制和脉冲Transformer来实现高准确度的图像识别。 |
| [^3] | [Doubly robust nearest neighbors in factor models.](http://arxiv.org/abs/2211.14297) | 该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。 |

# 详细

[^1]: 通过探索性数据分析和可解释的机器学习洞见分析男性家庭暴力

    Analyzing Male Domestic Violence through Exploratory Data Analysis and Explainable Machine Learning Insights

    [https://arxiv.org/abs/2403.15594](https://arxiv.org/abs/2403.15594)

    该研究是关于在孟加拉国背景下对男性家庭暴力进行开创性探索，揭示了男性受害者的存在、模式和潜在因素，填补了现有文献对男性受害者研究空白的重要性。

    

    家庭暴力通常被视为一个关于女性受害者的性别问题，在近年来越来越受到关注。尽管有这种关注，孟加拉国特别是男性受害者仍然主要被忽视。我们的研究代表了在孟加拉国背景下对男性家庭暴力（MDV）这一未被充分探讨领域的开创性探索，揭示了其普遍性、模式和潜在因素。现有文献主要强调家庭暴力情境中女性的受害，导致对男性受害者的研究空白。我们从孟加拉国主要城市收集了数据，并进行了探索性数据分析以了解潜在动态。我们使用了11种传统机器学习模型（包括默认和优化的超参数）、2种深度学习和4种集成模型。尽管采用了各种方法，CatBoost由于其...

    arXiv:2403.15594v1 Announce Type: cross  Abstract: Domestic violence, which is often perceived as a gendered issue among female victims, has gained increasing attention in recent years. Despite this focus, male victims of domestic abuse remain primarily overlooked, particularly in Bangladesh. Our study represents a pioneering exploration of the underexplored realm of male domestic violence (MDV) within the Bangladeshi context, shedding light on its prevalence, patterns, and underlying factors. Existing literature predominantly emphasizes female victimization in domestic violence scenarios, leading to an absence of research on male victims. We collected data from the major cities of Bangladesh and conducted exploratory data analysis to understand the underlying dynamics. We implemented 11 traditional machine learning models with default and optimized hyperparameters, 2 deep learning, and 4 ensemble models. Despite various approaches, CatBoost has emerged as the top performer due to its 
    
[^2]: Spikformer V2：通过SNN Ticket在ImageNet上实现高准确度

    Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket. (arXiv:2401.02020v1 [cs.NE])

    [http://arxiv.org/abs/2401.02020](http://arxiv.org/abs/2401.02020)

    Spikformer V2是一种基于SNNs和自注意机制的脉冲神经网络，通过提出脉冲自注意机制和脉冲Transformer来实现高准确度的图像识别。

    

    脉冲神经网络（SNNs）因其生物学合理的结构而闻名，但其性能受到限制。基于生物启发结构的高性能Transformer中的自注意机制在现有的SNNs中缺失。为此，我们探索了利用自注意能力和SNNs的生物特性的潜力，并提出了一种新颖的脉冲自注意（SSA）和脉冲Transformer（Spikformer）。SSA机制消除了对softmax的需求，并利用基于脉冲的查询、键和值捕获稀疏的视觉特征。这种无乘法的稀疏计算使得SSA高效且节能。此外，我们还开发了一种脉冲卷积干细胞（SCS）和补充卷积层来增强Spikformer的架构。加上SCS的Spikformer被称为Spikformer V2。为了训练更大更深的Spikformer V2，我们引入了一种开创性的探+

    Spiking Neural Networks (SNNs), known for their biologically plausible architecture, face the challenge of limited performance. The self-attention mechanism, which is the cornerstone of the high-performance Transformer and also a biologically inspired structure, is absent in existing SNNs. To this end, we explore the potential of leveraging both self-attention capability and biological properties of SNNs, and propose a novel Spiking Self-Attention (SSA) and Spiking Transformer (Spikformer). The SSA mechanism eliminates the need for softmax and captures the sparse visual feature employing spike-based Query, Key, and Value. This sparse computation without multiplication makes SSA efficient and energy-saving. Further, we develop a Spiking Convolutional Stem (SCS) with supplementary convolutional layers to enhance the architecture of Spikformer. The Spikformer enhanced with the SCS is referred to as Spikformer V2. To train larger and deeper Spikformer V2, we introduce a pioneering explorat
    
[^3]: 因子模型中的双重稳健最近邻方法

    Doubly robust nearest neighbors in factor models. (arXiv:2211.14297v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14297](http://arxiv.org/abs/2211.14297)

    该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。

    

    我们介绍并分析了在潜在因子模型中处理缺失数据的改进最近邻（NN）方法。我们考虑一个带有缺失数据的矩阵补全问题，其中当被观察到时，第$(i, t)$个条目由其均值$f(u_i, v_t)$加上均值为零的噪声给出，其中$f$为未知函数，$u_i$和$v_t$为潜在因子。之前的NN策略，如单元-单元NN，用于估计均值$f(u_i, v_t)$，依赖于存在其他行$j$使得$u_j \approx u_i$。类似地，时间-时间NN策略依赖于存在列$t'$使得$v_{t'} \approx v_t$。当相似行或相似列不可用时，这些策略的性能较差。我们的估计在两个方面对这种不足是双重稳健的：(1) 只要存在良好的行或列邻居，我们的估计提供一致的估计。 (2) 此外，如果存在良好的行和列邻居，它提供了（近似）二次改进非渐近性能。

    We introduce and analyze an improved variant of nearest neighbors (NN) for estimation with missing data in latent factor models. We consider a matrix completion problem with missing data, where the $(i, t)$-th entry, when observed, is given by its mean $f(u_i, v_t)$ plus mean-zero noise for an unknown function $f$ and latent factors $u_i$ and $v_t$. Prior NN strategies, like unit-unit NN, for estimating the mean $f(u_i, v_t)$ relies on existence of other rows $j$ with $u_j \approx u_i$. Similarly, time-time NN strategy relies on existence of columns $t'$ with $v_{t'} \approx v_t$. These strategies provide poor performance respectively when similar rows or similar columns are not available. Our estimate is doubly robust to this deficit in two ways: (1) As long as there exist either good row or good column neighbors, our estimate provides a consistent estimate. (2) Furthermore, if both good row and good column neighbors exist, it provides a (near-)quadratic improvement in the non-asympto
    

