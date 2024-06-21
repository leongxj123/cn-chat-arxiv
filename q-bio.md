# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation](https://rss.arxiv.org/abs/2402.01542) | 本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。 |
| [^2] | [Multi-intention Inverse Q-learning for Interpretable Behavior Representation](https://rss.arxiv.org/abs/2311.13870) | 本研究引入了一种多意图逆Q学习算法，用于解决在推断离散时变奖励时的挑战。通过聚类观察到的专家轨迹并独立解决每个意图的逆强化学习问题，我们的方法在动物行为预测方面超越了当前的基准，产生了可解释的奖励函数。 |
| [^3] | [Prospector Heads: Generalized Feature Attribution for Large Models & Data](https://arxiv.org/abs/2402.11729) | Prospector heads是一种高效且可解释的基于特征归因的替代方法，它可以应用于任何编码器和任何数据形态，并且通过对不同数据形态的实验，表现优越于传统方法。 |
| [^4] | [Compressed representation of brain genetic transcription.](http://arxiv.org/abs/2310.16113) | 本文研究了大脑基因转录的压缩表示方法，通过比较不同的线性和非线性方法，评估了它们在重建、解剖和预测方面的性能。 |
| [^5] | [RFold: RNA Secondary Structure Prediction with Decoupled Optimization.](http://arxiv.org/abs/2212.14041) | 所提出的RFold方法采用解耦优化过程和注意力机制进行简单又有效的RNA二级结构预测，具有较高的准确性和速度。 |

# 详细

[^1]: 使用标记数据增强的测地插值方法学习蛋白质折叠的集体变量

    Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation

    [https://rss.arxiv.org/abs/2402.01542](https://rss.arxiv.org/abs/2402.01542)

    本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。

    

    在分子动力学（MD）模拟中，通常通过增强采样技术来研究蛋白质折叠等罕见事件，其中大部分依赖于沿着加速发生的集体变量（CV）的定义。获得富有表达力的CV至关重要，但往往受到关于特定事件的信息不足的阻碍，例如从未折叠到折叠构象的转变。我们提出了一种模拟无关的数据增强策略，利用受物理启发的度量来生成类似蛋白质折叠转变的测地插值，从而提高采样效率，而无需真实的过渡态样本。通过利用插值进度参数，我们引入了基于回归的学习方案来构建CV模型，当过渡态数据有限且嘈杂时，该方法表现优于基于分类器的方法。

    In molecular dynamics (MD) simulations, rare events, such as protein folding, are typically studied by means of enhanced sampling techniques, most of which rely on the definition of a collective variable (CV) along which the acceleration occurs. Obtaining an expressive CV is crucial, but often hindered by the lack of information about the particular event, e.g., the transition from unfolded to folded conformation. We propose a simulation-free data augmentation strategy using physics-inspired metrics to generate geodesic interpolations resembling protein folding transitions, thereby improving sampling efficiency without true transition state samples. Leveraging interpolation progress parameters, we introduce a regression-based learning scheme for CV models, which outperforms classifier-based methods when transition state data is limited and noisy
    
[^2]: 可解释行为表示的多意图逆Q学习

    Multi-intention Inverse Q-learning for Interpretable Behavior Representation

    [https://rss.arxiv.org/abs/2311.13870](https://rss.arxiv.org/abs/2311.13870)

    本研究引入了一种多意图逆Q学习算法，用于解决在推断离散时变奖励时的挑战。通过聚类观察到的专家轨迹并独立解决每个意图的逆强化学习问题，我们的方法在动物行为预测方面超越了当前的基准，产生了可解释的奖励函数。

    

    在推动决策过程理解方面，逆强化学习（IRL）在重构动物复杂行为中的多个意图方面证明了其重要性。鉴于最近发展的连续时间多意图IRL框架，人们一直在研究如何使用IRL推断离散的时变奖励。为了解决这个挑战，我们引入了潜（马尔科夫）变量逆Q学习（L(M)V-IQL），这是一种专门用于适应离散内在奖励函数的IRL算法。通过利用期望最大化方法，我们将观察到的专家轨迹聚类成不同的意图，并为每个意图独立解决IRL问题。通过模拟实验和对不同真实鼠类行为数据集的应用，我们的方法在动物行为预测方面超越了当前的基准，产生了可解释的奖励函数。这一进展有望打开推动科学与工程应用的新机遇。

    In advancing the understanding of decision-making processes, Inverse Reinforcement Learning (IRL) have proven instrumental in reconstructing animal's multiple intentions amidst complex behaviors. Given the recent development of a continuous-time multi-intention IRL framework, there has been persistent inquiry into inferring discrete time-varying rewards with IRL. To tackle the challenge, we introduce Latent (Markov) Variable Inverse Q-learning (L(M)V-IQL), a novel class of IRL algorthms tailored for accommodating discrete intrinsic reward functions. Leveraging an Expectation-Maximization approach, we cluster observed expert trajectories into distinct intentions and independently solve the IRL problem for each. Demonstrating the efficacy of L(M)V-IQL through simulated experiments and its application to different real mouse behavior datasets, our approach surpasses current benchmarks in animal behavior prediction, producing interpretable reward functions. This advancement holds promise f
    
[^3]: Prospector Heads:大规模模型和数据的广义特征归因

    Prospector Heads: Generalized Feature Attribution for Large Models & Data

    [https://arxiv.org/abs/2402.11729](https://arxiv.org/abs/2402.11729)

    Prospector heads是一种高效且可解释的基于特征归因的替代方法，它可以应用于任何编码器和任何数据形态，并且通过对不同数据形态的实验，表现优越于传统方法。

    

    特征归因是一种定位输入数据中与分类相关的区域的能力，对于科学和生物医学领域的机器学习模型而言，这是一种重要的能力。当前的特征归因方法依赖于“解释”端到端分类器的预测，存在特征定位不精确以及由于计算挑战而无法在小样本尺寸和高维数据集上使用的问题。我们引入了探寻者头部（prospector heads），这是一种高效且可解释的基于特征归因的替代方法，可以应用于任何编码器和任何数据形态。通过对序列（文本）、图像（病理学）和图（蛋白质结构）的实验，探寻者头部在模态之间进行概括，表现优于基线归因方法，平均局部化AUPRC得分提升了高达49点。我们还演示了探寻者头部如何实现了改进的解释能力。

    arXiv:2402.11729v1 Announce Type: cross  Abstract: Feature attribution, the ability to localize regions of the input data that are relevant for classification, is an important capability for machine learning models in scientific and biomedical domains. Current methods for feature attribution, which rely on "explaining" the predictions of end-to-end classifiers, suffer from imprecise feature localization and are inadequate for use with small sample sizes and high-dimensional datasets due to computational challenges. We introduce prospector heads, an efficient and interpretable alternative to explanation-based methods for feature attribution that can be applied to any encoder and any data modality. Prospector heads generalize across modalities through experiments on sequences (text), images (pathology), and graphs (protein structures), outperforming baseline attribution methods by up to 49 points in mean localization AUPRC. We also demonstrate how prospector heads enable improved interpr
    
[^4]: 大脑基因转录的压缩表示

    Compressed representation of brain genetic transcription. (arXiv:2310.16113v1 [cs.LG])

    [http://arxiv.org/abs/2310.16113](http://arxiv.org/abs/2310.16113)

    本文研究了大脑基因转录的压缩表示方法，通过比较不同的线性和非线性方法，评估了它们在重建、解剖和预测方面的性能。

    

    大脑的结构过于复杂，无法直观地进行观察，需要使用压缩表示将其变化投影到紧凑、可导航的空间中。在高维数据（如基因表达）中，尤其具有挑战性，其中解剖和转录模式的联合复杂性要求最大压缩。目前的实践是使用标准的主成分分析（PCA），其计算效率受到限制，尤其在大压缩比下表现力有限。本研究利用全脑体素级Allen大脑图谱转录数据，系统比较了基于最广泛支持的线性和非线性方法（PCA，核PCA，非负矩阵分解（NMF），t-随机邻居嵌入（t-SNE），统一流形逼近和投影（UMAP），深度自编码）的压缩表示，量化重建保真度，解剖连贯性和预测效果。

    The architecture of the brain is too complex to be intuitively surveyable without the use of compressed representations that project its variation into a compact, navigable space. The task is especially challenging with high-dimensional data, such as gene expression, where the joint complexity of anatomical and transcriptional patterns demands maximum compression. Established practice is to use standard principal component analysis (PCA), whose computational felicity is offset by limited expressivity, especially at great compression ratios. Employing whole-brain, voxel-wise Allen Brain Atlas transcription data, here we systematically compare compressed representations based on the most widely supported linear and non-linear methods-PCA, kernel PCA, non-negative matrix factorization (NMF), t-stochastic neighbour embedding (t-SNE), uniform manifold approximation and projection (UMAP), and deep auto-encoding-quantifying reconstruction fidelity, anatomical coherence, and predictive utility
    
[^5]: RFold：基于解耦优化方法的RNA二级结构预测

    RFold: RNA Secondary Structure Prediction with Decoupled Optimization. (arXiv:2212.14041v2 [q-bio.BM] UPDATED)

    [http://arxiv.org/abs/2212.14041](http://arxiv.org/abs/2212.14041)

    所提出的RFold方法采用解耦优化过程和注意力机制进行简单又有效的RNA二级结构预测，具有较高的准确性和速度。

    

    核糖核酸（RNA）的二级结构比三级结构更稳定和更易于在细胞中访问，因此对于功能预测至关重要。尽管深度学习在这个领域中显示出了很好的结果，但当前的方法存在泛化性差和复杂性高的问题。在这项工作中，我们提出了一种简单而有效的RNA二级结构预测方法RFold。RFold引入了一种解耦优化的过程，将传统的约束满足问题分解为逐行和逐列优化，简化了求解过程，同时保证了输出的有效性。此外，RFold采用注意力地图作为信息表示，而不是设计手工特征。广泛的实验表明，RFold具有竞争性能，并且比现有最先进的方法具有约8倍的推理效率。代码和Colab演示可在\href{this http URL}{this http UR}上找到。

    The secondary structure of ribonucleic acid (RNA) is more stable and accessible in the cell than its tertiary structure, making it essential for functional prediction. Although deep learning has shown promising results in this field, current methods suffer from poor generalization and high complexity. In this work, we present RFold, a simple yet effective RNA secondary structure prediction in an end-to-end manner. RFold introduces a decoupled optimization process that decomposes the vanilla constraint satisfaction problem into row-wise and column-wise optimization, simplifying the solving process while guaranteeing the validity of the output. Moreover, RFold adopts attention maps as informative representations instead of designing hand-crafted features. Extensive experiments demonstrate that RFold achieves competitive performance and about eight times faster inference efficiency than the state-of-the-art method. The code and Colab demo are available in \href{this http URL}{this http UR
    

