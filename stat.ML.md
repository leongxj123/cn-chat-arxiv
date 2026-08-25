# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Clustering Evaluation: How to Validate Internal Clustering Validation Measures](https://arxiv.org/abs/2403.14830) | 本文解决了深度聚类方法在评估聚类质量时面临的挑战，提出了一种系统方法来应用聚类有效性指标。 |
| [^2] | [Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects.](http://arxiv.org/abs/2310.08115) | 提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。 |
| [^3] | [Nested Elimination: A Simple Algorithm for Best-Item Identification from Choice-Based Feedback.](http://arxiv.org/abs/2307.09295) | 嵌套消除是一种简单易实现的算法，通过利用创新的消除准则和嵌套结构，能够以最少的样本数量和高置信水平识别出最受欢迎的项目。 |
| [^4] | [Neuro-Causal Factor Analysis.](http://arxiv.org/abs/2305.19802) | 该论文提出了一种名为神经因果因素分析（NCFA）的新方法，它通过学习到的图形匹配马尔可夫因式分解的分布来识别因素，并使用变分自编码器（VAE）对数据进行重建任务。与标准VAE相比，NCFA具有更稀疏的架构和低模型复杂度，具有因果解释性。 |
| [^5] | [Signal identification without signal formulation.](http://arxiv.org/abs/2304.06522) | 该研究提出了一种无需信号建模即可识别信号的方法，该方法基于样本和其邻居之间相对距离，可以在小样本和高维数据中识别“类似于信号”的变量。 |

# 详细

[^1]: 深度聚类评估：如何验证内部聚类有效性测量方法

    Deep Clustering Evaluation: How to Validate Internal Clustering Validation Measures

    [https://arxiv.org/abs/2403.14830](https://arxiv.org/abs/2403.14830)

    本文解决了深度聚类方法在评估聚类质量时面临的挑战，提出了一种系统方法来应用聚类有效性指标。

    

    arXiv:2403.14830v1 通告类型：跨领域 摘要：深度聚类是一种使用深度神经网络对复杂、高维数据进行划分的方法，它面临着独特的评估挑战。传统的聚类验证方法，设计用于低维空间，对于涉及将数据投影到较低维嵌入空间后再进行划分的深度聚类来说是有问题的。论文确定了两个关键问题：1）在将这些方法应用于原始数据时的维度灾难，2）由于不同聚类模型的训练过程和参数设置的变化而导致不同嵌入空间中的聚类结果无法可靠比较。本文解决了在深度学习中评估聚类质量所面临的挑战。我们提出了一个理论框架来强调在原始数据和嵌入数据上使用内部验证方法可能出现的无效性，并提出了一种系统方法来应用深度聚类有效性指标。

    arXiv:2403.14830v1 Announce Type: cross  Abstract: Deep clustering, a method for partitioning complex, high-dimensional data using deep neural networks, presents unique evaluation challenges. Traditional clustering validation measures, designed for low-dimensional spaces, are problematic for deep clustering, which involves projecting data into lower-dimensional embeddings before partitioning. Two key issues are identified: 1) the curse of dimensionality when applying these measures to raw data, and 2) the unreliable comparison of clustering results across different embedding spaces stemming from variations in training procedures and parameter settings in different clustering models. This paper addresses these challenges in evaluating clustering quality in deep learning. We present a theoretical framework to highlight ineffectiveness arising from using internal validation measures on raw and embedded data and propose a systematic approach to applying clustering validity indices in deep 
    
[^2]: 模型不可知的辅助推断方法在部分可辨识因果效应上的应用

    Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects. (arXiv:2310.08115v1 [econ.EM])

    [http://arxiv.org/abs/2310.08115](http://arxiv.org/abs/2310.08115)

    提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。

    

    很多因果估计是部分可辨识的，因为它们依赖于潜在结果之间的不可观察联合分布。基于前处理协变量的分层可以获得更明确的部分可辨识性范围；然而，除非协变量为离散且支撑度相对较小，否则这种方法通常需要对给定协变量的潜在结果的条件分布进行一致估计。因此，现有的方法在模型错误或一致性假设被违反时可能失败。在本研究中，我们提出了一种基于最优输运问题的对偶理论的统一且模型不可知的推断方法，适用于广泛类别的部分可辨识估计。在随机实验中，我们的方法可以结合任何对条件分布的估计，并提供统一有效的推断，即使初始估计是任意不准确的。此外，我们的方法在观测研究中也是双重鲁棒的。

    Many causal estimands are only partially identifiable since they depend on the unobservable joint distribution between potential outcomes. Stratification on pretreatment covariates can yield sharper partial identification bounds; however, unless the covariates are discrete with relatively small support, this approach typically requires consistent estimation of the conditional distributions of the potential outcomes given the covariates. Thus, existing approaches may fail under model misspecification or if consistency assumptions are violated. In this study, we propose a unified and model-agnostic inferential approach for a wide class of partially identified estimands, based on duality theory for optimal transport problems. In randomized experiments, our approach can wrap around any estimates of the conditional distributions and provide uniformly valid inference, even if the initial estimates are arbitrarily inaccurate. Also, our approach is doubly robust in observational studies. Notab
    
[^3]: 嵌套消除：一种从基于选择的反馈中识别最佳项目的简单算法

    Nested Elimination: A Simple Algorithm for Best-Item Identification from Choice-Based Feedback. (arXiv:2307.09295v1 [cs.LG])

    [http://arxiv.org/abs/2307.09295](http://arxiv.org/abs/2307.09295)

    嵌套消除是一种简单易实现的算法，通过利用创新的消除准则和嵌套结构，能够以最少的样本数量和高置信水平识别出最受欢迎的项目。

    

    我们研究了基于选择的反馈中识别最佳项目的问题。在这个问题中，公司依次向一群顾客展示显示集，并收集他们的选择。目标是以最少的样本数量和高置信水平识别出最受欢迎的项目。我们提出了一种基于消除的算法，即嵌套消除(Nested Elimination，NE)，它受到信息理论下界所暗示的嵌套结构的启发。NE的结构简单，易于实施，具有对样本复杂度的强大理论保证。具体而言，NE利用了一种创新的消除准则，并避免了解决任何复杂的组合优化问题的需要。我们提供了NE的特定实例和非渐近性的样本复杂度的上界。我们还展示了NE实现了高阶最坏情况渐近最优性。最后，来自合成和真实数据的数值实验验证了我们的理论。

    We study the problem of best-item identification from choice-based feedback. In this problem, a company sequentially and adaptively shows display sets to a population of customers and collects their choices. The objective is to identify the most preferred item with the least number of samples and at a high confidence level. We propose an elimination-based algorithm, namely Nested Elimination (NE), which is inspired by the nested structure implied by the information-theoretic lower bound. NE is simple in structure, easy to implement, and has a strong theoretical guarantee for sample complexity. Specifically, NE utilizes an innovative elimination criterion and circumvents the need to solve any complex combinatorial optimization problem. We provide an instance-specific and non-asymptotic bound on the expected sample complexity of NE. We also show NE achieves high-order worst-case asymptotic optimality. Finally, numerical experiments from both synthetic and real data corroborate our theore
    
[^4]: 神经因果因素分析

    Neuro-Causal Factor Analysis. (arXiv:2305.19802v1 [stat.ML])

    [http://arxiv.org/abs/2305.19802](http://arxiv.org/abs/2305.19802)

    该论文提出了一种名为神经因果因素分析（NCFA）的新方法，它通过学习到的图形匹配马尔可夫因式分解的分布来识别因素，并使用变分自编码器（VAE）对数据进行重建任务。与标准VAE相比，NCFA具有更稀疏的架构和低模型复杂度，具有因果解释性。

    

    因素分析是一种通过研究带有一些相互依赖关系的观察变量可以如何表示为相互独立的未观察因素的函数的统计工具，并广泛应用于心理学、生物学和物理科学领域。我们从因果发现和深度学习的新视角重新审视这种经典方法，引入了神经因果因素分析（NCFA）的框架。我们的方法是完全非参数的：它通过潜在的因果发现方法识别因素，然后使用变分自编码器（VAE），该VAE受到与学习图的关于马尔可夫因式分解的分布相符的限制。我们评估了NCFA在真实的和合成的数据集上，发现它在数据重建任务上的表现与标准VAE相当，但具有更稀疏的架构、更低的模型复杂度和因果可解释性。与传统的FA方法不同，我们提出的NCFA方法可以通过学习到的图形表示因素之间的因果关系，从而具有因果解释性。

    Factor analysis (FA) is a statistical tool for studying how observed variables with some mutual dependences can be expressed as functions of mutually independent unobserved factors, and it is widely applied throughout the psychological, biological, and physical sciences. We revisit this classic method from the comparatively new perspective given by advancements in causal discovery and deep learning, introducing a framework for Neuro-Causal Factor Analysis (NCFA). Our approach is fully nonparametric: it identifies factors via latent causal discovery methods and then uses a variational autoencoder (VAE) that is constrained to abide by the Markov factorization of the distribution with respect to the learned graph. We evaluate NCFA on real and synthetic data sets, finding that it performs comparably to standard VAEs on data reconstruction tasks but with the advantages of sparser architecture, lower model complexity, and causal interpretability. Unlike traditional FA methods, our proposed N
    
[^5]: 无需信号建模的信号识别方法

    Signal identification without signal formulation. (arXiv:2304.06522v1 [physics.data-an])

    [http://arxiv.org/abs/2304.06522](http://arxiv.org/abs/2304.06522)

    该研究提出了一种无需信号建模即可识别信号的方法，该方法基于样本和其邻居之间相对距离，可以在小样本和高维数据中识别“类似于信号”的变量。

    

    当信号和噪声混合时，物理学家通常通过信号建模来识别信号，而统计学家则相反，他们试图对噪声进行建模来识别信号。在本研究中，我们应用了统计学家的信号检测概念，对具有小样本和高维数据的物理数据进行了处理，而不对信号进行建模。自然界中的大部分数据，无论是噪声还是信号，都被假定为是由动态系统生成的；因此，在这些生成过程之间基本上没有区别。我们提出了动态系统的相关长度和样本数对于在这样的系统中生成的信号变量中区分噪声变量的实际定义至关重要。由于具有短期相关性的变量随着样本数的减少会更快地达到正态分布，因此它们被认为是“类似于噪声”的变量，而具有相反特性的变量则是“类似于信号”的变量。正态性检验不适用于小样本和高维数据，因此我们提出了一种基于样本和其邻居之间相对距离的新方法来识别“类似于噪声”的变量。实验证明，所提出的方法可以在不进行任何信号建模的情况下识别“类似于信号”的变量。

    When there are signals and noises, physicists try to identify signals by modeling them, whereas statisticians oppositely try to model noise to identify signals. In this study, we applied the statisticians' concept of signal detection of physics data with small-size samples and high dimensions without modeling the signals. Most of the data in nature, whether noises or signals, are assumed to be generated by dynamical systems; thus, there is essentially no distinction between these generating processes. We propose that the correlation length of a dynamical system and the number of samples are crucial for the practical definition of noise variables among the signal variables generated by such a system. Since variables with short-term correlations reach normal distributions faster as the number of samples decreases, they are regarded to be ``noise-like'' variables, whereas variables with opposite properties are ``signal-like'' variables. Normality tests are not effective for data of small-
    

