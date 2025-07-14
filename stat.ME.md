# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Signed Diverse Multiplex Networks: Clustering and Inference](https://arxiv.org/abs/2402.10242) | 保留边的符号在网络构建过程中提高了估计和聚类精度，有助于解决现实世界问题。 |
| [^2] | [A step towards the integration of machine learning and small area estimation](https://arxiv.org/abs/2402.07521) | 本文提出了一个基于机器学习算法的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征，并分析了在实际生活中更重要的背景下的性能。 |
| [^3] | [Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach](https://arxiv.org/abs/2402.02672) | 本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。 |
| [^4] | [Nonparametric Treatment Effect Identification in School Choice.](http://arxiv.org/abs/2112.03872) | 本论文研究了集中学校分配中的非参数化治疗效果识别和估计方法，通过识别原子治疗效应，该研究揭示了在回归不连续和抽签驱动的变异下的学校选择的异质性和重要性。 |

# 详细

[^1]: 有符号多样化多重网络：聚类和推断

    Signed Diverse Multiplex Networks: Clustering and Inference

    [https://arxiv.org/abs/2402.10242](https://arxiv.org/abs/2402.10242)

    保留边的符号在网络构建过程中提高了估计和聚类精度，有助于解决现实世界问题。

    

    该论文介绍了一种有符号的广义随机点积图（SGRDPG）模型，这是广义随机点积图（GRDPG）的一个变种，其中边可以是正的也可以是负的。该设置被扩展为多重网络版本，其中所有层具有相同的节点集合并遵循SGRDPG。网络层的唯一公共特征是它们可以被划分为具有共同子空间结构的组，而其他情况下所有连接概率矩阵可能是完全不同的。上述设置非常灵活，并包括各种现有多重网络模型作为其特例。论文实现了两个目标。首先，它表明在网络构建过程中保留边的符号会导致更好的估计和聚类精度，因此有助于应对诸如大脑网络分析之类的现实问题。

    arXiv:2402.10242v1 Announce Type: cross  Abstract: The paper introduces a Signed Generalized Random Dot Product Graph (SGRDPG) model, which is a variant of the Generalized Random Dot Product Graph (GRDPG), where, in addition, edges can be positive or negative. The setting is extended to a multiplex version, where all layers have the same collection of nodes and follow the SGRDPG. The only common feature of the layers of the network is that they can be partitioned into groups with common subspace structures, while otherwise all matrices of connection probabilities can be all different. The setting above is extremely flexible and includes a variety of existing multiplex network models as its particular cases. The paper fulfills two objectives. First, it shows that keeping signs of the edges in the process of network construction leads to a better precision of estimation and clustering and, hence, is beneficial for tackling real world problems such as analysis of brain networks. Second, b
    
[^2]: 机器学习与小区域估计的整合步骤

    A step towards the integration of machine learning and small area estimation

    [https://arxiv.org/abs/2402.07521](https://arxiv.org/abs/2402.07521)

    本文提出了一个基于机器学习算法的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征，并分析了在实际生活中更重要的背景下的性能。

    

    机器学习技术的应用已经在许多研究领域得到了发展。目前，在统计学中，包括正式统计学在内，也广泛应用于数据收集（如卫星图像、网络爬取和文本挖掘、数据清洗、集成和插补）以及数据分析。然而，在调查抽样包括小区域估计方面，这些方法的使用仍然非常有限。因此，我们提出一个由这些算法支持的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征。机器学习方法已经显示出在识别和建模变量之间复杂和非线性关系方面非常强大，这意味着在强烈偏离经典假设的情况下，它们具有非常好的性能。因此，我们分析了我们的模型在一种不同的背景下的表现，这个背景在我们看来在实际生活中更重要。

    The use of machine-learning techniques has grown in numerous research areas. Currently, it is also widely used in statistics, including the official statistics for data collection (e.g. satellite imagery, web scraping and text mining, data cleaning, integration and imputation) but also for data analysis. However, the usage of these methods in survey sampling including small area estimation is still very limited. Therefore, we propose a predictor supported by these algorithms which can be used to predict any population or subpopulation characteristics based on cross-sectional and longitudinal data. Machine learning methods have already been shown to be very powerful in identifying and modelling complex and nonlinear relationships between the variables, which means that they have very good properties in case of strong departures from the classic assumptions. Therefore, we analyse the performance of our proposal under a different set-up, in our opinion of greater importance in real-life s
    
[^3]: 对分布式数据的条件平均治疗效果估计：一种保护隐私的方法

    Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach

    [https://arxiv.org/abs/2402.02672](https://arxiv.org/abs/2402.02672)

    本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。

    

    在医学和社会科学等各个领域中，对条件平均治疗效果（CATEs）的估计是一个重要的课题。如果分布在多个参与方之间的数据可以集中，可以对CATEs进行高精度的估计。然而，如果这些数据包含隐私信息，则很难进行数据聚合。为了解决这个问题，我们提出了数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计CATE模型，并通过数值实验对该方法进行了评估。我们的贡献总结如下三点。首先，我们的方法能够在分布式数据上进行非迭代通信的半参数CATE模型的估计和测试。半参数或非参数的CATE模型能够比参数模型更稳健地进行估计和测试，对于模型偏差的鲁棒性更强。然而，据我们所知，目前还没有提出有效的通信方法来估计和测试这些模型。

    Estimation of conditional average treatment effects (CATEs) is an important topic in various fields such as medical and social sciences. CATEs can be estimated with high accuracy if distributed data across multiple parties can be centralized. However, it is difficult to aggregate such data if they contain privacy information. To address this issue, we proposed data collaboration double machine learning (DC-DML), a method that can estimate CATE models with privacy preservation of distributed data, and evaluated the method through numerical experiments. Our contributions are summarized in the following three points. First, our method enables estimation and testing of semi-parametric CATE models without iterative communication on distributed data. Semi-parametric or non-parametric CATE models enable estimation and testing that is more robust to model mis-specification than parametric models. However, to our knowledge, no communication-efficient method has been proposed for estimating and 
    
[^4]: 学校选择中的非参数化治疗效果识别

    Nonparametric Treatment Effect Identification in School Choice. (arXiv:2112.03872v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2112.03872](http://arxiv.org/abs/2112.03872)

    本论文研究了集中学校分配中的非参数化治疗效果识别和估计方法，通过识别原子治疗效应，该研究揭示了在回归不连续和抽签驱动的变异下的学校选择的异质性和重要性。

    

    本论文研究集中学校分配中因果效应的非参数化识别和估计。在许多集中分配设置中，学生既受到抽签驱动的变异，也受到回归不连续（RD）驱动的变异。我们刻画了被识别的原子治疗效应（aTEs）的完整集合，定义为在给定学生特征时，一对学校之间的条件平均治疗效应。原子治疗效应是治疗对比的基础，常见的估计方法会掩盖重要的异质性。特别地，许多原子治疗效应的聚合将在RD变异驱动下置零权重，并且这种聚合的估计器在渐进下将对RD驱动的原子治疗效应放置逐渐消失的权重。我们开发了一种用于经验评估RD变异驱动下的原子治疗效应权重的诊断工具。最后，我们提供了估计器和相应的渐进结果以进行推断。

    This paper studies nonparametric identification and estimation of causal effects in centralized school assignment. In many centralized assignment settings, students are subjected to both lottery-driven variation and regression discontinuity (RD) driven variation. We characterize the full set of identified atomic treatment effects (aTEs), defined as the conditional average treatment effect between a pair of schools, given student characteristics. Atomic treatment effects are the building blocks of more aggregated notions of treatment contrasts, and common approaches estimating aggregations of aTEs can mask important heterogeneity. In particular, many aggregations of aTEs put zero weight on aTEs driven by RD variation, and estimators of such aggregations put asymptotically vanishing weight on the RD-driven aTEs. We develop a diagnostic tool for empirically assessing the weight put on aTEs driven by RD variation. Lastly, we provide estimators and accompanying asymptotic results for infere
    

