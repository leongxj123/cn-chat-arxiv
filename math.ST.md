# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Limits of Assumption-free Tests for Algorithm Performance](https://arxiv.org/abs/2402.07388) | 这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。 |
| [^2] | [The Local Approach to Causal Inference under Network Interference.](http://arxiv.org/abs/2105.03810) | 我们提出了一种新的非参数建模框架，用于网络干扰条件下的因果推断，通过对代理人之间的连接方式进行建模和学习政策或治疗分配的影响。我们还提出了一种有效的测试方法来检验政策无关性/治疗效应，并对平均或分布式政策效应/治疗反应的估计器给出了上界。 |

# 详细

[^1]: 无假设测试算法性能的限制

    The Limits of Assumption-free Tests for Algorithm Performance

    [https://arxiv.org/abs/2402.07388](https://arxiv.org/abs/2402.07388)

    这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。

    

    算法评价和比较是机器学习和统计学中基本的问题，一个算法在给定的建模任务中表现如何，哪个算法表现最佳？许多方法已经开发出来评估算法性能，通常基于交叉验证策略，将感兴趣的算法在不同的数据子集上重新训练，并评估其在留出数据点上的性能。尽管广泛使用这些程序，但对于这些方法的理论性质尚未完全理解。在这项工作中，我们探讨了在有限的数据量下回答这些问题的一些基本限制。特别地，我们区分了两个问题: 算法$A$在大小为$n$的训练集上学习问题有多好，以及在特定大小为$n$的训练数据集上运行$A$所产生的特定拟合模型有多好？我们的主要结果证明，对于任何将算法视为黑盒的测试方法，无法准确地回答这两个问题。

    Algorithm evaluation and comparison are fundamental questions in machine learning and statistics -- how well does an algorithm perform at a given modeling task, and which algorithm performs best? Many methods have been developed to assess algorithm performance, often based around cross-validation type strategies, retraining the algorithm of interest on different subsets of the data and assessing its performance on the held-out data points. Despite the broad use of such procedures, the theoretical properties of these methods are not yet fully understood. In this work, we explore some fundamental limits for answering these questions with limited amounts of data. In particular, we make a distinction between two questions: how good is an algorithm $A$ at the problem of learning from a training set of size $n$, versus, how good is a particular fitted model produced by running $A$ on a particular training data set of size $n$?   Our main results prove that, for any test that treats the algor
    
[^2]: 网络干扰条件下因果推断的局部方法

    The Local Approach to Causal Inference under Network Interference. (arXiv:2105.03810v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2105.03810](http://arxiv.org/abs/2105.03810)

    我们提出了一种新的非参数建模框架，用于网络干扰条件下的因果推断，通过对代理人之间的连接方式进行建模和学习政策或治疗分配的影响。我们还提出了一种有效的测试方法来检验政策无关性/治疗效应，并对平均或分布式政策效应/治疗反应的估计器给出了上界。

    

    我们提出了一种新的非参数建模框架，用于处理社交或经济网络中代理人之间连接方式对结果产生影响的因果推断问题。这种网络干扰描述了关于治疗溢出、社交互动、社会学习、信息扩散、疾病和金融传染、社会资本形成等领域的大量文献。我们的方法首先通过测量路径距离来描述代理人在网络中的连接方式，然后通过汇集具有类似配置的代理人的结果数据来学习政策或治疗分配的影响。我们通过提出一个渐近有效的测试来演示该方法，该测试用于检验政策无关性/治疗效应的假设，并给出了针对平均或分布式政策效应/治疗反应的k最近邻估计器的均方误差的上界。

    We propose a new nonparametric modeling framework for causal inference when outcomes depend on how agents are linked in a social or economic network. Such network interference describes a large literature on treatment spillovers, social interactions, social learning, information diffusion, disease and financial contagion, social capital formation, and more. Our approach works by first characterizing how an agent is linked in the network using the configuration of other agents and connections nearby as measured by path distance. The impact of a policy or treatment assignment is then learned by pooling outcome data across similarly configured agents. We demonstrate the approach by proposing an asymptotically valid test for the hypothesis of policy irrelevance/no treatment effects and bounding the mean-squared error of a k-nearest-neighbor estimator for the average or distributional policy effect/treatment response.
    

