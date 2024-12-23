# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimizing Heat Alert Issuance with Reinforcement Learning](https://arxiv.org/abs/2312.14196) | 本研究利用强化学习优化热预警系统，通过引入新颖强化学习环境和综合数据集，解决了气候和健康环境中的低信号效应和空间异质性。 |
| [^2] | [Learning ECG signal features without backpropagation.](http://arxiv.org/abs/2307.01930) | 该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。 |
| [^3] | [Variable Selection for Kernel Two-Sample Tests.](http://arxiv.org/abs/2302.07415) | 本文提出了一种解决双样本检验中变量选择问题的框架，利用核最大均值差异统计量，以最大化方差正则化的MMD统计量。实验结果证明其超群表现。 |

# 详细

[^1]: 用强化学习优化热预警的发布

    Optimizing Heat Alert Issuance with Reinforcement Learning

    [https://arxiv.org/abs/2312.14196](https://arxiv.org/abs/2312.14196)

    本研究利用强化学习优化热预警系统，通过引入新颖强化学习环境和综合数据集，解决了气候和健康环境中的低信号效应和空间异质性。

    

    社会适应气候变化的关键战略之一是利用预警系统减少极端高温事件的不利健康影响，以促使预防性行动。本文研究了强化学习（RL）作为优化此类系统效果的工具。我们的贡献有三个方面。首先，我们引入了一个新颖的强化学习环境，评估热预警政策的有效性，以减少与高温有关的住院人数。奖励模型基于历史天气、医疗保险健康记录以及社会经济/地理特征的全面数据集进行训练。我们使用变分贝叶斯技术解决了在气候和健康环境中常见的低信号效应和空间异质性。转换模型结合了真实的历史天气模式，并通过基于气候区域相似性的数据增强机制进行增强。

    arXiv:2312.14196v2 Announce Type: replace  Abstract: A key strategy in societal adaptation to climate change is the use of alert systems to reduce the adverse health impacts of extreme heat events by prompting preventative action. In this work, we investigate reinforcement learning (RL) as a tool to optimize the effectiveness of such systems. Our contributions are threefold. First, we introduce a novel RL environment enabling the evaluation of the effectiveness of heat alert policies to reduce heat-related hospitalizations. The rewards model is trained from a comprehensive dataset of historical weather, Medicare health records, and socioeconomic/geographic features. We use variational Bayesian techniques to address low-signal effects and spatial heterogeneity, which are commonly encountered in climate & health settings. The transition model incorporates real historical weather patterns enriched by a data augmentation mechanism based on climate region similarity. Second, we use this env
    
[^2]: 学习ECG信号特征的非反向传播方法

    Learning ECG signal features without backpropagation. (arXiv:2307.01930v1 [cs.LG])

    [http://arxiv.org/abs/2307.01930](http://arxiv.org/abs/2307.01930)

    该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。

    

    表示学习已经成为机器学习领域的一个关键研究领域，它旨在发现用于提高分类和预测等下游任务的原始数据的有效特征的有效方法。在本文中，我们提出了一种用于生成时间序列类型数据表示的新方法。这种方法依靠理论物理的思想以数据驱动的方式构建紧凑的表示，并可以捕捉到数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性。这个新方法旨在识别能够有效捕捉属于特定类别的样本之间共享特征的线性规律。通过随后利用这些规律在前向方式下生成一个与分类器无关的表示，它们可以在广义设置中应用。我们展示了我们方法的有效性。

    Representation learning has become a crucial area of research in machine learning, as it aims to discover efficient ways of representing raw data with useful features to increase the effectiveness, scope and applicability of downstream tasks such as classification and prediction. In this paper, we propose a novel method to generate representations for time series-type data. This method relies on ideas from theoretical physics to construct a compact representation in a data-driven way, and it can capture both the underlying structure of the data and task-specific information while still remaining intuitive, interpretable and verifiable. This novel methodology aims to identify linear laws that can effectively capture a shared characteristic among samples belonging to a specific class. By subsequently utilizing these laws to generate a classifier-agnostic representation in a forward manner, they become applicable in a generalized setting. We demonstrate the effectiveness of our approach o
    
[^3]: 变量选择在核双样本检验中的应用

    Variable Selection for Kernel Two-Sample Tests. (arXiv:2302.07415v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.07415](http://arxiv.org/abs/2302.07415)

    本文提出了一种解决双样本检验中变量选择问题的框架，利用核最大均值差异统计量，以最大化方差正则化的MMD统计量。实验结果证明其超群表现。

    

    本文考虑了两样本检验中的变量选择问题，旨在选择区分两组样本的最有信息变量。为了解决该问题，我们提出了一种基于核最大均值差异（MMD）的框架。我们的方法寻求一组变量，其预先确定的大小最大化方差正则化的MMD统计量。这种计算形式也对应于在文献中研究的控制类型I错误的同时最小化异质类型II错误。我们介绍了混合整数编程公式，并提供了线性和二次类型内核函数的精确和近似算法，并具有性能保证。实验结果证明了我们的框架的卓越性能。

    We consider the variable selection problem for two-sample tests, aiming to select the most informative variables to distinguish samples from two groups. To solve this problem, we propose a framework based on the kernel maximum mean discrepancy (MMD). Our approach seeks a group of variables with a pre-specified size that maximizes the variance-regularized MMD statistics. This formulation also corresponds to the minimization of asymptotic type-II error while controlling type-I error, as studied in the literature. We present mixed-integer programming formulations and offer exact and approximation algorithms with performance guarantees for linear and quadratic types of kernel functions. Experimental results demonstrate the superior performance of our framework.
    

