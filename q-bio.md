# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved prediction of ligand-protein binding affinities by meta-modeling.](http://arxiv.org/abs/2310.03946) | 通过整合基于结构的对接和基于序列的深度学习模型，开发了一个元模型框架，显著改善了配体-蛋白质结合亲和力预测的性能。 |
| [^2] | [A benchmark for computational analysis of animal behavior, using animal-borne tags.](http://arxiv.org/abs/2305.10740) | 该论文介绍了一个名为BEBE的动物行为计算分析基准，其中包括了1654小时的动物生态生理学数据，这是迄今为止最大、最具分类多样性的公开可用的数据集合。在这个基准上，作者使用了十种机器学习方法并确定了未来工作中需要解决的关键问题。 |

# 详细

[^1]: 通过元模型改进了配体-蛋白质结合亲和力的预测

    Improved prediction of ligand-protein binding affinities by meta-modeling. (arXiv:2310.03946v1 [cs.LG])

    [http://arxiv.org/abs/2310.03946](http://arxiv.org/abs/2310.03946)

    通过整合基于结构的对接和基于序列的深度学习模型，开发了一个元模型框架，显著改善了配体-蛋白质结合亲和力预测的性能。

    

    通过计算方法准确筛选候选药物配体与靶蛋白的结合是药物开发的主要关注点，因为筛选潜在候选物能够节省找药物的时间和费用。这种虚拟筛选部分依赖于预测配体和蛋白质之间的结合亲和力的方法。鉴于存在许多计算模型对不同目标的结合亲和力预测结果不同，我们在这里开发了一个元模型框架，通过整合已发表的基于结构的对接和基于序列的深度学习模型来构建。在构建这个框架时，我们评估了许多组合的个别模型、训练数据库以及线性和非线性的元模型方法。我们显示出许多元模型在亲和力预测上显著改善了个别基础模型的性能。我们最好的元模型达到了与最先进的纯结构为基础的深度学习工具相当的性能。总体而言，我们证明了这个元模型框架可以显著改善配体-蛋白质结合亲和力预测的性能。

    The accurate screening of candidate drug ligands against target proteins through computational approaches is of prime interest to drug development efforts, as filtering potential candidates would save time and expenses for finding drugs. Such virtual screening depends in part on methods to predict the binding affinity between ligands and proteins. Given many computational models for binding affinity prediction with varying results across targets, we herein develop a meta-modeling framework by integrating published empirical structure-based docking and sequence-based deep learning models. In building this framework, we evaluate many combinations of individual models, training databases, and linear and nonlinear meta-modeling approaches. We show that many of our meta-models significantly improve affinity predictions over individual base models. Our best meta-models achieve comparable performance to state-of-the-art exclusively structure-based deep learning tools. Overall, we demonstrate 
    
[^2]: 一种动物行为计算分析的基准，使用动物携带的标签。

    A benchmark for computational analysis of animal behavior, using animal-borne tags. (arXiv:2305.10740v1 [cs.LG])

    [http://arxiv.org/abs/2305.10740](http://arxiv.org/abs/2305.10740)

    该论文介绍了一个名为BEBE的动物行为计算分析基准，其中包括了1654小时的动物生态生理学数据，这是迄今为止最大、最具分类多样性的公开可用的数据集合。在这个基准上，作者使用了十种机器学习方法并确定了未来工作中需要解决的关键问题。

    

    动物携带的传感器（“生物记录器”）可以记录一系列动力学和环境数据，揭示动物生态生理学并改善保护工作。机器学习技术对于解释生物记录器记录的大量数据非常有用，但在这个领域中没有标准来比较不同的机器学习技术。为了解决这个问题，我们提出了Bio-logger Ethogram Benchmark（BEBE），这是一个带有行为注释，标准化建模任务和评估指标的数据集合。BEBE是迄今为止最大、最具分类多样性和公开可用的这种基准，包括来自九个分类单元中149个个体收集的1654小时数据。我们在BEBE上评估了十种不同的机器学习方法的性能，并确定了未来工作中需要解决的关键问题。数据集、模型和评估代码已公开发布在https://github.com/earthspecies/BEBE，以便社区使用。

    Animal-borne sensors ('bio-loggers') can record a suite of kinematic and environmental data, which can elucidate animal ecophysiology and improve conservation efforts. Machine learning techniques are useful for interpreting the large amounts of data recorded by bio-loggers, but there exists no standard for comparing the different machine learning techniques in this domain. To address this, we present the Bio-logger Ethogram Benchmark (BEBE), a collection of datasets with behavioral annotations, standardized modeling tasks, and evaluation metrics. BEBE is to date the largest, most taxonomically diverse, publicly available benchmark of this type, and includes 1654 hours of data collected from 149 individuals across nine taxa. We evaluate the performance of ten different machine learning methods on BEBE, and identify key challenges to be addressed in future work. Datasets, models, and evaluation code are made publicly available at https://github.com/earthspecies/BEBE, to enable community 
    

