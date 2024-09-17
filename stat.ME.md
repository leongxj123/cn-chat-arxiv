# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Assumption-lean and Data-adaptive Post-Prediction Inference](https://arxiv.org/abs/2311.14220) | 这项工作介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，可以有效且有力地基于机器学习预测结果进行统计推断。 |
| [^2] | [Generative neural networks for characteristic functions.](http://arxiv.org/abs/2401.04778) | 本论文研究了利用生成神经网络模拟特征函数的问题，并通过构建一个普适且无需假设的生成神经网络来解决。研究基于最大均值差异度量，并提出了有关逼近质量的有限样本保证。 |
| [^3] | [Federated Epidemic Surveillance.](http://arxiv.org/abs/2307.02616) | 本研究旨在探索联邦方法在流行病监测中的应用。我们提出了一个假设检验框架，通过推送到保管人的防火墙并进行元分析，来解决数据分布和共享限制的问题。通过实验验证了我们的方法的有效性，并提出了适合的$p$-值合并方法。这些发现为联邦流行病监测提供了有价值的见解。 |

# 详细

[^1]: 假设简化和数据自适应的后预测推断

    Assumption-lean and Data-adaptive Post-Prediction Inference

    [https://arxiv.org/abs/2311.14220](https://arxiv.org/abs/2311.14220)

    这项工作介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，可以有效且有力地基于机器学习预测结果进行统计推断。

    

    现代科学研究面临的主要挑战是黄金标准数据的有限可用性，而获取这些数据既耗费时间又费力。随着机器学习（ML）的快速发展，科学家们依赖于ML算法使用易得的协变量来预测这些黄金标准结果。然而，这些预测结果常常直接用于后续的统计分析中，忽略了预测过程引入的不精确性和异质性。这可能导致虚假的正面结果和无效的科学结论。在这项工作中，我们介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，它允许基于ML预测结果进行有效和有力的推断。它的“假设简化”属性保证在广泛的统计量上不基于ML预测做出可靠的统计推断。它的“数据自适应”特性保证了相较于现有方法的效率提高。

    A primary challenge facing modern scientific research is the limited availability of gold-standard data which can be both costly and labor-intensive to obtain. With the rapid development of machine learning (ML), scientists have relied on ML algorithms to predict these gold-standard outcomes with easily obtained covariates. However, these predicted outcomes are often used directly in subsequent statistical analyses, ignoring imprecision and heterogeneity introduced by the prediction procedure. This will likely result in false positive findings and invalid scientific conclusions. In this work, we introduce an assumption-lean and data-adaptive Post-Prediction Inference (POP-Inf) procedure that allows valid and powerful inference based on ML-predicted outcomes. Its "assumption-lean" property guarantees reliable statistical inference without assumptions on the ML-prediction, for a wide range of statistical quantities. Its "data-adaptive'" feature guarantees an efficiency gain over existing
    
[^2]: 利用生成神经网络模拟特征函数

    Generative neural networks for characteristic functions. (arXiv:2401.04778v1 [stat.ML])

    [http://arxiv.org/abs/2401.04778](http://arxiv.org/abs/2401.04778)

    本论文研究了利用生成神经网络模拟特征函数的问题，并通过构建一个普适且无需假设的生成神经网络来解决。研究基于最大均值差异度量，并提出了有关逼近质量的有限样本保证。

    

    在这项工作中，我们提供了一个模拟算法来从一个（多元）特征函数中模拟，该特征函数仅以黑盒格式可访问。我们构建了一个生成神经网络，其损失函数利用最大均值差异度量的特定表示，直接结合目标特征函数。这种构造具有普遍性，不依赖于维度，并且不需要对给定特征函数进行任何假设。此外，还得出了关于最大均值差异度量的逼近质量的有限样本保证。该方法在一个短期模拟研究中进行了说明。

    In this work, we provide a simulation algorithm to simulate from a (multivariate) characteristic function, which is only accessible in a black-box format. We construct a generative neural network, whose loss function exploits a specific representation of the Maximum-Mean-Discrepancy metric to directly incorporate the targeted characteristic function. The construction is universal in the sense that it is independent of the dimension and that it does not require any assumptions on the given characteristic function. Furthermore, finite sample guarantees on the approximation quality in terms of the Maximum-Mean Discrepancy metric are derived. The method is illustrated in a short simulation study.
    
[^3]: 联邦流行病监测

    Federated Epidemic Surveillance. (arXiv:2307.02616v1 [stat.AP])

    [http://arxiv.org/abs/2307.02616](http://arxiv.org/abs/2307.02616)

    本研究旨在探索联邦方法在流行病监测中的应用。我们提出了一个假设检验框架，通过推送到保管人的防火墙并进行元分析，来解决数据分布和共享限制的问题。通过实验验证了我们的方法的有效性，并提出了适合的$p$-值合并方法。这些发现为联邦流行病监测提供了有价值的见解。

    

    流行病的监测是一项具有挑战性的任务，特别是当关键数据分散且利益相关方无法或不愿共享时。为了克服这一障碍，应开发联邦方法来整合实体愿意提供的较不敏感的证据。本研究旨在探索将假设检验推送到每个保管人的防火墙后，再通过元分析来合并结果的可行性，并确定重建假设检验和优化推理的最佳方法。我们提出了一个假设检验框架来识别指标的激增，并对真实数据和半合成数据进行功效分析和实验，以展示我们所提出的假设检验的性质，并提出合适的$p$-值合并方法。我们的研究结果凸显了使用$p$-值合并作为流行病监测的联邦方法的潜力，并为整合可用信息提供了宝贵的见解。

    The surveillance of a pandemic is a challenging task, especially when crucial data is distributed and stakeholders cannot or are unwilling to share. To overcome this obstacle, federated methodologies should be developed to incorporate less sensitive evidence that entities are willing to provide. This study aims to explore the feasibility of pushing hypothesis tests behind each custodian's firewall and then meta-analysis to combine the results, and to determine the optimal approach for reconstructing the hypothesis test and optimizing the inference. We propose a hypothesis testing framework to identify a surge in the indicators and conduct power analyses and experiments on real and semi-synthetic data to showcase the properties of our proposed hypothesis test and suggest suitable methods for combining $p$-values. Our findings highlight the potential of using $p$-value combination as a federated methodology for pandemic surveillance and provide valuable insights into integrating availabl
    

