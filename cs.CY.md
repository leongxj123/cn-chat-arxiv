# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatGPT Based Data Augmentation for Improved Parameter-Efficient Debiasing of LLMs](https://arxiv.org/abs/2402.11764) | 本研究提出了一种利用ChatGPT生成合成训练数据来增强LLMs去偏见化的新方法，能够高效地去除已知偏见并跨越不同类别进行去偏见化。 |
| [^2] | [Federated Epidemic Surveillance.](http://arxiv.org/abs/2307.02616) | 本研究旨在探索联邦方法在流行病监测中的应用。我们提出了一个假设检验框架，通过推送到保管人的防火墙并进行元分析，来解决数据分布和共享限制的问题。通过实验验证了我们的方法的有效性，并提出了适合的$p$-值合并方法。这些发现为联邦流行病监测提供了有价值的见解。 |

# 详细

[^1]: 基于ChatGPT的数据增强技术用于改善LLMs的参数高效去偏见化

    ChatGPT Based Data Augmentation for Improved Parameter-Efficient Debiasing of LLMs

    [https://arxiv.org/abs/2402.11764](https://arxiv.org/abs/2402.11764)

    本研究提出了一种利用ChatGPT生成合成训练数据来增强LLMs去偏见化的新方法，能够高效地去除已知偏见并跨越不同类别进行去偏见化。

    

    大语言模型（LLMs）虽然功能强大，但存在有害的社会偏见。由于计算成本、数据约束和可能降低多任务语言能力，去偏见化通常具有挑战性。本文介绍了一种利用ChatGPT生成合成训练数据的新方法，旨在增强LLMs的去偏见化。我们提出了两种策略：目标提示，对已知偏见提供有效的去偏见化，但需要事先指定问题中的偏见; 一般提示，虽然效果稍逊，但能够跨各种类别进行去偏见化。我们利用适配器调整来实现资源高效的LLM去偏见化，并比较了我们的合成数据与现有去偏见化数据集的效果。我们的结果表明：（1）ChatGPT可以高效地生成用于去偏见化其他LLMs的高质量训练数据；（2）通过我们的方法生成的数据超越了现有数据集在去偏见化上的效果。

    arXiv:2402.11764v1 Announce Type: cross  Abstract: Large Language models (LLMs), while powerful, exhibit harmful social biases. Debiasing is often challenging due to computational costs, data constraints, and potential degradation of multi-task language capabilities. This work introduces a novel approach utilizing ChatGPT to generate synthetic training data, aiming to enhance the debiasing of LLMs. We propose two strategies: Targeted Prompting, which provides effective debiasing for known biases but necessitates prior specification of bias in question; and General Prompting, which, while slightly less effective, offers debiasing across various categories. We leverage resource-efficient LLM debiasing using adapter tuning and compare the effectiveness of our synthetic data to existing debiasing datasets. Our results reveal that: (1) ChatGPT can efficiently produce high-quality training data for debiasing other LLMs; (2) data produced via our approach surpasses existing datasets in debias
    
[^2]: 联邦流行病监测

    Federated Epidemic Surveillance. (arXiv:2307.02616v1 [stat.AP])

    [http://arxiv.org/abs/2307.02616](http://arxiv.org/abs/2307.02616)

    本研究旨在探索联邦方法在流行病监测中的应用。我们提出了一个假设检验框架，通过推送到保管人的防火墙并进行元分析，来解决数据分布和共享限制的问题。通过实验验证了我们的方法的有效性，并提出了适合的$p$-值合并方法。这些发现为联邦流行病监测提供了有价值的见解。

    

    流行病的监测是一项具有挑战性的任务，特别是当关键数据分散且利益相关方无法或不愿共享时。为了克服这一障碍，应开发联邦方法来整合实体愿意提供的较不敏感的证据。本研究旨在探索将假设检验推送到每个保管人的防火墙后，再通过元分析来合并结果的可行性，并确定重建假设检验和优化推理的最佳方法。我们提出了一个假设检验框架来识别指标的激增，并对真实数据和半合成数据进行功效分析和实验，以展示我们所提出的假设检验的性质，并提出合适的$p$-值合并方法。我们的研究结果凸显了使用$p$-值合并作为流行病监测的联邦方法的潜力，并为整合可用信息提供了宝贵的见解。

    The surveillance of a pandemic is a challenging task, especially when crucial data is distributed and stakeholders cannot or are unwilling to share. To overcome this obstacle, federated methodologies should be developed to incorporate less sensitive evidence that entities are willing to provide. This study aims to explore the feasibility of pushing hypothesis tests behind each custodian's firewall and then meta-analysis to combine the results, and to determine the optimal approach for reconstructing the hypothesis test and optimizing the inference. We propose a hypothesis testing framework to identify a surge in the indicators and conduct power analyses and experiments on real and semi-synthetic data to showcase the properties of our proposed hypothesis test and suggest suitable methods for combining $p$-values. Our findings highlight the potential of using $p$-value combination as a federated methodology for pandemic surveillance and provide valuable insights into integrating availabl
    

