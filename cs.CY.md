# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards an AI Accountability Policy.](http://arxiv.org/abs/2307.13658) | 这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。 |
| [^2] | [Learning from Discriminatory Training Data.](http://arxiv.org/abs/1912.08189) | 本文提出了一种公平学习方法，该方法能够在可能带有歧视的数据集上进行训练，且能够在公平的测试数据集上表现良好，且该方法可在消除歧视的情况下使用，并在受保护群体之间取得平衡。 |

# 详细

[^1]: 关于AI问责政策的探索

    Towards an AI Accountability Policy. (arXiv:2307.13658v1 [cs.CY])

    [http://arxiv.org/abs/2307.13658](http://arxiv.org/abs/2307.13658)

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。

    

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”作出的回应。在回答相关问题的关键句子末尾，提供了要求评论的问题编号的上标。该白皮书提出了一组相互关联的AI问责政策建议。

    This white paper is a response to the "AI Accountability Policy Request for Comments" by the National Telecommunications and Information Administration of the United States. The question numbers for which comments were requested are provided in superscripts at the end of key sentences answering the respective questions. The white paper offers a set of interconnected recommendations for an AI accountability policy.
    
[^2]: 从带有歧视性质的训练数据中学习

    Learning from Discriminatory Training Data. (arXiv:1912.08189v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1912.08189](http://arxiv.org/abs/1912.08189)

    本文提出了一种公平学习方法，该方法能够在可能带有歧视的数据集上进行训练，且能够在公平的测试数据集上表现良好，且该方法可在消除歧视的情况下使用，并在受保护群体之间取得平衡。

    

    监督学习系统是通过历史数据训练的，如果这些数据受到歧视性质的影响，那么该系统可能会在保护组中产生歧视。本文提出了公平学习的方法，即使在潜在的歧视性的数据集上训练，也将在公平的测试数据集上表现良好。这样的数据集转变为特定公平学习方法的应用方案。例如，消除直接歧视可以被表示为特定的数据集转变问题。对于这种情况，我们提出了一种学习方法，该方法在盲目训练包含直接加性歧视的数据集的同时，在公平数据集上可以证明最小化模型误差。该方法与现有的法律体系兼容，并通过在受保护群体之间取得平衡来解决广泛讨论的受保护群体交叉的问题。从技术上讲，该方法应用了概率干预，并具有因果和反事实公式。

    Supervised learning systems are trained using historical data and, if the data was tainted by discrimination, they may unintentionally learn to discriminate against protected groups. We propose that fair learning methods, despite training on potentially discriminatory datasets, shall perform well on fair test datasets. Such dataset shifts crystallize application scenarios for specific fair learning methods. For instance, the removal of direct discrimination can be represented as a particular dataset shift problem. For this scenario, we propose a learning method that provably minimizes model error on fair datasets, while blindly training on datasets poisoned with direct additive discrimination. The method is compatible with existing legal systems and provides a solution to the widely discussed issue of protected groups' intersectionality by striking a balance between the protected groups. Technically, the method applies probabilistic interventions, has causal and counterfactual formulat
    

