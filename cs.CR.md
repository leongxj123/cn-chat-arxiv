# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Complex Qeury Answering](https://arxiv.org/abs/2402.14609) | 研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战 |
| [^2] | [On the Vulnerability of Fairness Constrained Learning to Malicious Noise.](http://arxiv.org/abs/2307.11892) | 这项研究考虑了公正约束学习对恶意噪声的脆弱性，发现使用随机分类器可以在精度上只损失$\Theta(\alpha)$和$O(\sqrt{\alpha})$，对应不同的公正约束要求。 |
| [^3] | [Fast Adaptive Test-Time Defense with Robust Features.](http://arxiv.org/abs/2307.11672) | 本文提出了一种快速适应的测试防御策略，通过投影训练好的模型到最稳健的特征空间，降低了对抗攻击的脆弱性，无需额外的测试时间计算。 |

# 详细

[^1]: 联邦式复杂查询答案方法研究

    Federated Complex Qeury Answering

    [https://arxiv.org/abs/2402.14609](https://arxiv.org/abs/2402.14609)

    研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战

    

    知识图谱中的复杂逻辑查询答案是一个具有挑战性的任务，已经得到广泛研究。执行复杂逻辑推理的能力是必不可少的，并支持各种基于图推理的下游任务，比如搜索引擎。最近提出了一些方法，将知识图谱实体和逻辑查询表示为嵌入向量，并从知识图谱中找到逻辑查询的答案。然而，现有的方法主要集中在查询单个知识图谱上，并不能应用于多个图形。此外，直接共享带有敏感信息的知识图谱可能会带来隐私风险，使得共享和构建一个聚合知识图谱用于推理以检索查询答案是不切实际的。因此，目前仍然不清楚如何在多源知识图谱上回答查询。一个实体可能涉及到多个知识图谱，对多个知识图谱进行推理，并在多源知识图谱上回答复杂查询对于发现知识是重要的。

    arXiv:2402.14609v1 Announce Type: cross  Abstract: Complex logical query answering is a challenging task in knowledge graphs (KGs) that has been widely studied. The ability to perform complex logical reasoning is essential and supports various graph reasoning-based downstream tasks, such as search engines. Recent approaches are proposed to represent KG entities and logical queries into embedding vectors and find answers to logical queries from the KGs. However, existing proposed methods mainly focus on querying a single KG and cannot be applied to multiple graphs. In addition, directly sharing KGs with sensitive information may incur privacy risks, making it impractical to share and construct an aggregated KG for reasoning to retrieve query answers. Thus, it remains unknown how to answer queries on multi-source KGs. An entity can be involved in various knowledge graphs and reasoning on multiple KGs and answering complex queries on multi-source KGs is important in discovering knowledge 
    
[^2]: 关于受恶意噪声影响的公正约束学习的脆弱性

    On the Vulnerability of Fairness Constrained Learning to Malicious Noise. (arXiv:2307.11892v1 [cs.LG])

    [http://arxiv.org/abs/2307.11892](http://arxiv.org/abs/2307.11892)

    这项研究考虑了公正约束学习对恶意噪声的脆弱性，发现使用随机分类器可以在精度上只损失$\Theta(\alpha)$和$O(\sqrt{\alpha})$，对应不同的公正约束要求。

    

    我们考虑了公正约束学习对训练数据中微小恶意噪声的脆弱性。Konstantinov和Lampert (2021)在这个问题上进行了研究，并展示了负面结果，表明在不平衡的群组大小下存在一些数据分布，任何适当的学习器都会表现出较高的脆弱性。在这里，我们展示了更乐观的观点，如果允许随机分类器，则情况更加细致。例如，对于人口统计学平等性，我们显示只会产生$\Theta(\alpha)$的精度损失，其中$\alpha$是恶意噪声率，甚至可以与没有公正约束的情况完全匹配。对于机会均等性，我们显示只会产生$O(\sqrt{\alpha})$的损失，并给出一个匹配的$\Omega(\sqrt{\alpha})$的下界。相比之下，Konstantinov和Lampert (2021)示范了对于适当的学习器，这两个概念的精度损失都是$\Omega(1)$。关键的技术创新是

    We consider the vulnerability of fairness-constrained learning to small amounts of malicious noise in the training data. Konstantinov and Lampert (2021) initiated the study of this question and presented negative results showing there exist data distributions where for several fairness constraints, any proper learner will exhibit high vulnerability when group sizes are imbalanced. Here, we present a more optimistic view, showing that if we allow randomized classifiers, then the landscape is much more nuanced. For example, for Demographic Parity we show we can incur only a $\Theta(\alpha)$ loss in accuracy, where $\alpha$ is the malicious noise rate, matching the best possible even without fairness constraints. For Equal Opportunity, we show we can incur an $O(\sqrt{\alpha})$ loss, and give a matching $\Omega(\sqrt{\alpha})$lower bound. In contrast, Konstantinov and Lampert (2021) showed for proper learners the loss in accuracy for both notions is $\Omega(1)$. The key technical novelty 
    
[^3]: 快速自适应测试防御与稳健特征

    Fast Adaptive Test-Time Defense with Robust Features. (arXiv:2307.11672v1 [cs.LG])

    [http://arxiv.org/abs/2307.11672](http://arxiv.org/abs/2307.11672)

    本文提出了一种快速适应的测试防御策略，通过投影训练好的模型到最稳健的特征空间，降低了对抗攻击的脆弱性，无需额外的测试时间计算。

    

    自适应的测试防御被用来提高深度神经网络对抗性样本的鲁棒性。然而，现有方法由于对模型参数或输入进行额外的优化导致推理时间大幅增加。在本工作中，我们提出了一种新颖的自适应测试防御策略，它可以与任何现有（稳健的）训练过程轻松集成，并且无需额外的测试时间计算。基于我们提出的特征鲁棒性的概念，关键思想是将训练好的模型投影到最稳健的特征空间，从而降低对非稳健方向的对抗攻击的脆弱性。我们在广义可加性模型和使用神经切向核函数（NTK）等价法证明了特征矩阵的顶层特征空间更加稳健。我们在CIFAR-10和CIFAR-100数据集上进行了大量实验，用于几个稳健性基准测试。

    Adaptive test-time defenses are used to improve the robustness of deep neural networks to adversarial examples. However, existing methods significantly increase the inference time due to additional optimization on the model parameters or the input at test time. In this work, we propose a novel adaptive test-time defense strategy that is easy to integrate with any existing (robust) training procedure without additional test-time computation. Based on the notion of robustness of features that we present, the key idea is to project the trained models to the most robust feature space, thereby reducing the vulnerability to adversarial attacks in non-robust directions. We theoretically show that the top eigenspace of the feature matrix are more robust for a generalized additive model and support our argument for a large width neural network with the Neural Tangent Kernel (NTK) equivalence. We conduct extensive experiments on CIFAR-10 and CIFAR-100 datasets for several robustness benchmarks, 
    

