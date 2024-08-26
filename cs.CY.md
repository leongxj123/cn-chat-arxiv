# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Vulnerability of Fairness Constrained Learning to Malicious Noise.](http://arxiv.org/abs/2307.11892) | 这项研究考虑了公正约束学习对恶意噪声的脆弱性，发现使用随机分类器可以在精度上只损失$\Theta(\alpha)$和$O(\sqrt{\alpha})$，对应不同的公正约束要求。 |
| [^2] | [Intelligent Energy Management with IoT Framework in Smart Cities Using Intelligent Analysis: An Application of Machine Learning Methods for Complex Networks and Systems.](http://arxiv.org/abs/2306.05567) | 本研究开发了一个智能城市能源管理的物联网框架，结合智能分析和多组件的架构，研究了基于智能机制的智能能源管理解决方案，以期节能和优化管理。 |

# 详细

[^1]: 关于受恶意噪声影响的公正约束学习的脆弱性

    On the Vulnerability of Fairness Constrained Learning to Malicious Noise. (arXiv:2307.11892v1 [cs.LG])

    [http://arxiv.org/abs/2307.11892](http://arxiv.org/abs/2307.11892)

    这项研究考虑了公正约束学习对恶意噪声的脆弱性，发现使用随机分类器可以在精度上只损失$\Theta(\alpha)$和$O(\sqrt{\alpha})$，对应不同的公正约束要求。

    

    我们考虑了公正约束学习对训练数据中微小恶意噪声的脆弱性。Konstantinov和Lampert (2021)在这个问题上进行了研究，并展示了负面结果，表明在不平衡的群组大小下存在一些数据分布，任何适当的学习器都会表现出较高的脆弱性。在这里，我们展示了更乐观的观点，如果允许随机分类器，则情况更加细致。例如，对于人口统计学平等性，我们显示只会产生$\Theta(\alpha)$的精度损失，其中$\alpha$是恶意噪声率，甚至可以与没有公正约束的情况完全匹配。对于机会均等性，我们显示只会产生$O(\sqrt{\alpha})$的损失，并给出一个匹配的$\Omega(\sqrt{\alpha})$的下界。相比之下，Konstantinov和Lampert (2021)示范了对于适当的学习器，这两个概念的精度损失都是$\Omega(1)$。关键的技术创新是

    We consider the vulnerability of fairness-constrained learning to small amounts of malicious noise in the training data. Konstantinov and Lampert (2021) initiated the study of this question and presented negative results showing there exist data distributions where for several fairness constraints, any proper learner will exhibit high vulnerability when group sizes are imbalanced. Here, we present a more optimistic view, showing that if we allow randomized classifiers, then the landscape is much more nuanced. For example, for Demographic Parity we show we can incur only a $\Theta(\alpha)$ loss in accuracy, where $\alpha$ is the malicious noise rate, matching the best possible even without fairness constraints. For Equal Opportunity, we show we can incur an $O(\sqrt{\alpha})$ loss, and give a matching $\Omega(\sqrt{\alpha})$lower bound. In contrast, Konstantinov and Lampert (2021) showed for proper learners the loss in accuracy for both notions is $\Omega(1)$. The key technical novelty 
    
[^2]: 智能分析，在物联网框架下的智能城市能源管理：复杂网络和系统机器学习方法应用的案例研究

    Intelligent Energy Management with IoT Framework in Smart Cities Using Intelligent Analysis: An Application of Machine Learning Methods for Complex Networks and Systems. (arXiv:2306.05567v1 [cs.LG])

    [http://arxiv.org/abs/2306.05567](http://arxiv.org/abs/2306.05567)

    本研究开发了一个智能城市能源管理的物联网框架，结合智能分析和多组件的架构，研究了基于智能机制的智能能源管理解决方案，以期节能和优化管理。

    

    智能建筑越来越多地使用基于物联网的无线传感系统来降低能源消耗和环境影响。本研究的主要贡献是开发了一个全面的基于物联网的智能城市能源管理框架，融合了多个物联网架构和框架的组件。该框架通过智能分析，不仅收集和存储信息，而且还是其他企业开发应用的平台。此外，我们还研究了基于智能机制的智能能源管理解决方案。能源资源的消耗和需求增加导致了节能与优化管理的需求和挑战。

    Smart buildings are increasingly using Internet of Things (IoT)-based wireless sensing systems to reduce their energy consumption and environmental impact. As a result of their compact size and ability to sense, measure, and compute all electrical properties, Internet of Things devices have become increasingly important in our society. A major contribution of this study is the development of a comprehensive IoT-based framework for smart city energy management, incorporating multiple components of IoT architecture and framework. An IoT framework for intelligent energy management applications that employ intelligent analysis is an essential system component that collects and stores information. Additionally, it serves as a platform for the development of applications by other companies. Furthermore, we have studied intelligent energy management solutions based on intelligent mechanisms. The depletion of energy resources and the increase in energy demand have led to an increase in energy 
    

