# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpreting GNN-based IDS Detections Using Provenance Graph Structural Features.](http://arxiv.org/abs/2306.00934) | 基于PROVEXPLAINER框架，通过复制GNN-based security models的决策过程，利用决策树和图结构特征将抽象GNN决策边界投影到可解释的特征空间，以增强GNN安全模型的透明度和询问能力。 |
| [^2] | [Differential Privacy via Distributionally Robust Optimization.](http://arxiv.org/abs/2304.12681) | 本文开发了一类机制，以实现无条件最优性保证的差分隐私。该机制将机制设计问题制定为无限维分布鲁棒优化问题。 |

# 详细

[^1]: 基于权威图结构特征对基于GNN的IDS检测进行解释

    Interpreting GNN-based IDS Detections Using Provenance Graph Structural Features. (arXiv:2306.00934v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.00934](http://arxiv.org/abs/2306.00934)

    基于PROVEXPLAINER框架，通过复制GNN-based security models的决策过程，利用决策树和图结构特征将抽象GNN决策边界投影到可解释的特征空间，以增强GNN安全模型的透明度和询问能力。

    

    复杂神经网络模型的黑匣子本质妨碍了它们在安全领域的普及，因为它们缺乏逻辑解释和可执行后续行动的预测。为了增强在系统来源分析中使用的图神经网络（GNN）安全模型的透明度和问责制，我们提出了PROVEXPLAINER，一种将抽象GNN决策边界投影到可解释特征空间的框架。我们首先使用简单且可解释的模型，如决策树（DT），复制基于GNN的安全模型的决策过程。为了最大化替代模型的准确性和保真度，我们提出了一种基于经典图论的图结构特征，并通过安全领域知识的广泛数据研究对其进行了增强。我们的图结构特征与系统来源领域中的问题空间行动密切相关，这使检测结果可用人类语言描述和解释。

    The black-box nature of complex Neural Network (NN)-based models has hindered their widespread adoption in security domains due to the lack of logical explanations and actionable follow-ups for their predictions. To enhance the transparency and accountability of Graph Neural Network (GNN) security models used in system provenance analysis, we propose PROVEXPLAINER, a framework for projecting abstract GNN decision boundaries onto interpretable feature spaces.  We first replicate the decision-making process of GNNbased security models using simpler and explainable models such as Decision Trees (DTs). To maximize the accuracy and fidelity of the surrogate models, we propose novel graph structural features founded on classical graph theory and enhanced by extensive data study with security domain knowledge. Our graph structural features are closely tied to problem-space actions in the system provenance domain, which allows the detection results to be explained in descriptive, human languag
    
[^2]: 分布式鲁棒优化实现差分隐私保护

    Differential Privacy via Distributionally Robust Optimization. (arXiv:2304.12681v1 [cs.CR])

    [http://arxiv.org/abs/2304.12681](http://arxiv.org/abs/2304.12681)

    本文开发了一类机制，以实现无条件最优性保证的差分隐私。该机制将机制设计问题制定为无限维分布鲁棒优化问题。

    

    近年来，差分隐私已成为共享数据集统计信息并限制涉及个人的私人信息披露的事实标准。通过对将要发布的统计数据进行随机扰动来实现这一目标，这反过来导致了隐私和准确性之间的权衡：更大的扰动提供更强的隐私保证，但结果是提供较低实用度的统计数据和更低的准确性。因此，特别感兴趣的是在预选隐私水平的情况下提供最高准确性的最佳机制。迄今为止，这一领域的工作集中在事先指定扰动族并随后证明其渐近和/或最佳性上，本文则开发了一类机制，它们具有非渐近和无条件的最佳性保证。为此，我们将机制设计问题制定为无限维分布鲁棒优化问题。

    In recent years, differential privacy has emerged as the de facto standard for sharing statistics of datasets while limiting the disclosure of private information about the involved individuals. This is achieved by randomly perturbing the statistics to be published, which in turn leads to a privacy-accuracy trade-off: larger perturbations provide stronger privacy guarantees, but they result in less accurate statistics that offer lower utility to the recipients. Of particular interest are therefore optimal mechanisms that provide the highest accuracy for a pre-selected level of privacy. To date, work in this area has focused on specifying families of perturbations a priori and subsequently proving their asymptotic and/or best-in-class optimality. In this paper, we develop a class of mechanisms that enjoy non-asymptotic and unconditional optimality guarantees. To this end, we formulate the mechanism design problem as an infinite-dimensional distributionally robust optimization problem. W
    

