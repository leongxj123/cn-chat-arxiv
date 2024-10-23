# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FDINet: Protecting against DNN Model Extraction via Feature Distortion Index.](http://arxiv.org/abs/2306.11338) | FDINet是一种新颖的防御机制，该机制利用特征失真指数来保护DNN模型免受模型提取攻击，并利用FDI相似性来识别分布式提取攻击中的勾结敌人。 |
| [^2] | [Interpreting GNN-based IDS Detections Using Provenance Graph Structural Features.](http://arxiv.org/abs/2306.00934) | 基于PROVEXPLAINER框架，通过复制GNN-based security models的决策过程，利用决策树和图结构特征将抽象GNN决策边界投影到可解释的特征空间，以增强GNN安全模型的透明度和询问能力。 |

# 详细

[^1]: FDINet：利用特征失真指数保护 DNN 模型免受模型提取攻击

    FDINet: Protecting against DNN Model Extraction via Feature Distortion Index. (arXiv:2306.11338v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.11338](http://arxiv.org/abs/2306.11338)

    FDINet是一种新颖的防御机制，该机制利用特征失真指数来保护DNN模型免受模型提取攻击，并利用FDI相似性来识别分布式提取攻击中的勾结敌人。

    

    机器学习即服务（MLaaS）平台由于其易用性、成本效益、可扩展性和快速开发能力而变得越来越受欢迎。然而，最近的研究强调了 MLaaS 中基于云的模型对模型提取攻击的脆弱性。本文介绍了 FDINET，一种利用深度神经网络（DNN）模型特征分布的新颖防御机制。具体地，通过分析对手的查询的特征分布，我们揭示了这些查询的特征分布与模型的训练集不同。基于这个关键观察，我们提出了特征失真指数（FDI），这是一种度量设计，用于定量测量接收到的查询的特征分布偏差。所提出的 FDINET 利用 FDI 训练一个二进制检测器，并利用 FDI 相似性识别分布式提取攻击中的勾结敌人。我们进行了广泛的实验来评估 FDINET 对抗模型提取攻击的效果。

    Machine Learning as a Service (MLaaS) platforms have gained popularity due to their accessibility, cost-efficiency, scalability, and rapid development capabilities. However, recent research has highlighted the vulnerability of cloud-based models in MLaaS to model extraction attacks. In this paper, we introduce FDINET, a novel defense mechanism that leverages the feature distribution of deep neural network (DNN) models. Concretely, by analyzing the feature distribution from the adversary's queries, we reveal that the feature distribution of these queries deviates from that of the model's training set. Based on this key observation, we propose Feature Distortion Index (FDI), a metric designed to quantitatively measure the feature distribution deviation of received queries. The proposed FDINET utilizes FDI to train a binary detector and exploits FDI similarity to identify colluding adversaries from distributed extraction attacks. We conduct extensive experiments to evaluate FDINET against
    
[^2]: 基于权威图结构特征对基于GNN的IDS检测进行解释

    Interpreting GNN-based IDS Detections Using Provenance Graph Structural Features. (arXiv:2306.00934v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.00934](http://arxiv.org/abs/2306.00934)

    基于PROVEXPLAINER框架，通过复制GNN-based security models的决策过程，利用决策树和图结构特征将抽象GNN决策边界投影到可解释的特征空间，以增强GNN安全模型的透明度和询问能力。

    

    复杂神经网络模型的黑匣子本质妨碍了它们在安全领域的普及，因为它们缺乏逻辑解释和可执行后续行动的预测。为了增强在系统来源分析中使用的图神经网络（GNN）安全模型的透明度和问责制，我们提出了PROVEXPLAINER，一种将抽象GNN决策边界投影到可解释特征空间的框架。我们首先使用简单且可解释的模型，如决策树（DT），复制基于GNN的安全模型的决策过程。为了最大化替代模型的准确性和保真度，我们提出了一种基于经典图论的图结构特征，并通过安全领域知识的广泛数据研究对其进行了增强。我们的图结构特征与系统来源领域中的问题空间行动密切相关，这使检测结果可用人类语言描述和解释。

    The black-box nature of complex Neural Network (NN)-based models has hindered their widespread adoption in security domains due to the lack of logical explanations and actionable follow-ups for their predictions. To enhance the transparency and accountability of Graph Neural Network (GNN) security models used in system provenance analysis, we propose PROVEXPLAINER, a framework for projecting abstract GNN decision boundaries onto interpretable feature spaces.  We first replicate the decision-making process of GNNbased security models using simpler and explainable models such as Decision Trees (DTs). To maximize the accuracy and fidelity of the surrogate models, we propose novel graph structural features founded on classical graph theory and enhanced by extensive data study with security domain knowledge. Our graph structural features are closely tied to problem-space actions in the system provenance domain, which allows the detection results to be explained in descriptive, human languag
    

