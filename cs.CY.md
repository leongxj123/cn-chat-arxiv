# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks](https://arxiv.org/abs/2402.02933) | InterpretCC是一种新的解释性神经网络模型，通过条件计算和稀疏激活特征，在保持性能的同时实现了人类中心的解释能力。该模型适用于需要可信解释、可操作解释和准确预测的人类面向领域。 |

# 详细

[^1]: InterpretCC: 适于解释的神经网络的条件计算

    InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks

    [https://arxiv.org/abs/2402.02933](https://arxiv.org/abs/2402.02933)

    InterpretCC是一种新的解释性神经网络模型，通过条件计算和稀疏激活特征，在保持性能的同时实现了人类中心的解释能力。该模型适用于需要可信解释、可操作解释和准确预测的人类面向领域。

    

    神经网络的真实世界解释性在三个方面之间存在权衡：1）需要人类信任解释的近似（例如事后方法）；2）削弱了解释的可理解性（例如自动识别的特征掩码）；3）削弱了模型性能（例如决策树）。这些缺点对于面向人类的领域（如教育、医疗保健或自然语言）是不可接受的，这些领域需要可信的解释、可操作的解释和准确的预测。在这项工作中，我们提出了InterpretCC（可解释的条件计算），这是一种可解释性的设计神经网络系列，通过在预测之前自适应和稀疏地激活特征，确保人类中心的可解释性，同时保持与最先进模型相当的性能。我们将这个思想扩展为可解释的专家混合模型，允许人们离散地指定兴趣话题。

    Real-world interpretability for neural networks is a tradeoff between three concerns: 1) it requires humans to trust the explanation approximation (e.g. post-hoc approaches), 2) it compromises the understandability of the explanation (e.g. automatically identified feature masks), and 3) it compromises the model performance (e.g. decision trees). These shortcomings are unacceptable for human-facing domains, like education, healthcare, or natural language, which require trustworthy explanations, actionable interpretations, and accurate predictions. In this work, we present InterpretCC (interpretable conditional computation), a family of interpretable-by-design neural networks that guarantee human-centric interpretability while maintaining comparable performance to state-of-the-art models by adaptively and sparsely activating features before prediction. We extend this idea into an interpretable mixture-of-experts model, that allows humans to specify topics of interest, discretely separate
    

