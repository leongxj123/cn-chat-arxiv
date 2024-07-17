# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automating Psychological Hypothesis Generation with AI: Large Language Models Meet Causal Graph](https://arxiv.org/abs/2402.14424) | 利用大型语言模型和因果图结合的方法，在心理学假设生成中取得了突破，结果显示这种联合方法在新颖性方面明显优于仅使用大型语言模型的假设。 |
| [^2] | [Designing Decision Support Systems Using Counterfactual Prediction Sets.](http://arxiv.org/abs/2306.03928) | 本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。 |

# 详细

[^1]: 利用人工智能自动化心理学假设生成：大型语言模型结合因果图

    Automating Psychological Hypothesis Generation with AI: Large Language Models Meet Causal Graph

    [https://arxiv.org/abs/2402.14424](https://arxiv.org/abs/2402.14424)

    利用大型语言模型和因果图结合的方法，在心理学假设生成中取得了突破，结果显示这种联合方法在新颖性方面明显优于仅使用大型语言模型的假设。

    

    我们的研究利用因果知识图谱和大型语言模型（LLM）之间的协同作用，引入了一种突破性的计算方法来生成心理学假设。我们使用LLM分析了43,312篇心理学文章，提取了因果关系对，生成了一个专门针对心理学的因果图。应用链接预测算法，我们生成了130个关注“幸福”的潜在心理学假设，然后将其与博士学者构思的研究想法和仅由LLM产生的想法进行了比较。有趣的是，我们的LLM和因果图的联合方法在新颖性方面与专家水平的洞察力保持一致，明显优于仅LLM的假设（分别为t(59)=3.34，p=0.007和t(59)=4.32，p<0.001）。这种一致性进一步通过深度语义分析得到证实。我们的结果表明，将LLM与因果图等机器学习技术相结合，可以更好地生成心理学假设。

    arXiv:2402.14424v1 Announce Type: new  Abstract: Leveraging the synergy between causal knowledge graphs and a large language model (LLM), our study introduces a groundbreaking approach for computational hypothesis generation in psychology. We analyzed 43,312 psychology articles using a LLM to extract causal relation pairs. This analysis produced a specialized causal graph for psychology. Applying link prediction algorithms, we generated 130 potential psychological hypotheses focusing on `well-being', then compared them against research ideas conceived by doctoral scholars and those produced solely by the LLM. Interestingly, our combined approach of a LLM and causal graphs mirrored the expert-level insights in terms of novelty, clearly surpassing the LLM-only hypotheses (t(59) = 3.34, p=0.007 and t(59) = 4.32, p<0.001, respectively). This alignment was further corroborated using deep semantic analysis. Our results show that combining LLM with machine learning techniques such as causal k
    
[^2]: 使用反事实预测集设计决策支持系统

    Designing Decision Support Systems Using Counterfactual Prediction Sets. (arXiv:2306.03928v1 [cs.LG])

    [http://arxiv.org/abs/2306.03928](http://arxiv.org/abs/2306.03928)

    本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。

    

    分类任务的决策支持系统通常被设计用于预测地面实况标签的值。然而，由于它们的预测并不完美，这些系统还需要让人类专家了解何时以及如何使用这些预测来更新自己的预测。不幸的是，这被证明是具有挑战性的。最近有人认为，另一种类型的决策支持系统可能会避开这个挑战。这些系统不是提供单个标签预测，而是使用符合预测器构建一组标签预测值，即预测集，并强制要求专家从预测集中预测一个标签值。然而，这些系统的设计和评估迄今仍依赖于样式化的专家模型，这引发了人们对它们的承诺的质疑。本文从在线学习的角度重新审视了这种系统的设计，并开发了一种不需要。

    Decision support systems for classification tasks are predominantly designed to predict the value of the ground truth labels. However, since their predictions are not perfect, these systems also need to make human experts understand when and how to use these predictions to update their own predictions. Unfortunately, this has been proven challenging. In this context, it has been recently argued that an alternative type of decision support systems may circumvent this challenge. Rather than providing a single label prediction, these systems provide a set of label prediction values constructed using a conformal predictor, namely a prediction set, and forcefully ask experts to predict a label value from the prediction set. However, the design and evaluation of these systems have so far relied on stylized expert models, questioning their promise. In this paper, we revisit the design of this type of systems from the perspective of online learning and develop a methodology that does not requi
    

