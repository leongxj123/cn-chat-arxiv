# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Examining the Influence of Varied Levels of Domain Knowledge Base Inclusion in GPT-based Intelligent Tutors.](http://arxiv.org/abs/2309.12367) | 本文研究了在基于GPT的智能辅导系统中将领域知识库与语言模型集成，以提高回答的可靠性。通过设计可扩展的知识库和评估实验，我们展示了该系统的有效性。学生和领域专家对于智能辅导系统的回答进行了验证和排名。 |
| [^2] | [Designing Decision Support Systems Using Counterfactual Prediction Sets.](http://arxiv.org/abs/2306.03928) | 本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。 |

# 详细

[^1]: 在基于GPT的智能辅导系统中研究领域知识库不同程度的影响

    Examining the Influence of Varied Levels of Domain Knowledge Base Inclusion in GPT-based Intelligent Tutors. (arXiv:2309.12367v1 [cs.HC])

    [http://arxiv.org/abs/2309.12367](http://arxiv.org/abs/2309.12367)

    本文研究了在基于GPT的智能辅导系统中将领域知识库与语言模型集成，以提高回答的可靠性。通过设计可扩展的知识库和评估实验，我们展示了该系统的有效性。学生和领域专家对于智能辅导系统的回答进行了验证和排名。

    

    最近大型语言模型（LLM）的进展促进了具有复杂对话能力的聊天机器人的发展。然而，LLM对查询的回答经常不准确，这限制了在教育环境中的应用。本文研究了将知识库（KB）与LLM智能辅导系统集成以增加回答可靠性的效果。为了实现这一目标，我们设计了一个可扩展的知识库，教育监督员可以无缝集成课程，该课程会被智能辅导系统自动处理。然后，我们详细介绍了一个评估实验，学生参与者需要回答有关人工智能课程的问题。 GPT-4智能辅导系统具有不同层次的KB访问权限，并由人类领域专家评估这些回答。最后，学生对智能辅导系统的回答进行了与领域专家的交叉验证，并对它们的各种教学能力进行了排名。

    Recent advancements in large language models (LLMs) have facilitated the development of chatbots with sophisticated conversational capabilities. However, LLMs exhibit frequent inaccurate responses to queries, hindering applications in educational settings. In this paper, we investigate the effectiveness of integrating a knowledge base (KB) with LLM intelligent tutors to increase response reliability. To achieve this, we design a scaleable KB that affords educational supervisors seamless integration of lesson curricula, which is automatically processed by the intelligent tutoring system. We then detail an evaluation, where student participants were presented with questions about the artificial intelligence curriculum to respond to. GPT-4 intelligent tutors with varying hierarchies of KB access and human domain experts then assessed these responses. Lastly, students cross-examined the intelligent tutors' responses to the domain experts' and ranked their various pedagogical abilities. Res
    
[^2]: 使用反事实预测集设计决策支持系统

    Designing Decision Support Systems Using Counterfactual Prediction Sets. (arXiv:2306.03928v1 [cs.LG])

    [http://arxiv.org/abs/2306.03928](http://arxiv.org/abs/2306.03928)

    本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。

    

    分类任务的决策支持系统通常被设计用于预测地面实况标签的值。然而，由于它们的预测并不完美，这些系统还需要让人类专家了解何时以及如何使用这些预测来更新自己的预测。不幸的是，这被证明是具有挑战性的。最近有人认为，另一种类型的决策支持系统可能会避开这个挑战。这些系统不是提供单个标签预测，而是使用符合预测器构建一组标签预测值，即预测集，并强制要求专家从预测集中预测一个标签值。然而，这些系统的设计和评估迄今仍依赖于样式化的专家模型，这引发了人们对它们的承诺的质疑。本文从在线学习的角度重新审视了这种系统的设计，并开发了一种不需要。

    Decision support systems for classification tasks are predominantly designed to predict the value of the ground truth labels. However, since their predictions are not perfect, these systems also need to make human experts understand when and how to use these predictions to update their own predictions. Unfortunately, this has been proven challenging. In this context, it has been recently argued that an alternative type of decision support systems may circumvent this challenge. Rather than providing a single label prediction, these systems provide a set of label prediction values constructed using a conformal predictor, namely a prediction set, and forcefully ask experts to predict a label value from the prediction set. However, the design and evaluation of these systems have so far relied on stylized expert models, questioning their promise. In this paper, we revisit the design of this type of systems from the perspective of online learning and develop a methodology that does not requi
    

