# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Calibration Gap between Model and Human Confidence in Large Language Models.](http://arxiv.org/abs/2401.13835) | 该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。 |

# 详细

[^1]: 语言模型中模型和人类置信度之间的校准差距

    The Calibration Gap between Model and Human Confidence in Large Language Models. (arXiv:2401.13835v1 [cs.LG])

    [http://arxiv.org/abs/2401.13835](http://arxiv.org/abs/2401.13835)

    该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。

    

    为了使大型语言模型（LLM）能够获得人类的信任，它们需要在某种意义上实现良好的校准，即能够准确评估和传达它们的预测正确的可能性。最近的研究关注了LLM内部置信度评估的质量，但问题仍然是LLM能够如何将这种内部模型置信度传达给人类用户。本文探讨了人类对LLM响应的外部置信度与模型内部置信度之间的差距。通过涉及多项选择题的实验，我们系统地检查了人类用户识别LLM输出可信度的能力。我们的研究重点分为两个方面：（1）评估用户对真实LLM置信度的感知和（2）调查个性化解释对该感知的影响。研究结果显示，LLM的默认解释往往会导致用户过高估计模型的置信度和准确性。通过修改解释的方式可以减小这种误差。

    For large language models (LLMs) to be trusted by humans they need to be well-calibrated in the sense that they can accurately assess and communicate how likely it is that their predictions are correct. Recent work has focused on the quality of internal LLM confidence assessments, but the question remains of how well LLMs can communicate this internal model confidence to human users. This paper explores the disparity between external human confidence in an LLM's responses and the internal confidence of the model. Through experiments involving multiple-choice questions, we systematically examine human users' ability to discern the reliability of LLM outputs. Our study focuses on two key areas: (1) assessing users' perception of true LLM confidence and (2) investigating the impact of tailored explanations on this perception. The research highlights that default explanations from LLMs often lead to user overestimation of both the model's confidence and its' accuracy. By modifying the expl
    

