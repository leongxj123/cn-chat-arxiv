# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Auditing for Human Expertise.](http://arxiv.org/abs/2306.01646) | 人类专家的价值超出了算法可捕捉范围，我们可以用一个简单的程序测试这个问题。 |

# 详细

[^1]: 人类专家审核研究

    Auditing for Human Expertise. (arXiv:2306.01646v1 [stat.ML])

    [http://arxiv.org/abs/2306.01646](http://arxiv.org/abs/2306.01646)

    人类专家的价值超出了算法可捕捉范围，我们可以用一个简单的程序测试这个问题。

    

    高风险预测任务（例如患者诊断）通常由接受培训的人类专家处理。在这些设置中，自动化的一个常见问题是，专家可能运用很难建模的直觉，并且/或者可以获取信息（例如与患者的交谈），这些信息对于算法来说是不可用的。这引发了一个自然的问题，人类专家是否增加了无法被算法预测器捕捉到的价值。我们开发了一个统计框架，可以将这个问题提出为一个自然的假设检验。正如我们的框架所强调的那样，检测人类专业知识比简单比较专家预测准确性与特定学习算法做出的准确性更加微妙。而是提出了一个简单的程序，测试专家预测是否在“特征”可用而条件下是否与感兴趣的结果统计上独立。因此，我们测试的拒绝表明了人类专业知识确实增加了超出算法可捕捉范围的价值。

    High-stakes prediction tasks (e.g., patient diagnosis) are often handled by trained human experts. A common source of concern about automation in these settings is that experts may exercise intuition that is difficult to model and/or have access to information (e.g., conversations with a patient) that is simply unavailable to a would-be algorithm. This raises a natural question whether human experts add value which could not be captured by an algorithmic predictor. We develop a statistical framework under which we can pose this question as a natural hypothesis test. Indeed, as our framework highlights, detecting human expertise is more subtle than simply comparing the accuracy of expert predictions to those made by a particular learning algorithm. Instead, we propose a simple procedure which tests whether expert predictions are statistically independent from the outcomes of interest after conditioning on the available inputs (`features'). A rejection of our test thus suggests that huma
    

