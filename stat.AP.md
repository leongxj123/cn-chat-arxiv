# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-shot sampling of adversarial entities in biomedical question answering](https://arxiv.org/abs/2402.10527) | 在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。 |
| [^2] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |

# 详细

[^1]: 生物医学问题回答中的零样本采样对抗实体

    Zero-shot sampling of adversarial entities in biomedical question answering

    [https://arxiv.org/abs/2402.10527](https://arxiv.org/abs/2402.10527)

    在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。

    

    大型语言模型（LLM）中参数域知识的增加深度推动它们在现实世界应用中的快速部署。在高风险和知识密集型任务中，理解模型的漏洞对于量化模型预测的可信度和规范其使用至关重要。最近发现在自然语言处理任务中作为对抗示例的命名实体引发了关于它们在其他环境中可能的伪装的疑问。在这里，我们提出了一种在嵌入空间中的幂缩放距离加权采样方案，以发现多样化的对抗实体作为干扰因素。我们展示了它在生物医学主题的对抗性问题回答中优于随机采样的优势。我们的方法使得可以探索攻击表面上的不同区域，这揭示了两种在特征上明显不同的对抗性实体的制度。此外，我们展示了攻击方式如何...

    arXiv:2402.10527v1 Announce Type: new  Abstract: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks su
    
[^2]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    

