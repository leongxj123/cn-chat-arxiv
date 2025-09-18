# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tabdoor: Backdoor Vulnerabilities in Transformer-based Neural Networks for Tabular Data.](http://arxiv.org/abs/2311.07550) | 这项研究全面分析了使用DNNs对表格数据进行后门攻击，揭示了基于转换器的DNNs对表格数据非常容易受到后门攻击，甚至只需最小的特征值修改。该攻击还可以推广到其他模型。 |

# 详细

[^1]: Tabdoor：基于转换器的表格数据神经网络存在后门漏洞

    Tabdoor: Backdoor Vulnerabilities in Transformer-based Neural Networks for Tabular Data. (arXiv:2311.07550v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2311.07550](http://arxiv.org/abs/2311.07550)

    这项研究全面分析了使用DNNs对表格数据进行后门攻击，揭示了基于转换器的DNNs对表格数据非常容易受到后门攻击，甚至只需最小的特征值修改。该攻击还可以推广到其他模型。

    

    深度神经网络(DNNs)在各个领域都显示出巨大的潜力。与这些发展同时，与DNN训练相关的漏洞，如后门攻击，是一个重大关切。这些攻击涉及在模型训练过程中微妙地插入触发器，从而允许操纵预测。最近，由于转换器模型的崛起，DNNs用于表格数据越来越受关注。我们的研究对使用DNNs对表格数据进行后门攻击进行了全面分析，特别关注转换器。鉴于表格数据的固有复杂性，我们探究了嵌入后门的挑战。通过对基准数据集进行系统实验，我们发现基于转换器的DNNs对表格数据非常容易受到后门攻击，即使只有最小的特征值修改。我们还验证了我们的攻击可以推广到其他模型，如XGBoost和DeepFM。我们的研究结果几乎表明后门攻击可以完美实现。

    Deep Neural Networks (DNNs) have shown great promise in various domains. Alongside these developments, vulnerabilities associated with DNN training, such as backdoor attacks, are a significant concern. These attacks involve the subtle insertion of triggers during model training, allowing for manipulated predictions.More recently, DNNs for tabular data have gained increasing attention due to the rise of transformer models.  Our research presents a comprehensive analysis of backdoor attacks on tabular data using DNNs, particularly focusing on transformers. Given the inherent complexities of tabular data, we explore the challenges of embedding backdoors. Through systematic experimentation across benchmark datasets, we uncover that transformer-based DNNs for tabular data are highly susceptible to backdoor attacks, even with minimal feature value alterations. We also verify that our attack can be generalized to other models, like XGBoost and DeepFM. Our results indicate nearly perfect attac
    

