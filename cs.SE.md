# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UniASM: Binary Code Similarity Detection without Fine-tuning.](http://arxiv.org/abs/2211.01144) | 提出了一种新的二进制代码嵌入模型UniASM，并设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，提出了一种新的二进制函数tokenization方法，缓解了词汇外的问题，并通过消融实验得到了一些新的有价值的发现，实验证明UniASM优于其他模型。 |

# 详细

[^1]: UniASM：无需微调的二进制代码相似性检测

    UniASM: Binary Code Similarity Detection without Fine-tuning. (arXiv:2211.01144v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.01144](http://arxiv.org/abs/2211.01144)

    提出了一种新的二进制代码嵌入模型UniASM，并设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，提出了一种新的二进制函数tokenization方法，缓解了词汇外的问题，并通过消融实验得到了一些新的有价值的发现，实验证明UniASM优于其他模型。

    

    二进制代码相似性检测被广泛用于各种二进制分析任务，如漏洞搜索、恶意软件检测、克隆检测和补丁分析。最近的研究表明，基于学习的二进制代码嵌入模型比传统的基于特征的方法更好。本文提出了一种新的基于transformer的二进制代码嵌入模型UniASM，用于学习二进制函数的表示。我们设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，我们提出了一种新的二进制函数tokenization方法，增加了tokens的语义信息并缓解了词汇外的问题。通过消融实验进行了深入分析，得到了一些新的有价值的发现，实验证明UniASM优于其他模型。

    Binary code similarity detection (BCSD) is widely used in various binary analysis tasks such as vulnerability search, malware detection, clone detection, and patch analysis. Recent studies have shown that the learning-based binary code embedding models perform better than the traditional feature-based approaches. In this paper, we propose a novel transformer-based binary code embedding model named UniASM to learn representations of the binary functions. We design two new training tasks to make the spatial distribution of the generated vectors more uniform, which can be used directly in BCSD without any fine-tuning. In addition, we present a new tokenization approach for binary functions, which increases the token's semantic information and mitigates the out-of-vocabulary (OOV) problem. We conduct an in-depth analysis of the factors affecting model performance through ablation experiments and obtain some new and valuable findings. The experimental results show that UniASM outperforms th
    

