# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time (Extended Version)](https://rss.arxiv.org/abs/2402.01359) | 本文提出TESSERACT方法，消除了恶意软件分类中的实验偏差，解决了常见的空间和时间偏差问题。通过公平实验设计约束和新指标AUT实现了更准确和稳定的分类器。 |
| [^2] | [A Survey of Source Code Representations for Machine Learning-Based Cybersecurity Tasks](https://arxiv.org/abs/2403.10646) | 本研究调查了用于基于机器学习的网络安全任务的源代码表示方法，发现基于图的表示是最受欢迎的，而分词器和抽象语法树是最流行的两种表示方法。 |

# 详细

[^1]: TESSERACT: 消除恶意软件分类中的实验偏差的空间和时间方法（扩展版）

    TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time (Extended Version)

    [https://rss.arxiv.org/abs/2402.01359](https://rss.arxiv.org/abs/2402.01359)

    本文提出TESSERACT方法，消除了恶意软件分类中的实验偏差，解决了常见的空间和时间偏差问题。通过公平实验设计约束和新指标AUT实现了更准确和稳定的分类器。

    

    机器学习在检测恶意软件方面扮演着关键角色。尽管许多研究报告的F1分数高达0.99，但问题仍未完全解决。恶意软件检测器常常在操作系统和攻击方法不断演化时出现性能下降，这会导致之前学习到的知识对于新输入的准确决策变得不足够。本文认为常见的研究结果由于检测任务中的两种普遍的实验偏差而被夸大：空间偏差是由数据分布不代表真实部署的情况引起的；时间偏差是由于数据的不正确时间分割引起的，导致了不现实的配置。为了解决这些偏差，我们引入了一组公平实验设计的约束，并提出了一个用于在真实环境中评估分类器稳定性的新指标AUT。我们还提出了一种用于调整训练数据以提高分类器鲁棒性的算法。

    Machine learning (ML) plays a pivotal role in detecting malicious software. Despite the high F1-scores reported in numerous studies reaching upwards of 0.99, the issue is not completely solved. Malware detectors often experience performance decay due to constantly evolving operating systems and attack methods, which can render previously learned knowledge insufficient for accurate decision-making on new inputs. This paper argues that commonly reported results are inflated due to two pervasive sources of experimental bias in the detection task: spatial bias caused by data distributions that are not representative of a real-world deployment; and temporal bias caused by incorrect time splits of data, leading to unrealistic configurations. To address these biases, we introduce a set of constraints for fair experiment design, and propose a new metric, AUT, for classifier robustness in real-world settings. We additionally propose an algorithm designed to tune training data to enhance classif
    
[^2]: 用于基于机器学习的网络安全任务的源代码表示调查

    A Survey of Source Code Representations for Machine Learning-Based Cybersecurity Tasks

    [https://arxiv.org/abs/2403.10646](https://arxiv.org/abs/2403.10646)

    本研究调查了用于基于机器学习的网络安全任务的源代码表示方法，发现基于图的表示是最受欢迎的，而分词器和抽象语法树是最流行的两种表示方法。

    

    arXiv:2403.10646v1 公告类型: 新的 摘要: 用于网络安全相关软件工程任务的机器学习技术越来越受欢迎。源代码的表示是该技术的关键部分，可以影响模型学习源代码特征的方式。随着越来越多的这些技术被开发，了解该领域的当前状态对于更好地理解已有内容和尚未存在的内容是有价值的。本文对这些现有的基于ML的方法进行了研究，展示了不同网络安全任务和编程语言所使用的表示类型。此外，我们研究了不同表示所使用的模型类型。我们发现基于图的表示是最受欢迎的表示类别，而分词器和抽象语法树（ASTs）是最受欢迎的表示方法。我们还发现网络安全领域最受欢迎的

    arXiv:2403.10646v1 Announce Type: new  Abstract: Machine learning techniques for cybersecurity-related software engineering tasks are becoming increasingly popular. The representation of source code is a key portion of the technique that can impact the way the model is able to learn the features of the source code. With an increasing number of these techniques being developed, it is valuable to see the current state of the field to better understand what exists and what's not there yet. This paper presents a study of these existing ML-based approaches and demonstrates what type of representations were used for different cybersecurity tasks and programming languages. Additionally, we study what types of models are used with different representations. We have found that graph-based representations are the most popular category of representation, and Tokenizers and Abstract Syntax Trees (ASTs) are the two most popular representations overall. We also found that the most popular cybersecur
    

