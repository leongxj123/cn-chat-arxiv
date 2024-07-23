# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Emotion Detection with Transformers: A Comparative Study](https://arxiv.org/abs/2403.15454) | 本研究探索了在文本数据情感分类中应用基于Transformer的模型，并发现常用技术如去除标点符号和停用词可能会阻碍模型的性能，因为这些元素仍然能够传达情感或强调，而Transformer的优势在于理解文本内的语境关系。 |
| [^2] | [Resolution of Simpson's paradox via the common cause principle](https://arxiv.org/abs/2403.00957) | 通过对共同原因$C$进行条件设定，解决了辛普森悖论，推广了悖论，并表明在二元共同原因$C$上进行条件设定的关联方向与原始$B$上进行条件设定相同 |

# 详细

[^1]: 使用Transformer进行情感检测：一项比较研究

    Emotion Detection with Transformers: A Comparative Study

    [https://arxiv.org/abs/2403.15454](https://arxiv.org/abs/2403.15454)

    本研究探索了在文本数据情感分类中应用基于Transformer的模型，并发现常用技术如去除标点符号和停用词可能会阻碍模型的性能，因为这些元素仍然能够传达情感或强调，而Transformer的优势在于理解文本内的语境关系。

    

    在这项研究中，我们探讨了基于Transformer模型在文本数据情感分类中的应用。我们使用不同变体的Transformer对Emotion数据集进行训练和评估。论文还分析了一些影响模型性能的因素，比如Transformer层的微调、层的可训练性以及文本数据的预处理。我们的分析表明，常用技术如去除标点符号和停用词可能会阻碍模型的性能。这可能是因为Transformer的优势在于理解文本内的语境关系。像标点符号和停用词这样的元素仍然可以传达情感或强调，去除它们可能会破坏这种上下文。

    arXiv:2403.15454v1 Announce Type: new  Abstract: In this study, we explore the application of transformer-based models for emotion classification on text data. We train and evaluate several pre-trained transformer models, on the Emotion dataset using different variants of transformers. The paper also analyzes some factors that in-fluence the performance of the model, such as the fine-tuning of the transformer layer, the trainability of the layer, and the preprocessing of the text data. Our analysis reveals that commonly applied techniques like removing punctuation and stop words can hinder model performance. This might be because transformers strength lies in understanding contextual relationships within text. Elements like punctuation and stop words can still convey sentiment or emphasis and removing them might disrupt this context.
    
[^2]: 利用共因原则解决辛普森悖论

    Resolution of Simpson's paradox via the common cause principle

    [https://arxiv.org/abs/2403.00957](https://arxiv.org/abs/2403.00957)

    通过对共同原因$C$进行条件设定，解决了辛普森悖论，推广了悖论，并表明在二元共同原因$C$上进行条件设定的关联方向与原始$B$上进行条件设定相同

    

    辛普森悖论是建立两个事件$a_1$和$a_2$之间的概率关联时的障碍，给定第三个（潜在的）随机变量$B$。我们关注的情景是随机变量$A$（汇总了$a_1$、$a_2$及其补集）和$B$有一个可能未被观察到的共同原因$C$。或者，我们可以假设$C$将$A$从$B$中筛选出去。对于这种情况，正确的$a_1$和$a_2$之间的关联应该通过对$C$进行条件设定来定义。这一设置将原始辛普森悖论推广了。现在它的两个相互矛盾的选项简单地指的是两个特定且不同的原因$C$。我们表明，如果$B$和$C$是二进制的，$A$是四进制的（对于有效的辛普森悖论来说是最小且最常见的情况），在任何二元共同原因$C$上进行条件设定将建立与在原始$B$上进行条件设定相同的$a_1$和$a_2$之间的关联方向。

    arXiv:2403.00957v1 Announce Type: cross  Abstract: Simpson's paradox is an obstacle to establishing a probabilistic association between two events $a_1$ and $a_2$, given the third (lurking) random variable $B$. We focus on scenarios when the random variables $A$ (which combines $a_1$, $a_2$, and their complements) and $B$ have a common cause $C$ that need not be observed. Alternatively, we can assume that $C$ screens out $A$ from $B$. For such cases, the correct association between $a_1$ and $a_2$ is to be defined via conditioning over $C$. This set-up generalizes the original Simpson's paradox. Now its two contradicting options simply refer to two particular and different causes $C$. We show that if $B$ and $C$ are binary and $A$ is quaternary (the minimal and the most widespread situation for valid Simpson's paradox), the conditioning over any binary common cause $C$ establishes the same direction of the association between $a_1$ and $a_2$ as the conditioning over $B$ in the original
    

