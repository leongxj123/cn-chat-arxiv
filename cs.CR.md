# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Trigger Backdoor Attacks: More Triggers, More Threats.](http://arxiv.org/abs/2401.15295) | 本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。 |

# 详细

[^1]: 多触发后门攻击：更多触发器，更多威胁

    Multi-Trigger Backdoor Attacks: More Triggers, More Threats. (arXiv:2401.15295v1 [cs.LG])

    [http://arxiv.org/abs/2401.15295](http://arxiv.org/abs/2401.15295)

    本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。

    

    后门攻击已经成为深度神经网络（DNNs）的（预）训练和部署的主要威胁。尽管后门攻击在一些研究中已经得到了广泛的探讨，但其中大部分都集中在使用单个类型的触发器来污染数据集的单触发攻击上。可以说，在现实世界中，后门攻击可能更加复杂，例如，同一数据集可能存在多个对手，如果该数据集具有较高的价值。在这项工作中，我们研究了在多触发攻击设置下后门攻击的实际威胁，多个对手利用不同类型的触发器来污染同一数据集。通过提出和研究并行、顺序和混合攻击这三种类型的多触发攻击，我们提供了关于不同触发器对同一数据集的共存、覆写和交叉激活效果的重要认识。此外，我们还展示了单触发攻击往往容易引起覆写问题。

    Backdoor attacks have emerged as a primary threat to (pre-)training and deployment of deep neural networks (DNNs). While backdoor attacks have been extensively studied in a body of works, most of them were focused on single-trigger attacks that poison a dataset using a single type of trigger. Arguably, real-world backdoor attacks can be much more complex, e.g., the existence of multiple adversaries for the same dataset if it is of high value. In this work, we investigate the practical threat of backdoor attacks under the setting of \textbf{multi-trigger attacks} where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks, including parallel, sequential, and hybrid attacks, we provide a set of important understandings of the coexisting, overwriting, and cross-activating effects between different triggers on the same dataset. Moreover, we show that single-trigger attacks tend to cause over
    

