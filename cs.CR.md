# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Transferability of Adversarial Examples via Bayesian Attacks.](http://arxiv.org/abs/2307.11334) | 通过将贝叶斯公式应用于模型参数和模型输入，本文提出了一种改进对抗性样本可迁移性的方法，实证研究表明具有显著提高效果，并超过了当前最新技术。 |

# 详细

[^1]: 通过贝叶斯攻击提高对抗性样本的可迁移性

    Improving Transferability of Adversarial Examples via Bayesian Attacks. (arXiv:2307.11334v1 [cs.LG])

    [http://arxiv.org/abs/2307.11334](http://arxiv.org/abs/2307.11334)

    通过将贝叶斯公式应用于模型参数和模型输入，本文提出了一种改进对抗性样本可迁移性的方法，实证研究表明具有显著提高效果，并超过了当前最新技术。

    

    本文是对我们在ICLR上发表工作的重要扩展。我们的ICLR工作提出了将贝叶斯公式应用于模型参数，以提高对抗性样本的可迁移性，从而有效模拟了无限多个深度神经网络的集合。而在这篇论文中，我们通过将贝叶斯公式应用于模型输入，引入了一种新颖的扩展，使得模型输入和模型参数都能够进行联合多样化。我们的实证研究证明：1）对模型输入和模型参数同时应用贝叶斯公式可以显著提高可迁移性；2）通过引入对模型输入后验分布的高级近似，攻击无需模型微调时，对抗性可迁移性得到进一步提升，超过了所有的最新技术。此外，我们还提出了一种有原则的方法来对模型参数进行微调。

    This paper presents a substantial extension of our work published at ICLR. Our ICLR work advocated for enhancing transferability in adversarial examples by incorporating a Bayesian formulation into model parameters, which effectively emulates the ensemble of infinitely many deep neural networks, while, in this paper, we introduce a novel extension by incorporating the Bayesian formulation into the model input as well, enabling the joint diversification of both the model input and model parameters. Our empirical findings demonstrate that: 1) the combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability; 2) by introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Moreover, we propose a principled approach to fine-tune model parameters in such an ext
    

