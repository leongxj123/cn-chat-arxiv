# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning from Invalid Data: On Constraint Satisfaction in Generative Models.](http://arxiv.org/abs/2306.15166) | 本论文提出了一种新的训练机制，利用包含无效数据点的数据集进行生成模型的训练，以提高生成结果的精度和满足约束条件的能力。实验证明，与只使用有效数据点进行训练的标准模型相比，基于无效数据训练的模型明显优于标准模型。 |

# 详细

[^1]: 从无效数据中学习：关于生成模型中的约束满足问题

    Learning from Invalid Data: On Constraint Satisfaction in Generative Models. (arXiv:2306.15166v1 [cs.LG])

    [http://arxiv.org/abs/2306.15166](http://arxiv.org/abs/2306.15166)

    本论文提出了一种新的训练机制，利用包含无效数据点的数据集进行生成模型的训练，以提高生成结果的精度和满足约束条件的能力。实验证明，与只使用有效数据点进行训练的标准模型相比，基于无效数据训练的模型明显优于标准模型。

    

    生成模型在视觉、语言和语音等领域取得了令人印象深刻的结果。然而，即使有大量的数据集，它们仍然在精度上存在困难，生成出物理上无效或事实上不正确的数据。当生成的数据必须满足约束条件时，这一问题尤为严重，例如，在工程设计中满足产品规格或者在自然场景中遵守物理定律。为了提高精度并保持多样性和保真度，我们提出了一种新的训练机制，利用包含无效数据点的数据集进行训练。我们的方法最小化了生成分布与有效先验之间的差异，同时最大化了与无效分布之间的差异。我们演示了将GAN和DDPM等生成模型与无效数据一起训练的方法明显优于仅使用有效数据点进行训练的标准模型。例如，我们的训练过程生成了……

    Generative models have demonstrated impressive results in vision, language, and speech. However, even with massive datasets, they struggle with precision, generating physically invalid or factually incorrect data. This is particularly problematic when the generated data must satisfy constraints, for example, to meet product specifications in engineering design or to adhere to the laws of physics in a natural scene. To improve precision while preserving diversity and fidelity, we propose a novel training mechanism that leverages datasets of constraint-violating data points, which we consider invalid. Our approach minimizes the divergence between the generative distribution and the valid prior while maximizing the divergence with the invalid distribution. We demonstrate how generative models like GANs and DDPMs that we augment to train with invalid data vastly outperform their standard counterparts which solely train on valid data points. For example, our training procedure generates up 
    

