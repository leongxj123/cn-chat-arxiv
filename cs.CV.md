# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer](https://arxiv.org/abs/2403.19979) | 适配器调整方法在持续学习中展现出较优性能，提出了增量调整共享适配器和利用存储原型进行特征采样和更新的方法来增强模型学习能力。 |

# 详细

[^1]: 语义转移增量适配器调整是一种持续的 ViTransformer

    Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer

    [https://arxiv.org/abs/2403.19979](https://arxiv.org/abs/2403.19979)

    适配器调整方法在持续学习中展现出较优性能，提出了增量调整共享适配器和利用存储原型进行特征采样和更新的方法来增强模型学习能力。

    

    类增量学习（CIL）旨在使模型能够在克服灾难性遗忘的同时持续学习新的类别。本文重新审视了在持续学习背景下的不同参数高效调整（PET）方法。我们观察到适配器调整表现优于基于提示的方法，甚至在每个学习会话中没有参数扩展的情况下也如此。受此启发，我们提出了增量调整共享适配器而不施加参数更新约束，增强骨干的学习能力。此外，我们从存储的原型中抽取特征样本来重新训练统一的分类器，进一步提高其性能。我们估计旧原型的语义转移，而无法访问过去的样本，并逐个会话更新存储的原型。我们提出的方法消除了模型的扩展和...

    arXiv:2403.19979v1 Announce Type: cross  Abstract: Class-incremental learning (CIL) aims to enable models to continuously learn new classes while overcoming catastrophic forgetting. The introduction of pre-trained models has brought new tuning paradigms to CIL. In this paper, we revisit different parameter-efficient tuning (PET) methods within the context of continual learning. We observe that adapter tuning demonstrates superiority over prompt-based methods, even without parameter expansion in each learning session. Motivated by this, we propose incrementally tuning the shared adapter without imposing parameter update constraints, enhancing the learning capacity of the backbone. Additionally, we employ feature sampling from stored prototypes to retrain a unified classifier, further improving its performance. We estimate the semantic shift of old prototypes without access to past samples and update stored prototypes session by session. Our proposed method eliminates model expansion and
    

