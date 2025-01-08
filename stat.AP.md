# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Modeling for Tabular Data via Penalized Optimal Transport Network](https://arxiv.org/abs/2402.10456) | 提出了一种名为POTNet的生成建模网络，基于边缘惩罚的Wasserstein损失，能够有效地建模同时包含分类和连续特征的表格数据。 |

# 详细

[^1]: 通过惩罚最优输运网络对表格数据进行生成建模

    Generative Modeling for Tabular Data via Penalized Optimal Transport Network

    [https://arxiv.org/abs/2402.10456](https://arxiv.org/abs/2402.10456)

    提出了一种名为POTNet的生成建模网络，基于边缘惩罚的Wasserstein损失，能够有效地建模同时包含分类和连续特征的表格数据。

    

    准确学习表格数据中行的概率分布并生成真实的合成样本的任务既关键又非平凡。Wasserstein生成对抗网络(WGAN)在生成建模中取得了显著进展，解决了其前身生成对抗网络所面临的挑战。然而，由于表格数据中存在混合数据类型和多模态性，生成器和鉴别器之间的微妙平衡以及Wasserstein距离在高维度中的固有不稳定性，WGAN通常无法生成高保真样本。因此，我们提出了POTNet（惩罚最优输运网络），这是一种基于新颖、强大且可解释的边际惩罚Wasserstein（MPW）损失的生成深度神经网络。POTNet能够有效地建模包含分类和连续特征的表格数据。

    arXiv:2402.10456v1 Announce Type: cross  Abstract: The task of precisely learning the probability distribution of rows within tabular data and producing authentic synthetic samples is both crucial and non-trivial. Wasserstein generative adversarial network (WGAN) marks a notable improvement in generative modeling, addressing the challenges faced by its predecessor, generative adversarial network. However, due to the mixed data types and multimodalities prevalent in tabular data, the delicate equilibrium between the generator and discriminator, as well as the inherent instability of Wasserstein distance in high dimensions, WGAN often fails to produce high-fidelity samples. To this end, we propose POTNet (Penalized Optimal Transport Network), a generative deep neural network based on a novel, robust, and interpretable marginally-penalized Wasserstein (MPW) loss. POTNet can effectively model tabular data containing both categorical and continuous features. Moreover, it offers the flexibil
    

