# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Closed-Loop Unsupervised Representation Disentanglement with $\beta$-VAE Distillation and Diffusion Probabilistic Feedback](https://arxiv.org/abs/2402.02346) | 本文提出了闭环无监督表示解缠方法CL-Dis，使用扩散自动编码器（Diff-AE）和β-VAE共同提取语义解缠表示，以解决表示解缠面临的问题。 |

# 详细

[^1]: 闭环无监督表示解缠的β-VAE蒸馏与扩散概率反馈

    Closed-Loop Unsupervised Representation Disentanglement with $\beta$-VAE Distillation and Diffusion Probabilistic Feedback

    [https://arxiv.org/abs/2402.02346](https://arxiv.org/abs/2402.02346)

    本文提出了闭环无监督表示解缠方法CL-Dis，使用扩散自动编码器（Diff-AE）和β-VAE共同提取语义解缠表示，以解决表示解缠面临的问题。

    

    表示解缠可能有助于AI根本上理解现实世界，从而使判别和生成任务受益。目前至少有三个未解决的核心问题：（i）过于依赖标签注释和合成数据-导致在自然情景下泛化能力较差；（ii）启发式/手工制作的解缠约束使得难以自适应地实现最佳训练权衡；（iii）缺乏合理的评估指标，特别是对于真实的无标签数据。为了解决这些挑战，我们提出了一种被称为CL-Dis的闭环无监督表示解缠方法。具体地，我们使用基于扩散的自动编码器（Diff-AE）作为骨干，并使用β-VAE作为副驾驶员来提取语义解缠的表示。扩散模型的强大生成能力和VAE模型的良好解缠能力是互补的。为了加强解缠，使用VAE潜变量。

    Representation disentanglement may help AI fundamentally understand the real world and thus benefit both discrimination and generation tasks. It currently has at least three unresolved core issues: (i) heavy reliance on label annotation and synthetic data -- causing poor generalization on natural scenarios; (ii) heuristic/hand-craft disentangling constraints make it hard to adaptively achieve an optimal training trade-off; (iii) lacking reasonable evaluation metric, especially for the real label-free data. To address these challenges, we propose a \textbf{C}losed-\textbf{L}oop unsupervised representation \textbf{Dis}entanglement approach dubbed \textbf{CL-Dis}. Specifically, we use diffusion-based autoencoder (Diff-AE) as a backbone while resorting to $\beta$-VAE as a co-pilot to extract semantically disentangled representations. The strong generation ability of diffusion model and the good disentanglement ability of VAE model are complementary. To strengthen disentangling, VAE-latent 
    

