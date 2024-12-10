# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein.](http://arxiv.org/abs/2401.06199) | xTrimoPGLM是一个统一的100亿规模预训练蛋白质语言模型，能够同时处理蛋白质理解和生成任务，通过创新的预训练框架和大规模的参数训练，显著优于其他先进模型，在18个蛋白理解基准测试中取得了成功，并能够实现对蛋白质结构的原子分辨率观察。 |

# 详细

[^1]: xTrimoPGLM: 统一的百亿规模预训练蛋白质语言模型，用于解析蛋白质的语言

    xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein. (arXiv:2401.06199v1 [q-bio.QM])

    [http://arxiv.org/abs/2401.06199](http://arxiv.org/abs/2401.06199)

    xTrimoPGLM是一个统一的100亿规模预训练蛋白质语言模型，能够同时处理蛋白质理解和生成任务，通过创新的预训练框架和大规模的参数训练，显著优于其他先进模型，在18个蛋白理解基准测试中取得了成功，并能够实现对蛋白质结构的原子分辨率观察。

    

    蛋白质语言模型在学习蛋白质序列中的生物信息方面显示出显著的成功。然而，大多数现有模型局限于自编码或自回归的预训练目标，这使得它们在处理蛋白质理解和生成任务时很难同时进行。我们提出了一个统一的蛋白质语言模型，xTrimoPGLM，通过创新的预训练框架同时解决这两类任务。我们的关键技术贡献是探索这两类目标的兼容性和联合优化的潜力，从而导致了一个以前所未有的规模，使用1000亿参数和1万亿训练标记来训练xTrimoPGLM的策略。我们广泛的实验证明，1）xTrimoPGLM在四个类别的18个蛋白理解基准测试中明显优于其他先进基线。该模型还有助于对蛋白质结构进行原子分辨率的观察，从而实现了对蛋白质结构的理解和生成。

    Protein language models have shown remarkable success in learning biological information from protein sequences. However, most existing models are limited by either autoencoding or autoregressive pre-training objectives, which makes them struggle to handle protein understanding and generation tasks concurrently. We propose a unified protein language model, xTrimoPGLM, to address these two types of tasks simultaneously through an innovative pre-training framework. Our key technical contribution is an exploration of the compatibility and the potential for joint optimization of the two types of objectives, which has led to a strategy for training xTrimoPGLM at an unprecedented scale of 100 billion parameters and 1 trillion training tokens. Our extensive experiments reveal that 1) xTrimoPGLM significantly outperforms other advanced baselines in 18 protein understanding benchmarks across four categories. The model also facilitates an atomic-resolution view of protein structures, leading to 
    

