# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diverse Neural Audio Embeddings -- Bringing Features back !.](http://arxiv.org/abs/2309.08751) | 本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。 |
| [^2] | [JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models.](http://arxiv.org/abs/2308.04729) | JEN-1是一个高保真度通用音乐生成模型，通过结合自回归和非自回归训练，实现了文本引导的音乐生成、音乐修补和延续等生成任务，在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。 |

# 详细

[^1]: 多样的神经音频嵌入 - 恢复特征！

    Diverse Neural Audio Embeddings -- Bringing Features back !. (arXiv:2309.08751v1 [cs.SD])

    [http://arxiv.org/abs/2309.08751](http://arxiv.org/abs/2309.08751)

    本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。

    

    随着现代人工智能架构的出现，从端到端的架构开始流行。这种转变导致了神经架构在没有领域特定偏见/知识的情况下进行训练，根据任务进行优化。本文中，我们通过多样的特征表示（在本例中是领域特定的）学习音频嵌入。对于涉及数百种声音分类的情况，我们学习分别针对音高、音色和神经表示等多样的音频属性建立稳健的嵌入，同时也通过端到端架构进行学习。我们观察到手工制作的嵌入，例如基于音高和音色的嵌入，虽然单独使用时无法击败完全端到端的表示，但将这些嵌入与端到端嵌入相结合可以显著提高性能。这项工作将为在端到端模型中引入一些领域专业知识来学习稳健、多样化的表示铺平道路，并超越仅训练端到端模型的性能。

    With the advent of modern AI architectures, a shift has happened towards end-to-end architectures. This pivot has led to neural architectures being trained without domain-specific biases/knowledge, optimized according to the task. We in this paper, learn audio embeddings via diverse feature representations, in this case, domain-specific. For the case of audio classification over hundreds of categories of sound, we learn robust separate embeddings for diverse audio properties such as pitch, timbre, and neural representation, along with also learning it via an end-to-end architecture. We observe handcrafted embeddings, e.g., pitch and timbre-based, although on their own, are not able to beat a fully end-to-end representation, yet adding these together with end-to-end embedding helps us, significantly improve performance. This work would pave the way to bring some domain expertise with end-to-end models to learn robust, diverse representations, surpassing the performance of just training 
    
[^2]: JEN-1：具有全向扩散模型的文本引导通用音乐生成

    JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models. (arXiv:2308.04729v1 [cs.SD])

    [http://arxiv.org/abs/2308.04729](http://arxiv.org/abs/2308.04729)

    JEN-1是一个高保真度通用音乐生成模型，通过结合自回归和非自回归训练，实现了文本引导的音乐生成、音乐修补和延续等生成任务，在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。

    

    随着深度生成模型的进步，音乐生成引起了越来越多的关注。然而，基于文本描述生成音乐（即文本到音乐）仍然具有挑战性，原因是音乐结构的复杂性和高采样率的要求。尽管任务的重要性，当前的生成模型在音乐质量、计算效率和泛化能力方面存在局限性。本文介绍了JEN-1，这是一个用于文本到音乐生成的通用高保真模型。JEN-1是一个结合了自回归和非自回归训练的扩散模型。通过上下文学习，JEN-1可以执行各种生成任务，包括文本引导的音乐生成、音乐修补以及延续。评估结果表明，JEN-1在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。我们的演示可在此网址获取：http://URL

    Music generation has attracted growing interest with the advancement of deep generative models. However, generating music conditioned on textual descriptions, known as text-to-music, remains challenging due to the complexity of musical structures and high sampling rate requirements. Despite the task's significance, prevailing generative models exhibit limitations in music quality, computational efficiency, and generalization. This paper introduces JEN-1, a universal high-fidelity model for text-to-music generation. JEN-1 is a diffusion model incorporating both autoregressive and non-autoregressive training. Through in-context learning, JEN-1 performs various generation tasks including text-guided music generation, music inpainting, and continuation. Evaluations demonstrate JEN-1's superior performance over state-of-the-art methods in text-music alignment and music quality while maintaining computational efficiency. Our demos are available at this http URL
    

