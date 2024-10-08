# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction](https://arxiv.org/abs/2403.05010) | RFWave是一种新颖的多频带整流流动方法，可以从Mel频谱图中重建高保真度音频波形，仅需10个采样步骤即可实现出色的重建质量和优越的计算效率。 |
| [^2] | [Arrange, Inpaint, and Refine: Steerable Long-term Music Audio Generation and Editing via Content-based Controls](https://arxiv.org/abs/2402.09508) | 通过引入参数高效微调（PEFT）方法，本研究实现了自回归语言模型在音乐修复和音乐排列任务中的应用。在多个音乐编辑任务中，该方法展示了有希望的结果，并为未来的AI驱动音乐编辑工具提供了更灵活的控制。 |
| [^3] | [Content-based Controls For Music Large Language Modeling.](http://arxiv.org/abs/2310.17162) | 该论文提出了一种基于内容的控制方法，用于音乐大语言建模。通过对音高、和弦和鼓乐等固有音乐语言的直接控制，实现了高质量的音乐生成，并且使用了参数高效微调的方法，比原始模型的参数数量少于4%。 |
| [^4] | [Advancing Test-Time Adaptation for Acoustic Foundation Models in Open-World Shifts.](http://arxiv.org/abs/2310.09505) | 本文提出了一种针对声学基础模型的测试时间自适应方法，以解决开放世界数据转换中的分布变化问题。研究发现，噪声较大的语音帧包含重要的语义内容。 |
| [^5] | [Prosody Analysis of Audiobooks.](http://arxiv.org/abs/2310.06930) | 本研究通过使用一个含有93个书籍和对应有声书的数据集，提出了改进的模型来预测有声书文本中的韵律属性。结果显示，我们的预测韵律与人类朗读比商业级TTS系统更相关，并且人们更喜欢韵律增强的有声书朗读。 |

# 详细

[^1]: RFWave：用于音频波形重建的多频带整流流动

    RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction

    [https://arxiv.org/abs/2403.05010](https://arxiv.org/abs/2403.05010)

    RFWave是一种新颖的多频带整流流动方法，可以从Mel频谱图中重建高保真度音频波形，仅需10个采样步骤即可实现出色的重建质量和优越的计算效率。

    

    最近生成建模的进展在从不同表示中重建音频波形方面取得了显著进展。虽然扩散模型已被用于重建音频波形，但由于它们在个别样本点级别进行操作并且需要相对较大数量的采样步骤，因此它们往往会出现延迟问题。在本研究中，我们介绍了RFWave，一种新颖的多频带整流流动方法，它从Mel频谱图中重建高保真度音频波形。RFWave在生成复杂频谱图并在帧级别运行方面具有独特性，同时处理所有子带以增强效率。由于希望获得平缓传输轨迹的整流流动，RFWave仅需10个采样步骤。实证评估表明，RFWave实现了卓越的重建质量和优越的计算效率，能够以更快的速度生成音频。

    arXiv:2403.05010v1 Announce Type: cross  Abstract: Recent advancements in generative modeling have led to significant progress in audio waveform reconstruction from diverse representations. Although diffusion models have been used for reconstructing audio waveforms, they tend to exhibit latency issues because they operate at the level of individual sample points and require a relatively large number of sampling steps. In this study, we introduce RFWave, a novel multi-band Rectified Flow approach that reconstructs high-fidelity audio waveforms from Mel-spectrograms. RFWave is distinctive for generating complex spectrograms and operating at the frame level, processing all subbands concurrently to enhance efficiency. Thanks to Rectified Flow, which aims for a flat transport trajectory, RFWave requires only 10 sampling steps. Empirical evaluations demonstrate that RFWave achieves exceptional reconstruction quality and superior computational efficiency, capable of generating audio at a spee
    
[^2]: 排列、修复和改进：通过基于内容的控制实现可操控的长期音乐音频生成和编辑

    Arrange, Inpaint, and Refine: Steerable Long-term Music Audio Generation and Editing via Content-based Controls

    [https://arxiv.org/abs/2402.09508](https://arxiv.org/abs/2402.09508)

    通过引入参数高效微调（PEFT）方法，本研究实现了自回归语言模型在音乐修复和音乐排列任务中的应用。在多个音乐编辑任务中，该方法展示了有希望的结果，并为未来的AI驱动音乐编辑工具提供了更灵活的控制。

    

    可控音乐生成在人机音乐共创中起着重要作用。虽然大型语言模型（LLM）在生成高质量音乐方面表现出了潜力，但它们对自回归生成的依赖限制了它们在音乐编辑任务中的实用性。为了弥合这一差距，我们引入了一种新颖的参数高效微调（PEFT）方法。该方法使自回归语言模型能够无缝地解决音乐修复任务。此外，我们的PEFT方法集成了基于帧级内容的控制，促进了轨道条件音乐的精炼和分数条件音乐的排列。我们将该方法应用于MusicGen，一个领先的自回归音乐生成模型的微调。我们的实验在多个音乐编辑任务中展示了有希望的结果，为未来的AI驱动音乐编辑工具提供了更灵活的控制。

    arXiv:2402.09508v1 Announce Type: cross  Abstract: Controllable music generation plays a vital role in human-AI music co-creation. While Large Language Models (LLMs) have shown promise in generating high-quality music, their focus on autoregressive generation limits their utility in music editing tasks. To bridge this gap, we introduce a novel Parameter-Efficient Fine-Tuning (PEFT) method. This approach enables autoregressive language models to seamlessly address music inpainting tasks. Additionally, our PEFT method integrates frame-level content-based controls, facilitating track-conditioned music refinement and score-conditioned music arrangement. We apply this method to fine-tune MusicGen, a leading autoregressive music generation model. Our experiments demonstrate promising results across multiple music editing tasks, offering more flexible controls for future AI-driven music editing tools. A demo page\footnote{\url{https://kikyo-16.github.io/AIR/}.} showcasing our work and source 
    
[^3]: 基于内容的音乐大语言建模的控制

    Content-based Controls For Music Large Language Modeling. (arXiv:2310.17162v1 [cs.AI])

    [http://arxiv.org/abs/2310.17162](http://arxiv.org/abs/2310.17162)

    该论文提出了一种基于内容的控制方法，用于音乐大语言建模。通过对音高、和弦和鼓乐等固有音乐语言的直接控制，实现了高质量的音乐生成，并且使用了参数高效微调的方法，比原始模型的参数数量少于4%。

    

    近年来，在音乐音频领域出现了大规模语言模型的迅速增长。这些模型使得能够进行高质量音乐的端到端生成，并且一些模型可以使用文本描述进行条件生成。然而，文本在音乐上的控制能力本质上是有限的，因为它们只能通过元数据（如歌手和乐器）或高级表示（如流派和情感）间接地描述音乐。我们的目标是进一步提供对音高、和弦和鼓乐等固有音乐语言的直接和基于内容的控制能力。为此，我们提出了Coco-Mulla，这是一种用于音乐大语言建模的基于内容的控制方法。它使用了针对基于Transformer的音频模型量身定制的参数高效微调（PEFT）方法。实验表明，我们的方法在低资源半监督学习中实现了高质量的音乐生成，相比原始模型，参数调优的比例不到4%。

    Recent years have witnessed a rapid growth of large-scale language models in the domain of music audio. Such models enable end-to-end generation of higher-quality music, and some allow conditioned generation using text descriptions. However, the control power of text controls on music is intrinsically limited, as they can only describe music indirectly through meta-data (such as singers and instruments) or high-level representations (such as genre and emotion). We aim to further equip the models with direct and content-based controls on innate music languages such as pitch, chords and drum track. To this end, we contribute Coco-Mulla, a content-based control method for music large language modeling. It uses a parameter-efficient fine-tuning (PEFT) method tailored for Transformer-based audio models. Experiments show that our approach achieved high-quality music generation with low-resource semi-supervised learning, tuning with less than 4% parameters compared to the original model and t
    
[^4]: 在开放世界转换中推进声学基础模型的测试时间自适应

    Advancing Test-Time Adaptation for Acoustic Foundation Models in Open-World Shifts. (arXiv:2310.09505v1 [cs.SD])

    [http://arxiv.org/abs/2310.09505](http://arxiv.org/abs/2310.09505)

    本文提出了一种针对声学基础模型的测试时间自适应方法，以解决开放世界数据转换中的分布变化问题。研究发现，噪声较大的语音帧包含重要的语义内容。

    

    测试时间自适应（TTA）是在推理过程中解决分布转换问题的关键方法，特别是在视觉识别任务中。然而，虽然声学模型在测试时间的语音分布转换中面临相似的挑战，但针对声学建模在开放世界数据转换环境下的TTA技术仍然很少见。考虑到声学基础模型的特点：1）它们主要是基于具有层归一化的变压器架构构建的；2）它们以一种非静态的方式处理长度不同的测试时间语音数据。这些因素使得在视觉聚焦的TTA方法的直接应用变得不可行，这些方法大多依赖于批归一化并假设独立样本。在本文中，我们深入研究了面临开放世界数据转换的预训练声学模型的TTA方法。我们发现，噪声较大、熵较高的语音帧通常带有关键的语义内容。传统的视觉TTA方法的直接应用在声学建模中并不可行。

    Test-Time Adaptation (TTA) is a critical paradigm for tackling distribution shifts during inference, especially in visual recognition tasks. However, while acoustic models face similar challenges due to distribution shifts in test-time speech, TTA techniques specifically designed for acoustic modeling in the context of open-world data shifts remain scarce. This gap is further exacerbated when considering the unique characteristics of acoustic foundation models: 1) they are primarily built on transformer architectures with layer normalization and 2) they deal with test-time speech data of varying lengths in a non-stationary manner. These aspects make the direct application of vision-focused TTA methods, which are mostly reliant on batch normalization and assume independent samples, infeasible. In this paper, we delve into TTA for pre-trained acoustic models facing open-world data shifts. We find that noisy, high-entropy speech frames, often non-silent, carry key semantic content. Tradit
    
[^5]: 《有声书的韵律分析》

    Prosody Analysis of Audiobooks. (arXiv:2310.06930v1 [cs.SD])

    [http://arxiv.org/abs/2310.06930](http://arxiv.org/abs/2310.06930)

    本研究通过使用一个含有93个书籍和对应有声书的数据集，提出了改进的模型来预测有声书文本中的韵律属性。结果显示，我们的预测韵律与人类朗读比商业级TTS系统更相关，并且人们更喜欢韵律增强的有声书朗读。

    

    最近在文本转语音方面取得了一些进展，使得从文本中生成自然音效的音频成为可能。然而，有声书朗读涉及到读者的戏剧性声音和语调，更多地依赖情感、对话和叙述。使用我们的数据集，包括93本书与其对应的有声书，我们提出了改进的模型，用于从叙述文本中预测韵律属性（音高、音量和语速），并使用语言建模。我们预测的韵律属性与人类朗读的相关性要远高于商业级TTS系统的结果：在24本书中，我们预测的音高对22本书的人类阅读更具相关性，而我们预测的音量属性对23本书的人类阅读更加相似。最后，我们进行了一项人类评估研究，以量化人们更喜欢韵律增强的有声书朗读还是商业级文本转语音系统。

    Recent advances in text-to-speech have made it possible to generate natural-sounding audio from text. However, audiobook narrations involve dramatic vocalizations and intonations by the reader, with greater reliance on emotions, dialogues, and descriptions in the narrative. Using our dataset of 93 aligned book-audiobook pairs, we present improved models for prosody prediction properties (pitch, volume, and rate of speech) from narrative text using language modeling. Our predicted prosody attributes correlate much better with human audiobook readings than results from a state-of-the-art commercial TTS system: our predicted pitch shows a higher correlation with human reading for 22 out of the 24 books, while our predicted volume attribute proves more similar to human reading for 23 out of the 24 books. Finally, we present a human evaluation study to quantify the extent that people prefer prosody-enhanced audiobook readings over commercial text-to-speech systems.
    

