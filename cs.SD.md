# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AV2Wav: Diffusion-Based Re-synthesis from Continuous Self-supervised Features for Audio-Visual Speech Enhancement.](http://arxiv.org/abs/2309.08030) | 本论文提出了一种名为AV2Wav的音频-视觉语音增强方法，利用连续自监督特征和扩散模型生成干净的语音，克服了现实训练数据的挑战。与基于掩蔽的基线方法相比，该方法在声码任务上表现更好，并通过多任务训练进一步优化性能。 |

# 详细

[^1]: AV2Wav：基于连续自监督特征的扩散重合成技术用于音频-视觉语音增强

    AV2Wav: Diffusion-Based Re-synthesis from Continuous Self-supervised Features for Audio-Visual Speech Enhancement. (arXiv:2309.08030v1 [eess.AS])

    [http://arxiv.org/abs/2309.08030](http://arxiv.org/abs/2309.08030)

    本论文提出了一种名为AV2Wav的音频-视觉语音增强方法，利用连续自监督特征和扩散模型生成干净的语音，克服了现实训练数据的挑战。与基于掩蔽的基线方法相比，该方法在声码任务上表现更好，并通过多任务训练进一步优化性能。

    

    语音增强系统通常使用干净和噪声语音对进行训练。在音频-视觉语音增强中，干净的数据不够多；大多数音频-视觉数据集都是在现实环境中收集的，包含背景噪声和混响，这阻碍了音频-视觉语音增强的发展。在本研究中，我们引入了AV2Wav，一种基于重合成的音频-视觉语音增强方法，可以在现实训练数据的挑战下生成干净的语音。我们使用神经质量估计器从音频-视觉语料库中获取几乎干净的语音子集，并在此子集上训练一个扩散模型，该模型可以根据来自AV-HuBERT的连续语音表示生成声波形，具有噪声鲁棒训练。我们使用连续而不是离散表示来保留韵律和说话者信息。仅仅通过声码任务，该模型就比基于掩蔽的基线更好地执行语音增强。我们进一步fine-tune模型，以转化为在多任务下进行训练，通过联合多帧声学到语音转化来提高性能。

    Speech enhancement systems are typically trained using pairs of clean and noisy speech. In audio-visual speech enhancement (AVSE), there is not as much ground-truth clean data available; most audio-visual datasets are collected in real-world environments with background noise and reverberation, hampering the development of AVSE. In this work, we introduce AV2Wav, a resynthesis-based audio-visual speech enhancement approach that can generate clean speech despite the challenges of real-world training data. We obtain a subset of nearly clean speech from an audio-visual corpus using a neural quality estimator, and then train a diffusion model on this subset to generate waveforms conditioned on continuous speech representations from AV-HuBERT with noise-robust training. We use continuous rather than discrete representations to retain prosody and speaker information. With this vocoding task alone, the model can perform speech enhancement better than a masking-based baseline. We further fine-
    

