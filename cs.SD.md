# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Driven Room Acoustic Modeling Via Differentiable Feedback Delay Networks With Learnable Delay Lines](https://arxiv.org/abs/2404.00082) | 通过可学习延迟线实现可微分反馈延迟网络的参数优化，实现了对室内声学特性的数据驱动建模。 |
| [^2] | [Speech Robust Bench: A Robustness Benchmark For Speech Recognition](https://arxiv.org/abs/2403.07937) | 提出了一个全面基准（SRB），用于评估自动语音识别（ASR）模型对各种破坏的鲁棒性，发现模型大小和某些建模选择有助于提高鲁棒性，并观察到在不同人口亚组上模型的鲁棒性存在明显差异。 |
| [^3] | [Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion](https://arxiv.org/abs/2402.14285) | 介绍了一种用于符号音乐生成的不可微分规则引导的新方法，引入了可以与之即插即用的高时间分辨率潜在扩散架构，对音乐质量取得了显著进步 |
| [^4] | [The FruitShell French synthesis system at the Blizzard 2023 Challenge.](http://arxiv.org/abs/2309.00223) | 本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。 |

# 详细

[^1]: 基于可微分反馈延迟网络和可学习延迟线的数据驱动室内声学建模

    Data-Driven Room Acoustic Modeling Via Differentiable Feedback Delay Networks With Learnable Delay Lines

    [https://arxiv.org/abs/2404.00082](https://arxiv.org/abs/2404.00082)

    通过可学习延迟线实现可微分反馈延迟网络的参数优化，实现了对室内声学特性的数据驱动建模。

    

    在过去的几十年中，人们致力于设计人工混响算法，旨在模拟物理环境的室内声学。尽管取得了显著进展，但延迟网络模型的自动参数调整仍然是一个开放性挑战。我们提出了一种新方法，通过学习可微分反馈延迟网络（FDN）的参数，使其输出呈现出所测得的室内脉冲响应的感知特性。

    arXiv:2404.00082v1 Announce Type: cross  Abstract: Over the past few decades, extensive research has been devoted to the design of artificial reverberation algorithms aimed at emulating the room acoustics of physical environments. Despite significant advancements, automatic parameter tuning of delay-network models remains an open challenge. We introduce a novel method for finding the parameters of a Feedback Delay Network (FDN) such that its output renders the perceptual qualities of a measured room impulse response. The proposed approach involves the implementation of a differentiable FDN with trainable delay lines, which, for the first time, allows us to simultaneously learn each and every delay-network parameter via backpropagation. The iterative optimization process seeks to minimize a time-domain loss function incorporating differentiable terms accounting for energy decay and echo density. Through experimental validation, we show that the proposed method yields time-invariant freq
    
[^2]: 语音鲁棒基准：用于语音识别的鲁棒性基准

    Speech Robust Bench: A Robustness Benchmark For Speech Recognition

    [https://arxiv.org/abs/2403.07937](https://arxiv.org/abs/2403.07937)

    提出了一个全面基准（SRB），用于评估自动语音识别（ASR）模型对各种破坏的鲁棒性，发现模型大小和某些建模选择有助于提高鲁棒性，并观察到在不同人口亚组上模型的鲁棒性存在明显差异。

    

    随着自动语音识别（ASR）模型变得越来越普遍，确保它们在物理世界和数字世界中的各种破坏下进行可靠预测变得愈发重要。我们提出了语音鲁棒基准（SRB），这是一个用于评估ASR模型对各种破坏的鲁棒性的全面基准。SRB由69个输入扰动组成，旨在模拟ASR模型可能在物理世界和数字世界中遇到的各种破坏。我们使用SRB来评估几种最先进的ASR模型的鲁棒性，并观察到模型大小和某些建模选择（如离散表示和自我训练）似乎有助于提高鲁棒性。我们将此分析扩展到衡量ASR模型在来自各种人口亚组的数据上的鲁棒性，即英语和西班牙语使用者以及男性和女性，并观察到模型的鲁棒性在不同亚组之间存在明显差异。

    arXiv:2403.07937v1 Announce Type: cross  Abstract: As Automatic Speech Recognition (ASR) models become ever more pervasive, it is important to ensure that they make reliable predictions under corruptions present in the physical and digital world. We propose Speech Robust Bench (SRB), a comprehensive benchmark for evaluating the robustness of ASR models to diverse corruptions. SRB is composed of 69 input perturbations which are intended to simulate various corruptions that ASR models may encounter in the physical and digital world. We use SRB to evaluate the robustness of several state-of-the-art ASR models and observe that model size and certain modeling choices such as discrete representations, and self-training appear to be conducive to robustness. We extend this analysis to measure the robustness of ASR models on data from various demographic subgroups, namely English and Spanish speakers, and males and females, and observed noticeable disparities in the model's robustness across su
    
[^3]: 具有不可微分规则引导扩散的符号音乐生成

    Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion

    [https://arxiv.org/abs/2402.14285](https://arxiv.org/abs/2402.14285)

    介绍了一种用于符号音乐生成的不可微分规则引导的新方法，引入了可以与之即插即用的高时间分辨率潜在扩散架构，对音乐质量取得了显著进步

    

    我们研究了符号音乐生成的问题（例如生成钢琴卷谱），技术重点放在不可微分规则引导上。音乐规则通常以符号形式表达在音符特征上，如音符密度或和弦进行，许多规则是不可微分的，这在使用它们进行引导扩散时存在挑战。我们提出了一种新颖的引导方法，称为随机控制引导（SCG），它仅需要对规则函数进行前向评估，可以与预训练的扩散模型以即插即用的方式一起工作，从而首次实现了对不可微分规则的无训练引导。此外，我们引入了一种用于符号音乐生成的高时间分辨率潜在扩散架构，可以与SCG以即插即用的方式组合。与符号音乐生成中的标准强基线相比，该框架在音乐质量方面展示了明显的进展

    arXiv:2402.14285v1 Announce Type: cross  Abstract: We study the problem of symbolic music generation (e.g., generating piano rolls), with a technical focus on non-differentiable rule guidance. Musical rules are often expressed in symbolic form on note characteristics, such as note density or chord progression, many of which are non-differentiable which pose a challenge when using them for guided diffusion. We propose Stochastic Control Guidance (SCG), a novel guidance method that only requires forward evaluation of rule functions that can work with pre-trained diffusion models in a plug-and-play way, thus achieving training-free guidance for non-differentiable rules for the first time. Additionally, we introduce a latent diffusion architecture for symbolic music generation with high time resolution, which can be composed with SCG in a plug-and-play fashion. Compared to standard strong baselines in symbolic music generation, this framework demonstrates marked advancements in music quali
    
[^4]: FruitShell法语合成系统在Blizzard 2023挑战赛中的应用

    The FruitShell French synthesis system at the Blizzard 2023 Challenge. (arXiv:2309.00223v1 [eess.AS])

    [http://arxiv.org/abs/2309.00223](http://arxiv.org/abs/2309.00223)

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。

    

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统。该挑战包括两个任务：从女性演讲者生成高质量的语音和生成与特定个体相似的语音。关于比赛数据，我们进行了筛选过程，去除了缺失或错误的文本数据。我们对除音素以外的所有符号进行了整理，并消除了没有发音或持续时间为零的符号。此外，我们还在文本中添加了词边界和起始/结束符号，根据我们之前的经验，我们发现这样可以提高语音质量。对于Spoke任务，我们根据比赛规则进行了数据增强。我们使用了一个开源的G2P模型将法语文本转录为音素。由于G2P模型使用国际音标（IPA），我们对提供的比赛数据应用了相同的转录过程，以进行标准化。然而，由于编译器对某些技术限制的识别能力有限，所以我们为了保持竞争的公正，将数据按音标划分为不同的片段进行评估。

    This paper presents a French text-to-speech synthesis system for the Blizzard Challenge 2023. The challenge consists of two tasks: generating high-quality speech from female speakers and generating speech that closely resembles specific individuals. Regarding the competition data, we conducted a screening process to remove missing or erroneous text data. We organized all symbols except for phonemes and eliminated symbols that had no pronunciation or zero duration. Additionally, we added word boundary and start/end symbols to the text, which we have found to improve speech quality based on our previous experience. For the Spoke task, we performed data augmentation according to the competition rules. We used an open-source G2P model to transcribe the French texts into phonemes. As the G2P model uses the International Phonetic Alphabet (IPA), we applied the same transcription process to the provided competition data for standardization. However, due to compiler limitations in recognizing 
    

