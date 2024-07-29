# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension](https://arxiv.org/abs/2402.07729) | 这个论文介绍了AIR-Bench，这是第一个用于评估大型音频语言模型理解和生成音频信号的基准。 |
| [^2] | [Anticipatory Music Transformer.](http://arxiv.org/abs/2306.08620) | 该论文提出一种预测音乐转换器，它能够实现在符号音乐生成的过程中进行控制，包括补全控制任务和伴奏，并且在大型且多样的数据集上表现出色。 |

# 详细

[^1]: AIR-Bench: 通过生成性理解评估大型音频语言模型的基准

    AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension

    [https://arxiv.org/abs/2402.07729](https://arxiv.org/abs/2402.07729)

    这个论文介绍了AIR-Bench，这是第一个用于评估大型音频语言模型理解和生成音频信号的基准。

    

    最近，指导性的音频语言模型因其对人与音频的互动能力而受到广泛关注。然而，缺乏能够评估以音频为中心的互动能力的基准已经阻碍了该领域的进展。以往的模型主要关注评估不同的基本任务，如自动语音识别（ASR），缺乏对围绕音频的开放式生成能力的评估。因此，追踪大型音频语言模型（LALMs）领域的进展并为未来的改进提供指导是具有挑战性的。在本文中，我们介绍了AIR-Bench（音频指导基准），这是第一个旨在评估LALMs理解各种类型音频信号（包括人类语音、自然声音和音乐）以及与人以文本形式进行交互能力的基准。AIR-Bench包含两个维度：基础和生成理解。

    Recently, instruction-following audio-language models have received broad attention for human-audio interaction. However, the absence of benchmarks capable of evaluating audio-centric interaction capabilities has impeded advancements in this field. Previous models primarily focus on assessing different fundamental tasks, such as Automatic Speech Recognition (ASR), and lack an assessment of the open-ended generative capabilities centered around audio. Thus, it is challenging to track the progression in the Large Audio-Language Models (LALMs) domain and to provide guidance for future improvement. In this paper, we introduce AIR-Bench (\textbf{A}udio \textbf{I}nst\textbf{R}uction \textbf{Bench}mark), the first benchmark designed to evaluate the ability of LALMs to understand various types of audio signals (including human speech, natural sounds, and music), and furthermore, to interact with humans in the textual format. AIR-Bench encompasses two dimensions: \textit{foundation} and \textit
    
[^2]: 预测音乐转换器

    Anticipatory Music Transformer. (arXiv:2306.08620v1 [cs.SD])

    [http://arxiv.org/abs/2306.08620](http://arxiv.org/abs/2306.08620)

    该论文提出一种预测音乐转换器，它能够实现在符号音乐生成的过程中进行控制，包括补全控制任务和伴奏，并且在大型且多样的数据集上表现出色。

    

    我们引入了anticipation（预测）：一种构建生成模型的方法，该模型基于事件过程（时间点过程）的实现，以异步地控制与第二个相关过程（控制过程）的相关性。我们通过交错事件和控件序列来实现这一目标，使控件出现在事件序列的停止时间之后。这项工作的动机来自符号音乐生成控制中出现的问题。我们专注于infiling（补全）控制任务，其中控制事件是事件本身的子集，并且条件生成完成给定固定控制事件的事件序列。我们使用大型多样的Lakh MIDI音乐数据集训练预测infiling模型。这些模型与提示音乐生成的自回归模型性能相当，并具有执行infilling控制任务的附加能力，包括伴奏。人工评估员报告说，预测模型产生的伴奏具有高可辨性和优美性。

    We introduce anticipation: a method for constructing a controllable generative model of a temporal point process (the event process) conditioned asynchronously on realizations of a second, correlated process (the control process). We achieve this by interleaving sequences of events and controls, such that controls appear following stopping times in the event sequence. This work is motivated by problems arising in the control of symbolic music generation. We focus on infilling control tasks, whereby the controls are a subset of the events themselves, and conditional generation completes a sequence of events given the fixed control events. We train anticipatory infilling models using the large and diverse Lakh MIDI music dataset. These models match the performance of autoregressive models for prompted music generation, with the additional capability to perform infilling control tasks, including accompaniment. Human evaluators report that an anticipatory model produces accompaniments with
    

