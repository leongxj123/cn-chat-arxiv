# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Noisy Beat is Worth 16 Words: a Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers](https://arxiv.org/abs/2402.10748) | 提出了一种微型Transformer模型，用于在低功耗微控制器上对心电图信号进行分析，仅需6k个参数，准确率达到98.97%，适用于识别MIT-BIH心律失常数据库中的5个最常见心律失常类别 |
| [^2] | [GestureGPT: Zero-shot Interactive Gesture Understanding and Grounding with Large Language Model Agents.](http://arxiv.org/abs/2310.12821) | GestureGPT是一个零样本交互手势理解和对接框架，利用大语言模型代理解读手势描述并根据交互环境提供上下文信息，能够将用户意图对接到交互功能上。 |

# 详细

[^1]: 价值16个字的噪声节拍: 一种用于微控制器低功率心律失常分类的微型Transformer

    A Noisy Beat is Worth 16 Words: a Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers

    [https://arxiv.org/abs/2402.10748](https://arxiv.org/abs/2402.10748)

    提出了一种微型Transformer模型，用于在低功耗微控制器上对心电图信号进行分析，仅需6k个参数，准确率达到98.97%，适用于识别MIT-BIH心律失常数据库中的5个最常见心律失常类别

    

    长期监测心血管疾病的可穿戴系统正在成为诊断和治疗中广泛应用且有价值的资产。一种用于实时分析心电图（ECG）信号，以及检测心脏状况（如心律失常）的有前途的方法是Transformer机器学习模型。该模型是用于时间序列分类的强大模型，但在可穿戴领域的高效实现却面临着重大的设计挑战，需要在兼顾足够精度和适当复杂度的情况下进行。在这项工作中，我们提出了一种用于分析ECG信号的微型Transformer模型，仅需要6k个参数，在MIT-BIH心律失常数据库中识别5个最常见心律失常类别时达到了98.97%的准确率，考虑到对低功耗微控制器设备进行高效执行所需的8位整数推理。我们探索了一种

    arXiv:2402.10748v1 Announce Type: cross  Abstract: Wearable systems for the long-term monitoring of cardiovascular diseases are becoming widespread and valuable assets in diagnosis and therapy. A promising approach for real-time analysis of the electrocardiographic (ECG) signal and the detection of heart conditions, such as arrhythmia, is represented by the transformer machine learning model. Transformers are powerful models for the classification of time series, although efficient implementation in the wearable domain raises significant design challenges, to combine adequate accuracy and a suitable complexity. In this work, we present a tiny transformer model for the analysis of the ECG signal, requiring only 6k parameters and reaching 98.97% accuracy in the recognition of the 5 most common arrhythmia classes from the MIT-BIH Arrhythmia database, assessed considering 8-bit integer inference as required for efficient execution on low-power microcontroller-based devices. We explored an 
    
[^2]: GestureGPT: 零样本交互手势理解与基于大语言模型代理的对接

    GestureGPT: Zero-shot Interactive Gesture Understanding and Grounding with Large Language Model Agents. (arXiv:2310.12821v1 [cs.CL])

    [http://arxiv.org/abs/2310.12821](http://arxiv.org/abs/2310.12821)

    GestureGPT是一个零样本交互手势理解和对接框架，利用大语言模型代理解读手势描述并根据交互环境提供上下文信息，能够将用户意图对接到交互功能上。

    

    当前的手势识别系统主要关注识别预定义集合中的手势，未能将这些手势与交互式图形用户界面元素或系统功能相连接（例如，将“竖起大拇指”手势与“喜欢”按钮关联起来）。我们引入了GestureGPT，这是一个新颖的零样本手势理解和对接框架，利用大语言模型（LLM）。手势描述根据手势视频中的手部关键点坐标进行形式化，并输入到我们的双代理对话系统中。一个手势代理解读这些描述，并询问有关交互环境的信息（例如，界面、历史记录、凝视数据），一个上下文代理负责组织并提供这些信息。经过迭代的交流，手势代理能够理解用户意图，并将其对接到一个交互功能上。我们使用公开的第一视角和第三视角手势数据集验证了手势描述模块，并在视频流和智能家居物联网控制的两个真实场景中测试了整个系统。

    Current gesture recognition systems primarily focus on identifying gestures within a predefined set, leaving a gap in connecting these gestures to interactive GUI elements or system functions (e.g., linking a 'thumb-up' gesture to a 'like' button). We introduce GestureGPT, a novel zero-shot gesture understanding and grounding framework leveraging large language models (LLMs). Gesture descriptions are formulated based on hand landmark coordinates from gesture videos and fed into our dual-agent dialogue system. A gesture agent deciphers these descriptions and queries about the interaction context (e.g., interface, history, gaze data), which a context agent organizes and provides. Following iterative exchanges, the gesture agent discerns user intent, grounding it to an interactive function. We validated the gesture description module using public first-view and third-view gesture datasets and tested the whole system in two real-world settings: video streaming and smart home IoT control. T
    

