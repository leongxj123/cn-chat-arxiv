# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ImgTrojan: Jailbreaking Vision-Language Models with ONE Image](https://arxiv.org/abs/2403.02910) | 本文提出了一种针对视觉-语言模型的新型越狱攻击，通过在训练数据中插入恶意文本提示，成功实施越狱攻击，并分析了有毒数据比率和可训练参数位置对攻击成功率的影响。 |
| [^2] | [Multi-objective Differentiable Neural Architecture Search](https://arxiv.org/abs/2402.18213) | 提出了一种新颖的NAS算法，可以在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，生成精心选择的多设备架构。 |
| [^3] | [The cell signaling structure function.](http://arxiv.org/abs/2401.02501) | 该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。 |
| [^4] | [What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning.](http://arxiv.org/abs/2311.01487) | 通过综合复杂的视觉推理任务，可以有效改善多模式大型语言模型在评估基准上的性能。我们提出了一种自动创建高质量复杂视觉推理指令的系统方法。 |

# 详细

[^1]: ImgTrojan: 用一张图片对视觉-语言模型进行越狱

    ImgTrojan: Jailbreaking Vision-Language Models with ONE Image

    [https://arxiv.org/abs/2403.02910](https://arxiv.org/abs/2403.02910)

    本文提出了一种针对视觉-语言模型的新型越狱攻击，通过在训练数据中插入恶意文本提示，成功实施越狱攻击，并分析了有毒数据比率和可训练参数位置对攻击成功率的影响。

    

    近来，对于大型语言模型（LLMs）与人类价值观的对齐引起了越来越多的关注。然而，它们与视觉模块集成的安全问题，即视觉-语言模型（VLMs），仍然相对未被充分探讨。本文提出了一种针对VLMs的新型越狱攻击，旨在当用户输入有害指令时绕过其安全阻碍。假设我们的有毒（图像，文本）数据对包含在训练数据中。通过用恶意越狱提示替换原始文本标题，我们的方法可以利用有毒图像执行越狱攻击。此外，我们分析了有毒比率和可训练参数位置对攻击成功率的影响。为了评估，我们设计了两个度量标准来量化我们攻击的成功率和隐蔽性。结合一系列策划的有害指令，可以衡量攻击的有效性。

    arXiv:2403.02910v1 Announce Type: cross  Abstract: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficac
    
[^2]: 多目标可微神经架构搜索

    Multi-objective Differentiable Neural Architecture Search

    [https://arxiv.org/abs/2402.18213](https://arxiv.org/abs/2402.18213)

    提出了一种新颖的NAS算法，可以在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，生成精心选择的多设备架构。

    

    多目标优化（MOO）中的Pareto前沿轮廓剖析是具有挑战性的，尤其是在像神经网络训练这样的昂贵目标中。 相对于传统的NAS方法，我们提出了一种新颖的NAS算法，该算法在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，并生成精心选择的多设备架构。为此，我们通过一个超网络参数化跨多个设备和多个目标的联合架构分布，超网络可以根据硬件特征和偏好向量进行条件化，实现零次搜索。

    arXiv:2402.18213v1 Announce Type: new  Abstract: Pareto front profiling in multi-objective optimization (MOO), i.e. finding a diverse set of Pareto optimal solutions, is challenging, especially with expensive objectives like neural network training. Typically, in MOO neural architecture search (NAS), we aim to balance performance and hardware metrics across devices. Prior NAS approaches simplify this task by incorporating hardware constraints into the objective function, but profiling the Pareto front necessitates a search for each constraint. In this work, we propose a novel NAS algorithm that encodes user preferences for the trade-off between performance and hardware metrics, and yields representative and diverse architectures across multiple devices in just one search run. To this end, we parameterize the joint architectural distribution across devices and multiple objectives via a hypernetwork that can be conditioned on hardware features and preference vectors, enabling zero-shot t
    
[^3]: 细胞信号传导结构和功能

    The cell signaling structure function. (arXiv:2401.02501v1 [cs.CV])

    [http://arxiv.org/abs/2401.02501](http://arxiv.org/abs/2401.02501)

    该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。

    

    活体细胞显微镜捕捉到的五维$(x,y,z,channel,time)$视频显示了细胞运动和信号动力学的模式。我们在这里提出一种在五维活体细胞显微镜视频中寻找细胞信号动力学时空模式的方法，该方法独特之处在于不需要预先了解预期的模式动力学以及没有训练数据。所提出的细胞信号结构函数（SSF）是一种Kolmogorov结构函数，可以通过核心区域相对于周围细胞质的核糖体强度来最优地测量细胞信号状态，相比当前最先进的核糖体与细胞核比值有了显著的改进。通过度量归一化压缩距离（NCD）来识别相似的模式。NCD是一个用于表示输入的SSF构图在低维嵌入中作为点的Hilbert空间的再生核，可以最优地捕捉模式。

    Live cell microscopy captures 5-D $(x,y,z,channel,time)$ movies that display patterns of cellular motion and signaling dynamics. We present here an approach to finding spatiotemporal patterns of cell signaling dynamics in 5-D live cell microscopy movies unique in requiring no \emph{a priori} knowledge of expected pattern dynamics, and no training data. The proposed cell signaling structure function (SSF) is a Kolmogorov structure function that optimally measures cell signaling state as nuclear intensity w.r.t. surrounding cytoplasm, a significant improvement compared to the current state-of-the-art cytonuclear ratio. SSF kymographs store at each spatiotemporal cell centroid the SSF value, or a functional output such as velocity. Patterns of similarity are identified via the metric normalized compression distance (NCD). The NCD is a reproducing kernel for a Hilbert space that represents the input SSF kymographs as points in a low dimensional embedding that optimally captures the pattern
    
[^4]: 优秀的视觉指导有什么特点？综合复杂的视觉推理指令用于视觉指导调整

    What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning. (arXiv:2311.01487v1 [cs.CV])

    [http://arxiv.org/abs/2311.01487](http://arxiv.org/abs/2311.01487)

    通过综合复杂的视觉推理任务，可以有效改善多模式大型语言模型在评估基准上的性能。我们提出了一种自动创建高质量复杂视觉推理指令的系统方法。

    

    视觉指导调整是提高多模式大型语言模型（MLLMs）的零样本泛化能力的重要方法。最近提出了许多着眼于不同焦点和特征的视觉指导数据集，使得MLLMs在评估基准上取得了令人惊讶的结果。为了开发更强大的MLLMs，本文旨在研究一个更基本的问题：“什么样的视觉指导才是好的？”通过进行全面的实证研究，我们发现侧重于复杂视觉推理任务的指导对于改善MLLMs在评估基准上的性能特别有效。基于这一发现，我们设计了一个系统的方法来自动创建高质量的复杂视觉推理指令。我们的方法采用合成-复杂化-重构的范式，利用多个阶段逐渐增加指令的复杂性，同时保证质量。

    Visual instruction tuning is an essential approach to improving the zero-shot generalization capability of Multi-modal Large Language Models (MLLMs). A surge of visual instruction datasets with various focuses and characteristics have been proposed recently, enabling MLLMs to achieve surprising results on evaluation benchmarks. To develop more capable MLLMs, in this paper, we aim to investigate a more fundamental question: ``what makes for good visual instructions?''. By conducting a comprehensive empirical study, we find that instructions focused on complex visual reasoning tasks are particularly effective in improving the performance of MLLMs on evaluation benchmarks. Building upon this finding, we design a systematic approach to automatically creating high-quality complex visual reasoning instructions. Our approach employs a synthesis-complication-reformulation paradigm, leveraging multiple stages to gradually increase the complexity of the instructions while guaranteeing quality. B
    

