# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions](https://arxiv.org/abs/2403.09346) | AVIBench是一个框架，用于分析大型视觉-语言模型对抗各种形式的对抗性视觉指令的鲁棒性，包括图像和文本攻击以及内容偏见攻击。 |
| [^2] | [Demystifying CLIP Data.](http://arxiv.org/abs/2309.16671) | CLIP的成功主要归功于其数据而非模型架构或预训练目标。我们通过元数据整理方法引入了MetaCLIP，该方法从原始数据池和元数据中生成一个平衡的子集，提供了更加详细的数据信息。在实验中，我们发现MetaCLIP在处理400M个图像-文本数据对时取得了良好的性能。 |
| [^3] | [Canonical Factors for Hybrid Neural Fields.](http://arxiv.org/abs/2308.15461) | 该研究对混合神经网络领域的规范因素进行了研究，发现因子特征容量虽然简单高效，但存在不良偏差。通过学习一组规范化变换，该方法成功消除偏差，并在图像、距离和辐射场重建任务中实现了质量、鲁棒性和运行时间的改进。 |
| [^4] | [Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey).](http://arxiv.org/abs/2307.10246) | 本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。 |

# 详细

[^1]: AVIBench：评估大型视觉-语言模型在对抗性视觉指导上的鲁棒性

    AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions

    [https://arxiv.org/abs/2403.09346](https://arxiv.org/abs/2403.09346)

    AVIBench是一个框架，用于分析大型视觉-语言模型对抗各种形式的对抗性视觉指令的鲁棒性，包括图像和文本攻击以及内容偏见攻击。

    

    大型视觉-语言模型（LVLMs）在对用户的视觉指令作出良好响应方面取得了显著进展。然而，这些指令涵盖图像和文本，容易受到有意和无意攻击的影响。尽管LVLMs对抗此类威胁的鲁棒性至关重要，但当前该领域的研究仍然有限。为弥补这一差距，我们引入了AVIBench，一个旨在分析LVLMs在面对各种对抗性视觉指令（AVIs）时的鲁棒性的框架，包括四种基于图像的AVIs、十种基于文本的AVIs和九种内容偏见AVIs（如性别、暴力、文化和种族偏见等）。我们生成了26万个AVIs，涵盖五类多模态能力（九项任务）和内容偏见。然后，我们对包括14个开源LVLMs在内的模型进行全面评估以评估其性能。AVIBench还可作为一个便利的工具

    arXiv:2403.09346v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce AVIBench, a framework designed to analyze the robustness of LVLMs when facing various adversarial visual-instructions (AVIs), including four types of image-based AVIs, ten types of text-based AVIs, and nine types of content bias AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 260K AVIs encompassing five categories of multimodal capabilities (nine tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. AVIBench also serves as a convenie
    
[^2]: 揭秘CLIP数据

    Demystifying CLIP Data. (arXiv:2309.16671v1 [cs.CV])

    [http://arxiv.org/abs/2309.16671](http://arxiv.org/abs/2309.16671)

    CLIP的成功主要归功于其数据而非模型架构或预训练目标。我们通过元数据整理方法引入了MetaCLIP，该方法从原始数据池和元数据中生成一个平衡的子集，提供了更加详细的数据信息。在实验中，我们发现MetaCLIP在处理400M个图像-文本数据对时取得了良好的性能。

    

    对比语言-图像预训练（CLIP）是一种推动计算机视觉研究和应用的方法，为现代识别系统和生成模型注入了活力。我们认为，CLIP成功的主要因素是其数据，而不是模型架构或预训练目标。然而，CLIP只提供了关于其数据和如何收集数据的非常有限的信息，导致其他研究努力通过使用模型参数进行过滤来重现CLIP的数据。在这项工作中，我们意在揭示CLIP的数据整理方法，并在公开给社区的过程中引入元数据整理的语言-图像预训练（MetaCLIP）。MetaCLIP通过对元数据分布进行平衡，从原始数据池和元数据（从CLIP的概念中得出）中产生一个平衡的子集。我们的实验研究严格隔离了模型和训练设置，仅专注于数据。MetaCLIP应用于包含400M图像-文本数据对的CommonCrawl，并获得了较好的性能。

    Contrastive Language-Image Pre-training (CLIP) is an approach that has advanced research and applications in computer vision, fueling modern recognition systems and generative models. We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective. However, CLIP only provides very limited information about its data and how it has been collected, leading to works that aim to reproduce CLIP's data by filtering with its model parameters. In this work, we intend to reveal CLIP's data curation approach and in our pursuit of making it open to the community introduce Metadata-Curated Language-Image Pre-training (MetaCLIP). MetaCLIP takes a raw data pool and metadata (derived from CLIP's concepts) and yields a balanced subset over the metadata distribution. Our experimental study rigorously isolates the model and training settings, concentrating solely on data. MetaCLIP applied to CommonCrawl with 400M image-text data pairs outper
    
[^3]: 混合神经网络领域的规范因素

    Canonical Factors for Hybrid Neural Fields. (arXiv:2308.15461v1 [cs.CV])

    [http://arxiv.org/abs/2308.15461](http://arxiv.org/abs/2308.15461)

    该研究对混合神经网络领域的规范因素进行了研究，发现因子特征容量虽然简单高效，但存在不良偏差。通过学习一组规范化变换，该方法成功消除偏差，并在图像、距离和辐射场重建任务中实现了质量、鲁棒性和运行时间的改进。

    

    因子特征容量提供了一种简单的方法来构建更紧凑、高效和可解释的神经网络领域，但也引入了不一定有益于真实世界数据的偏差。在这项工作中，我们（1）对这些体系结构对齐轴信号的不良偏差进行了表征 - 它们可以导致辐射场重建的差异高达2 PSNR - 并（2）探索了通过学习一组规范化变换来提高表示的方法，从而消除这些偏差。在一个二维模型问题中，我们证明同时学习这些变换以及场景外观可以以大大提高的效率成功。我们使用图像、有符号距离和辐射场重建任务验证了所得到的体系结构，我们观察到质量、鲁棒性、紧凑性和运行时间方面的改进。结果表明，TILTED可以实现与基线相当的能力，而基线是2倍大。

    Factored feature volumes offer a simple way to build more compact, efficient, and intepretable neural fields, but also introduce biases that are not necessarily beneficial for real-world data. In this work, we (1) characterize the undesirable biases that these architectures have for axis-aligned signals -- they can lead to radiance field reconstruction differences of as high as 2 PSNR -- and (2) explore how learning a set of canonicalizing transformations can improve representations by removing these biases. We prove in a two-dimensional model problem that simultaneously learning these transformations together with scene appearance succeeds with drastically improved efficiency. We validate the resulting architectures, which we call TILTED, using image, signed distance, and radiance field reconstruction tasks, where we observe improvements across quality, robustness, compactness, and runtime. Results demonstrate that TILTED can enable capabilities comparable to baselines that are 2x lar
    
[^4]: 深度神经网络和脑对齐：脑编码和解码（综述）

    Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey). (arXiv:2307.10246v1 [q-bio.NC])

    [http://arxiv.org/abs/2307.10246](http://arxiv.org/abs/2307.10246)

    本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。

    

    大脑如何表示不同的信息模式？我们能否设计出一个可以自动理解用户思考内容的系统？这些问题可以通过研究功能磁共振成像（fMRI）等大脑记录来回答。作为第一步，神经科学界为被动阅读/听觉/观看概念词汇、叙述、图片和电影相关的认知神经科学数据集作出了贡献。过去二十年中，还提出了使用这些数据集的编码和解码模型。这些模型作为基础研究中的额外工具，在认知科学和神经科学领域有着多种实际应用。编码模型旨在自动地生成fMRI大脑表征，给定一个刺激。它们在评估和诊断神经系统疾病以及设计大脑损伤治疗方法方面有着多种实际应用。解码模型解决了根据fMRI重构刺激的逆问题。它们对于理解大脑如何处理信息以及设计脑机接口的发展都有着重要意义。

    How does the brain represent different modes of information? Can we design a system that automatically understands what the user is thinking? Such questions can be answered by studying brain recordings like functional magnetic resonance imaging (fMRI). As a first step, the neuroscience community has contributed several large cognitive neuroscience datasets related to passive reading/listening/viewing of concept words, narratives, pictures and movies. Encoding and decoding models using these datasets have also been proposed in the past two decades. These models serve as additional tools for basic research in cognitive science and neuroscience. Encoding models aim at generating fMRI brain representations given a stimulus automatically. They have several practical applications in evaluating and diagnosing neurological conditions and thus also help design therapies for brain damage. Decoding models solve the inverse problem of reconstructing the stimuli given the fMRI. They are useful for 
    

