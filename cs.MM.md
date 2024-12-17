# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding](https://arxiv.org/abs/2403.19723) | HGT框架结合了异质图增强的大型语言模型，通过软提示和多粒度自监督HG预训练目标，实现了少样本复杂表格理解任务的最新成果。 |
| [^2] | [Content-aware Masked Image Modeling Transformer for Stereo Image Compression](https://arxiv.org/abs/2403.08505) | 提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。 |
| [^3] | [DyRoNet: A Low-Rank Adapter Enhanced Dynamic Routing Network for Streaming Perception](https://arxiv.org/abs/2403.05050) | DyRoNet采用低秩动态路由并结合分支网络优化流媒体感知性能，为多种分支选择策略设定了新的性能标杆 |
| [^4] | [Evaluating Image Review Ability of Vision Language Models](https://arxiv.org/abs/2402.12121) | 本论文通过引入基于排名相关分析的评估方法，探讨了大规模视觉语言模型（LVLM）在生成图像评价文本方面的能力，并创建了一个评估数据集来验证这种方法。 |
| [^5] | [JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation.](http://arxiv.org/abs/2310.19180) | JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。 |

# 详细

[^1]: HGT：利用异质图增强的大型语言模型进行少样本复杂表格理解

    HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding

    [https://arxiv.org/abs/2403.19723](https://arxiv.org/abs/2403.19723)

    HGT框架结合了异质图增强的大型语言模型，通过软提示和多粒度自监督HG预训练目标，实现了少样本复杂表格理解任务的最新成果。

    

    表格理解 (TU) 取得了显著进展，但面临手动标记表格的稀缺性和复杂表格结构的挑战。为解决这些问题，我们提出了 HGT 框架，其中包含一个异质图 (HG) 增强的大型语言模型 (LLM)，用于解决少样本 TU 任务。它通过软提示和指导转换将表格语义与LLM的参数化知识对齐，并通过涉及三种新的多粒度自监督HG预训练目标的多任务预训练方案处理复杂表格。我们在几个基准测试上通过实证方法展示了HGT的有效性，表明它在少样本复杂TU方面的表现优于SOTA。

    arXiv:2403.19723v1 Announce Type: cross  Abstract: Table understanding (TU) has achieved promising advancements, but it faces the challenges of the scarcity of manually labeled tables and the presence of complex table structures.To address these challenges, we propose HGT, a framework with a heterogeneous graph (HG)-enhanced large language model (LLM) to tackle few-shot TU tasks.It leverages the LLM by aligning the table semantics with the LLM's parametric knowledge through soft prompts and instruction turning and deals with complex tables by a multi-task pre-training scheme involving three novel multi-granularity self-supervised HG pre-training objectives.We empirically demonstrate the effectiveness of HGT, showing that it outperforms the SOTA for few-shot complex TU on several benchmarks.
    
[^2]: 面向内容感知的掩码图像建模变压器用于立体图像压缩

    Content-aware Masked Image Modeling Transformer for Stereo Image Compression

    [https://arxiv.org/abs/2403.08505](https://arxiv.org/abs/2403.08505)

    提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。

    

    现有基于学习的立体图像编解码器采用了复杂的转换方法，但在编码潜在表示时却采用了从单个图像编解码器导出的简单熵模型。然而，这些熵模型难以有效捕捉立体图像固有的空间-视差特征，导致亚最优的率失真结果。本文提出了一种名为CAMSIC的立体图像压缩框架。 CAMSIC 独立地将每个图像转换为潜在表示，并采用强大的无解码器变压器熵模型来捕捉空间和视差依赖关系，引入了一种新颖的面向内容感知的掩码图像建模（MIM）技术。我们的面向内容感知的MIM促进了先验信息与估计令牌之间的高效双向交互，自然地消除了额外的Transformer解码器的需求。实验证明，我们的立体图像编解码器实现了最先进的率失真结果。

    arXiv:2403.08505v1 Announce Type: cross  Abstract: Existing learning-based stereo image codec adopt sophisticated transformation with simple entropy models derived from single image codecs to encode latent representations. However, those entropy models struggle to effectively capture the spatial-disparity characteristics inherent in stereo images, which leads to suboptimal rate-distortion results. In this paper, we propose a stereo image compression framework, named CAMSIC. CAMSIC independently transforms each image to latent representation and employs a powerful decoder-free Transformer entropy model to capture both spatial and disparity dependencies, by introducing a novel content-aware masked image modeling (MIM) technique. Our content-aware MIM facilitates efficient bidirectional interaction between prior information and estimated tokens, which naturally obviates the need for an extra Transformer decoder. Experiments show that our stereo image codec achieves state-of-the-art rate-d
    
[^3]: DyRoNet：一种低秩适配器增强的动态路由网络，用于流媒体感知

    DyRoNet: A Low-Rank Adapter Enhanced Dynamic Routing Network for Streaming Perception

    [https://arxiv.org/abs/2403.05050](https://arxiv.org/abs/2403.05050)

    DyRoNet采用低秩动态路由并结合分支网络优化流媒体感知性能，为多种分支选择策略设定了新的性能标杆

    

    自主驾驶系统需要实时、准确的感知来应对复杂环境。为解决这一问题，我们引入了动态路由网络（DyRoNet），这是一个创新性的框架，采用低秩动态路由以增强流媒体感知。通过集成专门预训练的分支网络，针对各种环境条件进行微调，DyRoNet在延迟和精度之间取得了平衡。其核心特征是速度路由模块，智能地将输入数据引导到最适合的分支网络，优化性能。广泛的评估结果显示，DyRoNet有效地适应多种分支选择策略，为各种场景性能设定了新的标杆。DyRoNet不仅为流媒体感知建立了新的标杆，还为未来的工作提供了宝贵的工程洞见。有关更多项目信息，请访问 https://tastevision.github.io/DyRoNet/

    arXiv:2403.05050v1 Announce Type: cross  Abstract: Autonomous driving systems demand real-time, accurate perception to navigate complex environments. Addressing this, we introduce the Dynamic Router Network (DyRoNet), a framework that innovates with low-rank dynamic routing for enhanced streaming perception. By integrating specialized pre-trained branch networks, fine-tuned for various environmental conditions, DyRoNet achieves a balance between latency and precision. Its core feature, the speed router module, intelligently directs input data to the best-suited branch network, optimizing performance. The extensive evaluations reveal that DyRoNet adapts effectively to multiple branch selection strategies, setting a new benchmark in performance across a range of scenarios. DyRoNet not only establishes a new benchmark for streaming perception but also provides valuable engineering insights for future work. More project information is available at https://tastevision.github.io/DyRoNet/
    
[^4]: 评估视觉语言模型的图像评价能力

    Evaluating Image Review Ability of Vision Language Models

    [https://arxiv.org/abs/2402.12121](https://arxiv.org/abs/2402.12121)

    本论文通过引入基于排名相关分析的评估方法，探讨了大规模视觉语言模型（LVLM）在生成图像评价文本方面的能力，并创建了一个评估数据集来验证这种方法。

    

    大规模视觉语言模型（LVLM）是能够通过单个模型处理图像和文本输入的语言模型。本文探讨了使用LVLM生成图像评价文本的方法。LVLM对图像的评价能力尚未完全被理解，突显了对其评价能力进行系统评估的必要性。与图像标题不同，评价文本可以从图像构图和曝光等不同视角撰写。这种评价角度的多样性使得难以唯一确定图像的正确评价。为了解决这一挑战，我们提出了一种基于排名相关分析的评估方法，通过人类和LVLM对评价文本进行排名，然后测量这些排名之间的相关性。我们进一步通过创建一个旨在评估最新LVLM图像评价能力的基准数据集来验证这种方法。

    arXiv:2402.12121v1 Announce Type: cross  Abstract: Large-scale vision language models (LVLMs) are language models that are capable of processing images and text inputs by a single model. This paper explores the use of LVLMs to generate review texts for images. The ability of LVLMs to review images is not fully understood, highlighting the need for a methodical evaluation of their review abilities. Unlike image captions, review texts can be written from various perspectives such as image composition and exposure. This diversity of review perspectives makes it difficult to uniquely determine a single correct review for an image. To address this challenge, we introduce an evaluation method based on rank correlation analysis, in which review texts are ranked by humans and LVLMs, then, measures the correlation between these rankings. We further validate this approach by creating a benchmark dataset aimed at assessing the image review ability of recent LVLMs. Our experiments with the dataset
    
[^5]: JEN-1 Composer: 一个用于高保真多音轨音乐生成的统一框架

    JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation. (arXiv:2310.19180v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2310.19180](http://arxiv.org/abs/2310.19180)

    JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。

    

    随着生成式人工智能的快速发展，从零开始生成音乐的文本到音乐合成任务已成为一个有前景的方向。然而，对于多音轨生成的更细粒度控制仍然是一个挑战。现有模型具有较强的原始生成能力，但缺乏以可控的方式单独组成和组合多音轨的灵活性，这与人类作曲家的典型工作流程不同。为了解决这个问题，我们提出了JEN-1 Composer，一个统一的框架，通过一个模型高效地建模多音轨音乐的边缘、条件和联合分布。JEN-1 Composer框架能够无缝地整合任何基于扩散的音乐生成系统，例如Jen-1，增强其多功能多音轨音乐生成能力。我们引入了一种课程训练策略，以逐步指导模型从单音轨生成到灵活的生成过程。

    With rapid advances in generative artificial intelligence, the text-to-music synthesis task has emerged as a promising direction for music generation from scratch. However, finer-grained control over multi-track generation remains an open challenge. Existing models exhibit strong raw generation capability but lack the flexibility to compose separate tracks and combine them in a controllable manner, differing from typical workflows of human composers. To address this issue, we propose JEN-1 Composer, a unified framework to efficiently model marginal, conditional, and joint distributions over multi-track music via a single model. JEN-1 Composer framework exhibits the capacity to seamlessly incorporate any diffusion-based music generation system, \textit{e.g.} Jen-1, enhancing its capacity for versatile multi-track music generation. We introduce a curriculum training strategy aimed at incrementally instructing the model in the transition from single-track generation to the flexible genera
    

