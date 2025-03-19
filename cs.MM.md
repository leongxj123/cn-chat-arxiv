# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis](https://arxiv.org/abs/2403.13501) | 提出了一种名为VSTAR的方法，通过引入生成时序护理（GTN）的概念，自动生成视频梗概并改善对时序动态的控制，从而实现生成更长、更动态的视频 |

# 详细

[^1]: VSTAR：用于生成长动态视频合成的时间护理

    VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis

    [https://arxiv.org/abs/2403.13501](https://arxiv.org/abs/2403.13501)

    提出了一种名为VSTAR的方法，通过引入生成时序护理（GTN）的概念，自动生成视频梗概并改善对时序动态的控制，从而实现生成更长、更动态的视频

    

    尽管在文本到视频（T2V）合成领域取得了巨大进展，但开源的T2V扩散模型难以生成具有动态变化和不断进化内容的较长视频。它们往往合成准静态视频，忽略了文本提示中涉及的必要随时间变化的视觉变化。与此同时，将这些模型扩展到实现更长、更动态的视频合成往往在计算上难以处理。为了解决这一挑战，我们引入了生成时序护理（GTN）的概念，旨在在推理过程中即时改变生成过程，以改善对时序动态的控制，并实现生成更长的视频。我们提出了一种GTN方法，名为VSTAR，它包括两个关键要素：1）视频梗概提示（VSP）-基于原始单个提示自动生成视频梗概，利用LLMs提供准确的文本指导，以实现对时序动态的精确控制。

    arXiv:2403.13501v1 Announce Type: cross  Abstract: Despite tremendous progress in the field of text-to-video (T2V) synthesis, open-sourced T2V diffusion models struggle to generate longer videos with dynamically varying and evolving content. They tend to synthesize quasi-static videos, ignoring the necessary visual change-over-time implied in the text prompt. At the same time, scaling these models to enable longer, more dynamic video synthesis often remains computationally intractable. To address this challenge, we introduce the concept of Generative Temporal Nursing (GTN), where we aim to alter the generative process on the fly during inference to improve control over the temporal dynamics and enable generation of longer videos. We propose a method for GTN, dubbed VSTAR, which consists of two key ingredients: 1) Video Synopsis Prompting (VSP) - automatic generation of a video synopsis based on the original single prompt leveraging LLMs, which gives accurate textual guidance to differe
    

