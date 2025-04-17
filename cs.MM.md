# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773) | StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。 |

# 详细

[^1]: StreamingT2V: 一种一致、动态和可扩展的基于文本的长视频生成方法

    StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

    [https://arxiv.org/abs/2403.14773](https://arxiv.org/abs/2403.14773)

    StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。

    

    arXiv:2403.14773v1 公告类型: 交叉 摘要: 文本到视频的扩散模型可以生成遵循文本指令的高质量视频，使得创建多样化和个性化内容变得更加容易。然而，现有方法大多集中在生成高质量的短视频（通常为16或24帧），当天真地扩展到长视频合成的情况时，通常会出现硬裁剪。为了克服这些限制，我们引入了StreamingT2V，这是一种自回归方法，用于生成80、240、600、1200或更多帧的长视频，具有平滑的过渡。主要组件包括：（i）一种名为条件注意力模块（CAM）的短期记忆块，通过注意机制将当前生成条件设置为先前块提取的特征，实现一致的块过渡，（ii）一种名为外观保存模块的长期记忆块，从第一个视频块中提取高级场景和对象特征，以防止th

    arXiv:2403.14773v1 Announce Type: cross  Abstract: Text-to-video diffusion models enable the generation of high-quality videos that follow text instructions, making it easy to create diverse and individual content. However, existing approaches mostly focus on high-quality short video generation (typically 16 or 24 frames), ending up with hard-cuts when naively extended to the case of long video synthesis. To overcome these limitations, we introduce StreamingT2V, an autoregressive approach for long video generation of 80, 240, 600, 1200 or more frames with smooth transitions. The key components are:(i) a short-term memory block called conditional attention module (CAM), which conditions the current generation on the features extracted from the previous chunk via an attentional mechanism, leading to consistent chunk transitions, (ii) a long-term memory block called appearance preservation module, which extracts high-level scene and object features from the first video chunk to prevent th
    

