# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Matter of Annotation: An Empirical Study on In Situ and Self-Recall Activity Annotations from Wearable Sensors](https://arxiv.org/abs/2305.08752) | 不同的标记方法对数据质量和深度学习分类器的性能有直接影响，原位方法产生的标签较少但更精确。 |
| [^2] | [A Temporal-Spectral Fusion Transformer with Subject-specific Adapter for Enhancing RSVP-BCI Decoding.](http://arxiv.org/abs/2401.06340) | 本文提出了一种基于主题专用适配器的时间-频谱融合Transformer (TSformer-SA) 用于增强RSVP-BCI解码。该方法通过引入多视图信息并减少准备时间，实现了解码性能的提升。 |

# 详细

[^1]: 注释问题：来自可穿戴传感器的原位和自我回忆活动注释的实证研究

    A Matter of Annotation: An Empirical Study on In Situ and Self-Recall Activity Annotations from Wearable Sensors

    [https://arxiv.org/abs/2305.08752](https://arxiv.org/abs/2305.08752)

    不同的标记方法对数据质量和深度学习分类器的性能有直接影响，原位方法产生的标签较少但更精确。

    

    人们对从可穿戴传感器中检测人类活动的研究是一个高度活跃的领域，使许多应用受益，从通过健康护理患者的步行监测到健身指导再到简化手工作业流程。我们提出了一项实证研究，比较了在野外数据用户研究中使用的4种不同常用的注释方法。这些方法可以分为用户驱动的、原位注释-即在记录活动之前或期间执行的注释-和回忆方法-参与者在当天结束时追溯地对其数据进行标注。我们的研究表明，不同的标记方法直接影响注释的质量，以及相应数据训练的深度学习分类器的能力。我们注意到，原位方法产生的标签较少，但更精确，而回忆方法产生的标签较多，但不够精确。此外，我们还结合了一本活动日记

    arXiv:2305.08752v2 Announce Type: replace-cross  Abstract: Research into the detection of human activities from wearable sensors is a highly active field, benefiting numerous applications, from ambulatory monitoring of healthcare patients via fitness coaching to streamlining manual work processes. We present an empirical study that compares 4 different commonly used annotation methods utilized in user studies that focus on in-the-wild data. These methods can be grouped in user-driven, in situ annotations - which are performed before or during the activity is recorded - and recall methods - where participants annotate their data in hindsight at the end of the day. Our study illustrates that different labeling methodologies directly impact the annotations' quality, as well as the capabilities of a deep learning classifier trained with the data respectively. We noticed that in situ methods produce less but more precise labels than recall methods. Furthermore, we combined an activity diary
    
[^2]: 基于主题专用适配器的时间-频谱融合Transformer用于增强RSVP-BCI解码

    A Temporal-Spectral Fusion Transformer with Subject-specific Adapter for Enhancing RSVP-BCI Decoding. (arXiv:2401.06340v1 [cs.HC])

    [http://arxiv.org/abs/2401.06340](http://arxiv.org/abs/2401.06340)

    本文提出了一种基于主题专用适配器的时间-频谱融合Transformer (TSformer-SA) 用于增强RSVP-BCI解码。该方法通过引入多视图信息并减少准备时间，实现了解码性能的提升。

    

    快速串联视觉呈现（RSVP）基于脑机接口（BCI）是一种利用脑电信号进行目标检索的高效技术。传统解码方法的性能改进依赖于大量来自新测试对象的训练数据，这增加了BCI系统的准备时间。一些研究引入了来自现有对象的数据以减少性能改进对新对象数据的依赖性，但它们基于对抗学习的优化策略以及大量数据的训练增加了准备过程中的训练时间。此外，大多数之前的方法只关注脑电信号的单视图信息，而忽略了其他视图的信息，这可能进一步改善性能。为了在减少准备时间的同时提高解码性能，我们提出了一种具有主题专用适配器的时间-频谱融合Transformer（TSformer-SA）。

    The Rapid Serial Visual Presentation (RSVP)-based Brain-Computer Interface (BCI) is an efficient technology for target retrieval using electroencephalography (EEG) signals. The performance improvement of traditional decoding methods relies on a substantial amount of training data from new test subjects, which increases preparation time for BCI systems. Several studies introduce data from existing subjects to reduce the dependence of performance improvement on data from new subjects, but their optimization strategy based on adversarial learning with extensive data increases training time during the preparation procedure. Moreover, most previous methods only focus on the single-view information of EEG signals, but ignore the information from other views which may further improve performance. To enhance decoding performance while reducing preparation time, we propose a Temporal-Spectral fusion transformer with Subject-specific Adapter (TSformer-SA). Specifically, a cross-view interaction 
    

