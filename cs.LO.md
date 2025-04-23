# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LASER: Neuro-Symbolic Learning of Semantic Video Representations.](http://arxiv.org/abs/2304.07647) | LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。 |

# 详细

[^1]: LASER：神经符号学习语义视频表示

    LASER: Neuro-Symbolic Learning of Semantic Video Representations. (arXiv:2304.07647v1 [cs.CV])

    [http://arxiv.org/abs/2304.07647](http://arxiv.org/abs/2304.07647)

    LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。

    

    现代涉及视频的AI应用（如视频-文本对齐、视频搜索和视频字幕）受益于对视频语义的细致理解。现有的视频理解方法要么需要大量注释，要么基于不可解释的通用嵌入，可能会忽略重要细节。我们提出了LASER，这是一种神经符号方法，通过利用能够捕捉视频数据中丰富的时空属性的逻辑规范来学习语义视频表示。特别地，我们通过原始视频与规范之间的对齐来公式化问题。对齐过程有效地训练了低层感知模型，以提取符合所需高层规范的细粒度视频表示。我们的流程可以端到端地训练，并可纳入从规范导出的对比和语义损失函数。我们在两个具有丰富空间和时间信息的数据集上评估了我们的方法。

    Modern AI applications involving video, such as video-text alignment, video search, and video captioning, benefit from a fine-grained understanding of video semantics. Existing approaches for video understanding are either data-hungry and need low-level annotation, or are based on general embeddings that are uninterpretable and can miss important details. We propose LASER, a neuro-symbolic approach that learns semantic video representations by leveraging logic specifications that can capture rich spatial and temporal properties in video data. In particular, we formulate the problem in terms of alignment between raw videos and specifications. The alignment process efficiently trains low-level perception models to extract a fine-grained video representation that conforms to the desired high-level specification. Our pipeline can be trained end-to-end and can incorporate contrastive and semantic loss functions derived from specifications. We evaluate our method on two datasets with rich sp
    

