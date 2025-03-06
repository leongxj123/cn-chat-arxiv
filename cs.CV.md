# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Domain Adaptation for Endoscopic Visual Odometry](https://arxiv.org/abs/2403.10860) | 这项工作提出了一个高效的内窥镜视觉里程计神经风格迁移框架，将从术前规划到测试阶段的时间缩短至不到五分钟，通过利用有限数量的真实图像和术前先验信息进行训练，以及引入测试时间自适应方法来减小训练和测试之间的光照条件差距。 |
| [^2] | [Multimodal Action Quality Assessment](https://arxiv.org/abs/2402.09444) | 该论文提出了一个名为PAMFN的渐进自适应多模态融合网络，用于多模态动作质量评估。该模型利用RGB、光流和音频信息，分别建模模态特定信息和混合模态信息，并通过充分利用音频信息，提高了评分回归的准确性。 |
| [^3] | [Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training](https://arxiv.org/abs/2312.02914) | 该方法提出了UNITE框架，利用图像教师模型和视频学生模型进行遮蔽预训练和协作自训练，在多个视频领域自适应基准上取得显著改进的结果。 |

# 详细

[^1]: 内窥镜视觉里程计的高效领域自适应

    Efficient Domain Adaptation for Endoscopic Visual Odometry

    [https://arxiv.org/abs/2403.10860](https://arxiv.org/abs/2403.10860)

    这项工作提出了一个高效的内窥镜视觉里程计神经风格迁移框架，将从术前规划到测试阶段的时间缩短至不到五分钟，通过利用有限数量的真实图像和术前先验信息进行训练，以及引入测试时间自适应方法来减小训练和测试之间的光照条件差距。

    

    arXiv:2403.10860v1 公告类型: 跨领域 摘要: 视觉里程计在内窥镜成像中起着至关重要的作用，然而缺乏具有地面真实性的图像对于学习里程计信息提出了重大挑战。因此，领域自适应为连接术前规划领域和术中实际领域学习里程计信息提供了一种有前途的方法。然而，现有方法在训练时间上存在低效性。本文提出了一种针对内窥镜视觉里程计的高效神经风格迁移框架，将从术前规划到测试阶段的时间缩短至不到五分钟。为了进行高效训练，本研究专注于用有限数量的真实图像训练模块，并利用术前先验信息大大减少训练时间。此外，在测试阶段，我们提出了一种新颖的测试时间自适应（TTA）方法来消除训练和测试之间的光照条件差距。

    arXiv:2403.10860v1 Announce Type: cross  Abstract: Visual odometry plays a crucial role in endoscopic imaging, yet the scarcity of realistic images with ground truth poses poses a significant challenge. Therefore, domain adaptation offers a promising approach to bridge the pre-operative planning domain with the intra-operative real domain for learning odometry information. However, existing methodologies suffer from inefficiencies in the training time. In this work, an efficient neural style transfer framework for endoscopic visual odometry is proposed, which compresses the time from pre-operative planning to testing phase to less than five minutes. For efficient traing, this work focuses on training modules with only a limited number of real images and we exploit pre-operative prior information to dramatically reduce training duration. Moreover, during the testing phase, we propose a novel Test Time Adaptation (TTA) method to mitigate the gap in lighting conditions between training an
    
[^2]: 多模态动作质量评估

    Multimodal Action Quality Assessment

    [https://arxiv.org/abs/2402.09444](https://arxiv.org/abs/2402.09444)

    该论文提出了一个名为PAMFN的渐进自适应多模态融合网络，用于多模态动作质量评估。该模型利用RGB、光流和音频信息，分别建模模态特定信息和混合模态信息，并通过充分利用音频信息，提高了评分回归的准确性。

    

    行动质量评估（AQA）是评估动作执行情况的方法。以往的研究仅利用视觉信息进行建模，忽视了音频信息。我们认为，虽然AQA高度依赖视觉信息，但音频也是提高评分回归准确性的有用补充信息，特别是在具有背景音乐的运动项目中，如花样滑冰和韵律体操。为了利用多模态信息进行AQA，即RGB、光流和音频信息，我们提出了一个渐进自适应多模态融合网络（PAMFN），它分别对模态特定信息和混合模态信息进行建模。我们的模型由三个模态特定分支和一个混合模态分支组成，独立地探索模态特定信息，并渐进地聚合来自模态特定分支的模态特定信息。

    arXiv:2402.09444v1 Announce Type: cross  Abstract: Action quality assessment (AQA) is to assess how well an action is performed. Previous works perform modelling by only the use of visual information, ignoring audio information. We argue that although AQA is highly dependent on visual information, the audio is useful complementary information for improving the score regression accuracy, especially for sports with background music, such as figure skating and rhythmic gymnastics. To leverage multimodal information for AQA, i.e., RGB, optical flow and audio information, we propose a Progressive Adaptive Multimodal Fusion Network (PAMFN) that separately models modality-specific information and mixed-modality information. Our model consists of with three modality-specific branches that independently explore modality-specific information and a mixed-modality branch that progressively aggregates the modality-specific information from the modality-specific branches. To build the bridge between
    
[^3]: 无监督视频域自适应：采用遮蔽预训练和协作自训练

    Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training

    [https://arxiv.org/abs/2312.02914](https://arxiv.org/abs/2312.02914)

    该方法提出了UNITE框架，利用图像教师模型和视频学生模型进行遮蔽预训练和协作自训练，在多个视频领域自适应基准上取得显著改进的结果。

    

    在这项工作中，我们解决了视频动作识别的无监督域自适应（UDA）问题。我们提出的方法称为UNITE，使用图像教师模型来调整视频学生模型到目标域。UNITE首先采用自监督预训练，通过教师引导的遮蔽蒸馏目标得到具有区分性的特征学习。然后我们对目标数据进行遮蔽自训练，利用视频学生模型和图像教师模型一起为未标记的目标视频生成改进的伪标签。我们的自训练过程成功利用了两个模型的优势，实现了跨域强大的转移性能。我们在多个视频域自适应基准上评估了我们的方法，并观察到相比先前报道的结果有显著改进。

    arXiv:2312.02914v3 Announce Type: replace-cross  Abstract: In this work, we tackle the problem of unsupervised domain adaptation (UDA) for video action recognition. Our approach, which we call UNITE, uses an image teacher model to adapt a video student model to the target domain. UNITE first employs self-supervised pre-training to promote discriminative feature learning on target domain videos using a teacher-guided masked distillation objective. We then perform self-training on masked target data, using the video student model and image teacher model together to generate improved pseudolabels for unlabeled target videos. Our self-training process successfully leverages the strengths of both models to achieve strong transfer performance across domains. We evaluate our approach on multiple video domain adaptation benchmarks and observe significant improvements upon previously reported results.
    

