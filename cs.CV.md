# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds.](http://arxiv.org/abs/2308.09908) | 本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。 |
| [^2] | [Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder.](http://arxiv.org/abs/2303.15564) | 本文提出了利用掩码自编码器的盲目防御框架（BDMAE），可以在测试时防御盲目后门攻击，不需要验证数据和模型参数，通过测试图像和 MAE 还原之间的结构相似性和标签一致性来检测后门攻击。 |

# 详细

[^1]: LEGO: 对于基于点云的在线多目标跟踪的学习和图优化的模块化跟踪器

    LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds. (arXiv:2308.09908v1 [cs.CV])

    [http://arxiv.org/abs/2308.09908](http://arxiv.org/abs/2308.09908)

    本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。

    

    在线多目标跟踪（MOT）在自主系统中起着关键作用。现有的最先进方法通常采用跟踪-检测方法，数据关联起到了至关重要的作用。本文提出了一个学习和图优化（LEGO）的模块化跟踪器，以提高数据关联性能。所提出的LEGO跟踪器集成了图优化和自注意力机制，能够有效地制定关联评分图，从而实现准确高效的目标匹配。为了进一步增强状态更新过程，本文还添加了卡尔曼滤波器，通过将对象状态的时间连贯性纳入跟踪中，确保一致的跟踪。与其他在线跟踪方法（包括基于LiDAR和基于LiDAR-相机融合的方法）相比，我们提出的仅利用LiDAR的方法表现出了卓越性能。在提交结果至KITTI目标跟踪评估排行榜时，LEGO排名第一。

    Online multi-object tracking (MOT) plays a pivotal role in autonomous systems. The state-of-the-art approaches usually employ a tracking-by-detection method, and data association plays a critical role. This paper proposes a learning and graph-optimized (LEGO) modular tracker to improve data association performance in the existing literature. The proposed LEGO tracker integrates graph optimization and self-attention mechanisms, which efficiently formulate the association score map, facilitating the accurate and efficient matching of objects across time frames. To further enhance the state update process, the Kalman filter is added to ensure consistent tracking by incorporating temporal coherence in the object states. Our proposed method utilizing LiDAR alone has shown exceptional performance compared to other online tracking approaches, including LiDAR-based and LiDAR-camera fusion-based methods. LEGO ranked 1st at the time of submitting results to KITTI object tracking evaluation ranki
    
[^2]: 掩码还原技术：利用掩码自编码器在测试时防御盲目后门攻击

    Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder. (arXiv:2303.15564v1 [cs.LG])

    [http://arxiv.org/abs/2303.15564](http://arxiv.org/abs/2303.15564)

    本文提出了利用掩码自编码器的盲目防御框架（BDMAE），可以在测试时防御盲目后门攻击，不需要验证数据和模型参数，通过测试图像和 MAE 还原之间的结构相似性和标签一致性来检测后门攻击。

    

    深度神经网络容易受到恶意攻击，攻击者会通过在图像上叠加特殊的触发器来恶意操纵模型行为，这称为后门攻击。现有的后门防御方法通常需要访问一些验证数据和模型参数，这在许多实际应用中是不切实际的，例如当模型作为云服务提供时。为了解决这个问题，本文致力于测试时的盲目后门防御实践，特别是针对黑盒模型。每个测试图像的真实标签需要从可疑模型的硬标签预测中恢复。然而，在图像空间中启发式触发器搜索不适用于复杂触发器或高分辨率的图片。我们通过利用通用图像生成模型，提出了一种利用掩码自编码器的盲目防御框架（BDMAE），通过测试图像和 MAE 还原之间的结构相似性和标签一致性来检测后门攻击。

    Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detec
    

