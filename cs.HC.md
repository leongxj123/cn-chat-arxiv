# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EgoPoser: Robust Real-Time Ego-Body Pose Estimation in Large Scenes.](http://arxiv.org/abs/2308.06493) | 本文提出了EgoPoser，一种能够在大场景中鲁棒地实时估计自我身体姿势的方法。通过重新思考输入表示、引入新的运动分解方法以及建模身体姿势，EgoPoser在定性和定量上均表现优于现有方法，并具有较高的推理速度。 |

# 详细

[^1]: EgoPoser：大场景下鲁棒的实时自我身体姿势估计

    EgoPoser: Robust Real-Time Ego-Body Pose Estimation in Large Scenes. (arXiv:2308.06493v1 [cs.CV])

    [http://arxiv.org/abs/2308.06493](http://arxiv.org/abs/2308.06493)

    本文提出了EgoPoser，一种能够在大场景中鲁棒地实时估计自我身体姿势的方法。通过重新思考输入表示、引入新的运动分解方法以及建模身体姿势，EgoPoser在定性和定量上均表现优于现有方法，并具有较高的推理速度。

    

    头部和手部姿势仅通过完整身体自我姿势估计已成为研究的一个热点领域，以为头戴式平台上的虚拟角色表达提供动力。然而，现有方法过于依赖数据集记录时的运动捕捉空间的限制，同时假设连续捕捉关节运动和均匀身体尺寸。在本文中，我们提出了EgoPoser，通过以下方式克服了这些限制：1）重新思考基于头戴式平台的自我姿势估计的输入表示，并引入一种新的运动分解方法来预测与全局位置无关的完整身体姿势，2）从头戴式设备视野内的间歇性手部姿势跟踪中鲁棒地建模身体姿势，3）针对不同用户的各种身体尺寸进行通用化推广。我们的实验表明，EgoPoser在定性和定量上优于现有的方法，并保持较高的推理速度。

    Full-body ego-pose estimation from head and hand poses alone has become an active area of research to power articulate avatar representation on headset-based platforms. However, existing methods over-rely on the confines of the motion-capture spaces in which datasets were recorded, while simultaneously assuming continuous capture of joint motions and uniform body dimensions. In this paper, we propose EgoPoser, which overcomes these limitations by 1) rethinking the input representation for headset-based ego-pose estimation and introducing a novel motion decomposition method that predicts full-body pose independent of global positions, 2) robustly modeling body pose from intermittent hand position and orientation tracking only when inside a headset's field of view, and 3) generalizing across various body sizes for different users. Our experiments show that EgoPoser outperforms state-of-the-art methods both qualitatively and quantitatively, while maintaining a high inference speed of over
    

