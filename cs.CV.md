# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation](https://arxiv.org/abs/2403.06759) | 提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。 |
| [^2] | [GoalNet: Goal Areas Oriented Pedestrian Trajectory Prediction](https://arxiv.org/abs/2402.19002) | 通过利用场景背景和观察到的轨迹信息，该研究提出了一种基于行人目标区域的轨迹预测神经网络，可以将不确定性限制在几个目标区域内。 |
| [^3] | [SDR-GAIN: A High Real-Time Occluded Pedestrian Pose Completion Method for Autonomous Driving.](http://arxiv.org/abs/2306.03538) | SDR-GAIN是一种用于解决行人姿态中部分遮挡问题的关键点补全方法，它通过对不完整的关键点进行降维，统一特征分布，并使用GAN框架的两种生成模型来完成姿态的补全。该方法的实验表明性能优于基本的GAIN框架。 |

# 详细

[^1]: 平均校准误差：一种可微损失函数，用于改善图像分割中的可靠性

    Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation

    [https://arxiv.org/abs/2403.06759](https://arxiv.org/abs/2403.06759)

    提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。

    

    医学图像分割的深度神经网络经常产生与经验观察不一致的过于自信的结果，这种校准错误挑战着它们的临床应用。我们提出使用平均L1校准误差（mL1-ACE）作为一种新颖的辅助损失函数，以改善像素级校准而不会损害分割质量。我们展示了，尽管使用硬分箱，这种损失是直接可微的，避免了需要近似但可微的替代或软分箱方法的必要性。我们的工作还引入了数据集可靠性直方图的概念，这一概念推广了标准的可靠性图，用于在数据集级别聚合的语义分割中细化校准的视觉评估。使用mL1-ACE，我们将平均和最大校准误差分别降低了45%和55%，同时在BraTS 2021数据集上保持了87%的Dice分数。我们在这里分享我们的代码: https://github

    arXiv:2403.06759v1 Announce Type: cross  Abstract: Deep neural networks for medical image segmentation often produce overconfident results misaligned with empirical observations. Such miscalibration, challenges their clinical translation. We propose to use marginal L1 average calibration error (mL1-ACE) as a novel auxiliary loss function to improve pixel-wise calibration without compromising segmentation quality. We show that this loss, despite using hard binning, is directly differentiable, bypassing the need for approximate but differentiable surrogate or soft binning approaches. Our work also introduces the concept of dataset reliability histograms which generalises standard reliability diagrams for refined visual assessment of calibration in semantic segmentation aggregated at the dataset level. Using mL1-ACE, we reduce average and maximum calibration error by 45% and 55% respectively, maintaining a Dice score of 87% on the BraTS 2021 dataset. We share our code here: https://github
    
[^2]: GoalNet: 面向目标区域的行人轨迹预测

    GoalNet: Goal Areas Oriented Pedestrian Trajectory Prediction

    [https://arxiv.org/abs/2402.19002](https://arxiv.org/abs/2402.19002)

    通过利用场景背景和观察到的轨迹信息，该研究提出了一种基于行人目标区域的轨迹预测神经网络，可以将不确定性限制在几个目标区域内。

    

    预测道路上行人未来的轨迹是自动驾驶中的重要任务。行人轨迹预测受场景路径、行人意图和决策影响，这是一个多模态问题。最近的研究大多使用过去的轨迹来预测各种潜在的未来轨迹分布，这并未考虑场景背景和行人目标。我们提出了一种不直接预测未来轨迹的方法，即首先使用场景背景和观察到的轨迹来预测目标点，然后重复使用目标点来预测未来轨迹。通过利用场景背景和观察到的轨迹信息，我们可以将不确定性限制在几个目标区域内，这些区域代表了行人的“目标”。在本文中，我们提出了GoalNet，一种基于行人目标区域的新轨迹预测神经网络。

    arXiv:2402.19002v1 Announce Type: cross  Abstract: Predicting the future trajectories of pedestrians on the road is an important task for autonomous driving. The pedestrian trajectory prediction is affected by scene paths, pedestrian's intentions and decision-making, which is a multi-modal problem. Most recent studies use past trajectories to predict a variety of potential future trajectory distributions, which do not account for the scene context and pedestrian targets. Instead of predicting the future trajectory directly, we propose to use scene context and observed trajectory to predict the goal points first, and then reuse the goal points to predict the future trajectories. By leveraging the information from scene context and observed trajectory, the uncertainty can be limited to a few target areas, which represent the "goals" of the pedestrians. In this paper, we propose GoalNet, a new trajectory prediction neural network based on the goal areas of a pedestrian. Our network can pr
    
[^3]: SDR-GAIN：一种用于自动驾驶的高实时遮挡行人姿态完成方法

    SDR-GAIN: A High Real-Time Occluded Pedestrian Pose Completion Method for Autonomous Driving. (arXiv:2306.03538v1 [cs.CV])

    [http://arxiv.org/abs/2306.03538](http://arxiv.org/abs/2306.03538)

    SDR-GAIN是一种用于解决行人姿态中部分遮挡问题的关键点补全方法，它通过对不完整的关键点进行降维，统一特征分布，并使用GAN框架的两种生成模型来完成姿态的补全。该方法的实验表明性能优于基本的GAIN框架。

    

    为了缓解基于人体姿态关键点的行人检测算法中部分遮挡带来的挑战，我们提出了一种称为分离和降维基于生成对抗性补全网络(SDR-GAIN)的新型行人姿势关键点补全方法。首先，我们利用OpenPose在图像中估计行人的姿态。然后，我们对由于遮挡或其他因素而不完整的行人头部和躯干关键点进行维度缩减，以增强特征并进一步统一特征分布。最后，我们引入了基于生成对抗网络(GAN)框架的两种生成模型，这些模型融合了Huber损失、残差结构和L1正则化来生成部分遮挡行人不完整头部和躯干姿态关键点的缺失部分，从而实现了姿态补全。我们在MS COCO和JAAD数据集上的实验表明，SDR-GAIN的性能优于基本的GAIN框架。

    To mitigate the challenges arising from partial occlusion in human pose keypoint based pedestrian detection methods , we present a novel pedestrian pose keypoint completion method called the separation and dimensionality reduction-based generative adversarial imputation networks (SDR-GAIN) . Firstly, we utilize OpenPose to estimate pedestrian poses in images. Then, we isolate the head and torso keypoints of pedestrians with incomplete keypoints due to occlusion or other factors and perform dimensionality reduction to enhance features and further unify feature distribution. Finally, we introduce two generative models based on the generative adversarial networks (GAN) framework, which incorporate Huber loss, residual structure, and L1 regularization to generate missing parts of the incomplete head and torso pose keypoints of partially occluded pedestrians, resulting in pose completion. Our experiments on MS COCO and JAAD datasets demonstrate that SDR-GAIN outperforms basic GAIN framework
    

