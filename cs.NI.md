# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Learning-based Target-To-User Association in Integrated Sensing and Communication Systems.](http://arxiv.org/abs/2401.12801) | 本文提出了一种深度学习方法，用于在集成感知和通信系统中将雷达目标与通信用户设备进行关联。该方法通过对雷达数据进行处理，实现了联合多目标检测和波束推理。这一方法在主动切换和波束预测等通信任务中具有潜在应用价值。 |

# 详细

[^1]: 深度学习在集成感知与通信系统中基于目标到用户关联的应用

    Deep Learning-based Target-To-User Association in Integrated Sensing and Communication Systems. (arXiv:2401.12801v1 [cs.NI])

    [http://arxiv.org/abs/2401.12801](http://arxiv.org/abs/2401.12801)

    本文提出了一种深度学习方法，用于在集成感知和通信系统中将雷达目标与通信用户设备进行关联。该方法通过对雷达数据进行处理，实现了联合多目标检测和波束推理。这一方法在主动切换和波束预测等通信任务中具有潜在应用价值。

    

    在集成感知与通信（ISAC）系统中，将雷达目标与通信用户设备（UEs）进行匹配对于几种通信任务是有意义的，如主动切换和波束预测。本文考虑了一个雷达辅助通信系统，一个基站（BS）配备有多输入多输出（MIMO）雷达，雷达具有双重目标：（i）将车辆雷达目标与通信波束空间中的车辆设备（VEs）关联起来，（ii）根据雷达数据预测每个VE的波束形成矢量。提出的目标到用户（T2U）关联分为两个阶段。首先，通过距离-角度图像检测车辆雷达目标，并为每个目标估计一个波束形成矢量。然后，将推断得到的每个目标的波束形成矢量与BS用于通信的波束形成矢量进行匹配，以执行目标到用户（T2U）关联。通过修改你只看脸部网络（YOLO）算法实现了联合多目标检测和波束推理。

    In Integrated Sensing and Communication (ISAC) systems, matching the radar targets with communication user equipments (UEs) is functional to several communication tasks, such as proactive handover and beam prediction. In this paper, we consider a radar-assisted communication system where a base station (BS) is equipped with a multiple-input-multiple-output (MIMO) radar that has a double aim: (i) associate vehicular radar targets to vehicular equipments (VEs) in the communication beamspace and (ii) predict the beamforming vector for each VE from radar data. The proposed target-to-user (T2U) association consists of two stages. First, vehicular radar targets are detected from range-angle images, and, for each, a beamforming vector is estimated. Then, the inferred per-target beamforming vectors are matched with the ones utilized at the BS for communication to perform target-to-user (T2U) association. Joint multi-target detection and beam inference is obtained by modifying the you only look
    

