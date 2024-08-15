# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VIRUS-NeRF -- Vision, InfraRed and UltraSonic based Neural Radiance Fields](https://arxiv.org/abs/2403.09477) | VIRUS-NeRF是基于视觉、红外和超声波的神经辐射场，通过整合超声波和红外传感器的深度测量数据，实现了在自主移动机器人中达到与LiDAR点云相媲美的映射性能。 |

# 详细

[^1]: 基于视觉、红外和超声波的神经辐射场——VIRUS-NeRF

    VIRUS-NeRF -- Vision, InfraRed and UltraSonic based Neural Radiance Fields

    [https://arxiv.org/abs/2403.09477](https://arxiv.org/abs/2403.09477)

    VIRUS-NeRF是基于视觉、红外和超声波的神经辐射场，通过整合超声波和红外传感器的深度测量数据，实现了在自主移动机器人中达到与LiDAR点云相媲美的映射性能。

    

    自主移动机器人在现代工厂和仓库操作中起着越来越重要的作用。障碍物检测、回避和路径规划是关键的安全相关任务，通常使用昂贵的LiDAR传感器和深度摄像头来解决。我们提出使用成本效益的低分辨率测距传感器，如超声波和红外时间飞行传感器，通过开发基于视觉、红外和超声波的神经辐射场(VIRUS-NeRF)来解决这一问题。VIRUS-NeRF构建在瞬时神经图形基元与多分辨率哈希编码(Instant-NGP)的基础上，融合了超声波和红外传感器的深度测量数据，并利用它们来更新用于光线跟踪的占据网格。在2D实验评估中，VIRUS-NeRF实现了与LiDAR点云相媲美的映射性能，尤其在小型环境中，其准确性与LiDAR测量相符。

    arXiv:2403.09477v1 Announce Type: cross  Abstract: Autonomous mobile robots are an increasingly integral part of modern factory and warehouse operations. Obstacle detection, avoidance and path planning are critical safety-relevant tasks, which are often solved using expensive LiDAR sensors and depth cameras. We propose to use cost-effective low-resolution ranging sensors, such as ultrasonic and infrared time-of-flight sensors by developing VIRUS-NeRF - Vision, InfraRed, and UltraSonic based Neural Radiance Fields. Building upon Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (Instant-NGP), VIRUS-NeRF incorporates depth measurements from ultrasonic and infrared sensors and utilizes them to update the occupancy grid used for ray marching. Experimental evaluation in 2D demonstrates that VIRUS-NeRF achieves comparable mapping performance to LiDAR point clouds regarding coverage. Notably, in small environments, its accuracy aligns with that of LiDAR measurements, whi
    

