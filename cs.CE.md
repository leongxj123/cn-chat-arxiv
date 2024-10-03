# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction](https://arxiv.org/abs/2403.08055) | DrivAerNet提供了一个大规模高保真度的汽车数据集，以解决工程应用中训练深度学习模型所需的数据不足问题，而RegDGCNN利用这一数据集直接从3D网格提供高精度的阻力估计。 |

# 详细

[^1]: DrivAerNet：用于数据驱动气动设计和基于图的阻力预测的参数化汽车数据集

    DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction

    [https://arxiv.org/abs/2403.08055](https://arxiv.org/abs/2403.08055)

    DrivAerNet提供了一个大规模高保真度的汽车数据集，以解决工程应用中训练深度学习模型所需的数据不足问题，而RegDGCNN利用这一数据集直接从3D网格提供高精度的阻力估计。

    

    本研究介绍了 DrivAerNet，这是一个大规模高保真度的CFD数据集，其中包含3D工业标准汽车形状，以及 RegDGCNN，这是一个动态图卷积神经网络模型，旨在通过机器学习进行汽车气动设计。DrivAerNet拥有4000个详细的3D汽车网格，使用50万个表面网格面和全面气动性能数据，包括完整的3D压力、速度场和壁面剪切应力，满足了工程应用中训练深度学习模型所需的大规模数据集的迫切需求。它比先前可用的最大公开汽车数据集大60％，也是唯一同时模拟轮毂和底盘的开源数据集。RegDGCNN利用这一大规模数据集，直接从3D网格提供高精度的阻力估计，绕过了传统限制，如需要2D图像渲染或符号距离场（SDF）。

    arXiv:2403.08055v1 Announce Type: new  Abstract: This study introduces DrivAerNet, a large-scale high-fidelity CFD dataset of 3D industry-standard car shapes, and RegDGCNN, a dynamic graph convolutional neural network model, both aimed at aerodynamic car design through machine learning. DrivAerNet, with its 4000 detailed 3D car meshes using 0.5 million surface mesh faces and comprehensive aerodynamic performance data comprising of full 3D pressure, velocity fields, and wall-shear stresses, addresses the critical need for extensive datasets to train deep learning models in engineering applications. It is 60\% larger than the previously available largest public dataset of cars, and is the only open-source dataset that also models wheels and underbody. RegDGCNN leverages this large-scale dataset to provide high-precision drag estimates directly from 3D meshes, bypassing traditional limitations such as the need for 2D image rendering or Signed Distance Fields (SDF). By enabling fast drag e
    

