# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation.](http://arxiv.org/abs/2309.16118) | D$^3$Fields是一个动态的三维描述符场，将底层三维环境的动态特性以及语义特征和实例掩模编码起来。它可以灵活地使用不同背景、风格和实例的二维图像指定目标，实现零样本机器人操作任务的可泛化。 |

# 详细

[^1]: D$^3$Fields: 动态三维描述符场用于零样本可泛化机器人操作

    D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation. (arXiv:2309.16118v1 [cs.RO])

    [http://arxiv.org/abs/2309.16118](http://arxiv.org/abs/2309.16118)

    D$^3$Fields是一个动态的三维描述符场，将底层三维环境的动态特性以及语义特征和实例掩模编码起来。它可以灵活地使用不同背景、风格和实例的二维图像指定目标，实现零样本机器人操作任务的可泛化。

    

    场景表示是机器人操作系统中一个关键的设计选择。一个理想的表示应该是三维的、动态的和语义化的，以满足不同操作任务的需求。然而，先前的工作往往同时缺乏这三个属性。在这项工作中，我们介绍了D$^3$Fields动态三维描述符场。这些场捕捉了底层三维环境的动态特性，编码了语义特征和实例掩模。具体而言，我们将工作区域中的任意三维点投影到多视角的二维视觉观察中，并插值从基本模型中得到的特征。由此得到的融合描述符场可以使用具有不同背景、风格和实例的二维图像灵活地指定目标。为了评估这些描述符场的有效性，我们以零样本方式将我们的表示应用于各种机器人操作任务。通过在真实场景和模拟中的广泛评估，我们展示了该方法的有效性。

    Scene representation has been a crucial design choice in robotic manipulation systems. An ideal representation should be 3D, dynamic, and semantic to meet the demands of diverse manipulation tasks. However, previous works often lack all three properties simultaneously. In this work, we introduce D$^3$Fields dynamic 3D descriptor fields. These fields capture the dynamics of the underlying 3D environment and encode both semantic features and instance masks. Specifically, we project arbitrary 3D points in the workspace onto multi-view 2D visual observations and interpolate features derived from foundational models. The resulting fused descriptor fields allow for flexible goal specifications using 2D images with varied contexts, styles, and instances. To evaluate the effectiveness of these descriptor fields, we apply our representation to a wide range of robotic manipulation tasks in a zero-shot manner. Through extensive evaluation in both real-world scenarios and simulations, we demonst
    

