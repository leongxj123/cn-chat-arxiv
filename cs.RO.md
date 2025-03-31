# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamics-Guided Diffusion Model for Robot Manipulator Design](https://arxiv.org/abs/2402.15038) | 该论文提出了动态引导扩散模型，利用共享的动力学网络为不同操作任务生成 manipulator 几何设计，通过设计目标构建的梯度引导手指几何设计的完善过程。 |

# 详细

[^1]: 动态引导扩散模型用于机器人 manipulator 设计

    Dynamics-Guided Diffusion Model for Robot Manipulator Design

    [https://arxiv.org/abs/2402.15038](https://arxiv.org/abs/2402.15038)

    该论文提出了动态引导扩散模型，利用共享的动力学网络为不同操作任务生成 manipulator 几何设计，通过设计目标构建的梯度引导手指几何设计的完善过程。

    

    我们提出了一个名为动态引导扩散模型的数据驱动框架，用于为给定操作任务生成 manipulator 几何设计。与为每个任务训练不同的设计模型不同，我们的方法采用一个跨任务共享的学习动力学网络。对于新的操作任务，我们首先将其分解为一组称为目标相互作用配置文件的个别运动目标，其中每个个别运动可以由共享的动力学网络建模。从目标和预测的相互作用配置文件构建的设计目标为任务的手指几何设计提供了梯度引导。这个设计过程被执行为一种分类器引导的扩散过程，其中设计目标作为分类器引导。我们在只使用开环平行夹爪运动的无传感器设置下，在各种操作任务上评估了我们的框架。

    arXiv:2402.15038v1 Announce Type: cross  Abstract: We present Dynamics-Guided Diffusion Model, a data-driven framework for generating manipulator geometry designs for a given manipulation task. Instead of training different design models for each task, our approach employs a learned dynamics network shared across tasks. For a new manipulation task, we first decompose it into a collection of individual motion targets which we call target interaction profile, where each individual motion can be modeled by the shared dynamics network. The design objective constructed from the target and predicted interaction profiles provides a gradient to guide the refinement of finger geometry for the task. This refinement process is executed as a classifier-guided diffusion process, where the design objective acts as the classifier guidance. We evaluate our framework on various manipulation tasks, under the sensor-less setting using only an open-loop parallel jaw motion. Our generated designs outperfor
    

