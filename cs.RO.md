# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping.](http://arxiv.org/abs/2309.06038) | 本文提出了一个名为“人类助力灵巧抓取”的新型任务，通过使用Grasping Gradient Field和基于历史条件的残差策略，训练控制机器人手指以适应不同用户意图和物体几何形状的灵巧抓取操作。 |

# 详细

[^1]: 为人类助力灵巧抓取学习基于得分的抓取原语

    Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping. (arXiv:2309.06038v1 [cs.RO])

    [http://arxiv.org/abs/2309.06038](http://arxiv.org/abs/2309.06038)

    本文提出了一个名为“人类助力灵巧抓取”的新型任务，通过使用Grasping Gradient Field和基于历史条件的残差策略，训练控制机器人手指以适应不同用户意图和物体几何形状的灵巧抓取操作。

    

    在本文中，我们提出了一种名为“人类助力灵巧抓取”的新型任务，旨在训练控制机器人手指以帮助用户抓取物体的策略。与传统的灵巧抓取不同，这个任务面临着更复杂的挑战，因为策略需要适应不同的用户意图和物体的几何形状。我们通过提出一个由两个子模块组成的方法来解决这个挑战：一种手-物体条件抓取原语称为Grasping Gradient Field（GraspGF），以及一种基于历史条件的残差策略。GraspGF通过估计来自成功抓取示例集的梯度来学习“如何”抓取，而残差策略根据轨迹历史确定“何时”和以何种速度执行抓取动作。实验结果证明了我们方法的有效性。

    The use of anthropomorphic robotic hands for assisting individuals in situations where human hands may be unavailable or unsuitable has gained significant importance. In this paper, we propose a novel task called human-assisting dexterous grasping that aims to train a policy for controlling a robotic hand's fingers to assist users in grasping objects. Unlike conventional dexterous grasping, this task presents a more complex challenge as the policy needs to adapt to diverse user intentions, in addition to the object's geometry. We address this challenge by proposing an approach consisting of two sub-modules: a hand-object-conditional grasping primitive called Grasping Gradient Field~(GraspGF), and a history-conditional residual policy. GraspGF learns `how' to grasp by estimating the gradient from a success grasping example set, while the residual policy determines `when' and at what speed the grasping action should be executed based on the trajectory history. Experimental results demons
    

