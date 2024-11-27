# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Generalizable Feature Fields for Mobile Manipulation](https://arxiv.org/abs/2403.07563) | 提出了GeFF（通用特征场），作为导航和操作的统一表示，可以实时执行，通过将生成的丰富场景先验与自然语言对齐来提高效果。 |
| [^2] | [DexTouch: Learning to Seek and Manipulate Objects with Tactile Dexterity.](http://arxiv.org/abs/2401.12496) | 本论文介绍了一种利用触觉灵巧性寻找和操作物体的多指机器人系统。通过使用触觉传感器进行物体搜索和操作，我们证明了即使在没有依赖于视觉信息的情况下，机器人也能够具备类似人类的触觉能力。 |

# 详细

[^1]: 学习移动操作的通用特征场

    Learning Generalizable Feature Fields for Mobile Manipulation

    [https://arxiv.org/abs/2403.07563](https://arxiv.org/abs/2403.07563)

    提出了GeFF（通用特征场），作为导航和操作的统一表示，可以实时执行，通过将生成的丰富场景先验与自然语言对齐来提高效果。

    

    移动操作中的一个悬而未决的问题是如何以统一的方式表示物体和场景，使得机器人可以同时用于在环境中导航和操作物体。本工作提出了GeFF（通用特征场），这是一个场景级的通用神经特征场，作为导航和操作的统一表示，可以实时执行。为此，我们将生成新视图合成视为一个预训练任务，然后通过CLIP特征提炼将生成的丰富场景先验与自然语言对齐。

    arXiv:2403.07563v1 Announce Type: cross  Abstract: An open problem in mobile manipulation is how to represent objects and scenes in a unified manner, so that robots can use it both for navigating in the environment and manipulating objects. The latter requires capturing intricate geometry while understanding fine-grained semantics, whereas the former involves capturing the complexity inherit to an expansive physical scale. In this work, we present GeFF (Generalizable Feature Fields), a scene-level generalizable neural feature field that acts as a unified representation for both navigation and manipulation that performs in real-time. To do so, we treat generative novel view synthesis as a pre-training task, and then align the resulting rich scene priors with natural language via CLIP feature distillation. We demonstrate the effectiveness of this approach by deploying GeFF on a quadrupedal robot equipped with a manipulator. We evaluate GeFF's ability to generalize to open-set objects as 
    
[^2]: DexTouch：学习使用触觉灵巧性寻找和操作物体

    DexTouch: Learning to Seek and Manipulate Objects with Tactile Dexterity. (arXiv:2401.12496v1 [cs.RO])

    [http://arxiv.org/abs/2401.12496](http://arxiv.org/abs/2401.12496)

    本论文介绍了一种利用触觉灵巧性寻找和操作物体的多指机器人系统。通过使用触觉传感器进行物体搜索和操作，我们证明了即使在没有依赖于视觉信息的情况下，机器人也能够具备类似人类的触觉能力。

    

    触觉能力对于熟练执行各种任务是至关重要的，它能够在没有依赖于视觉信息的情况下搜索和操作物体。随着时间的推移，已经进行了大量研究将人类的触觉能力应用于机器人。在本文中，我们介绍了一个多指机器人系统，旨在利用触觉感受器搜索和操作物体，而不依赖于视觉信息。使用触觉传感器来搜索随机放置的目标物体，并进行模拟日常任务的物体操作。本研究的目标是赋予机器人类似人类的触觉能力。为了实现这一目标，我们在机器人手的一侧实现了二值触觉传感器，以尽量减少模拟与真实环境之间的差距。通过在仿真中通过强化学习训练策略，并将训练好的策略转移到真实环境中，我们证明了使用触觉传感器进行物体搜索和操作是可行的，即使在没有依赖于视觉信息的情况下。

    The sense of touch is an essential ability for skillfully performing a variety of tasks, providing the capacity to search and manipulate objects without relying on visual information. Extensive research has been conducted over time to apply these human tactile abilities to robots. In this paper, we introduce a multi-finger robot system designed to search for and manipulate objects using the sense of touch without relying on visual information. Randomly located target objects are searched using tactile sensors, and the objects are manipulated for tasks that mimic daily-life. The objective of the study is to endow robots with human-like tactile capabilities. To achieve this, binary tactile sensors are implemented on one side of the robot hand to minimize the Sim2Real gap. Training the policy through reinforcement learning in simulation and transferring the trained policy to the real environment, we demonstrate that object search and manipulation using tactile sensors is possible even in 
    

