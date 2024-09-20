# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A novel framework for adaptive stress testing of autonomous vehicles in highways](https://arxiv.org/abs/2402.11813) | 提出了一种新颖的框架，利用自适应压力测试方法和深度强化学习来系统探索可能导致高速公路交通场景中安全问题的边界情况 |
| [^2] | [Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation](https://arxiv.org/abs/2402.07127) |  |

# 详细

[^1]: 自适应压力测试高速公路自动驾驶车辆的新框架

    A novel framework for adaptive stress testing of autonomous vehicles in highways

    [https://arxiv.org/abs/2402.11813](https://arxiv.org/abs/2402.11813)

    提出了一种新颖的框架，利用自适应压力测试方法和深度强化学习来系统探索可能导致高速公路交通场景中安全问题的边界情况

    

    保证自动驾驶车辆（AVs）的安全运行对于它们的广泛应用和公众接受至关重要。因此，不仅对AV进行标准安全测试的评估，还发现可能导致不安全行为或情况的被测试AV的潜在边界情况具有极其重要的意义。本文提出了一个新颖的框架，用于系统地探索可能导致高速公路交通场景中安全问题的边界情况。该框架基于一种自适应压力测试（AST）方法，这是一种利用马尔可夫决策过程制定场景以及深度强化学习（DRL）发现代表边界情况的理想模式的新兴验证方法。为此，我们为DRL开发了一个新的奖励函数，以指导AST根据被测试AV（即自车）与其他车辆之间的碰撞概率估计来识别碰撞场景。

    arXiv:2402.11813v1 Announce Type: cross  Abstract: Guaranteeing the safe operations of autonomous vehicles (AVs) is crucial for their widespread adoption and public acceptance. It is thus of a great significance to not only assess the AV against the standard safety tests, but also discover potential corner cases of the AV under test that could lead to unsafe behaviour or scenario. In this paper, we propose a novel framework to systematically explore corner cases that can result in safety concerns in a highway traffic scenario. The framework is based on an adaptive stress testing (AST) approach, an emerging validation method that leverages a Markov decision process to formulate the scenarios and deep reinforcement learning (DRL) to discover the desirable patterns representing corner cases. To this end, we develop a new reward function for DRL to guide the AST in identifying crash scenarios based on the collision probability estimate between the AV under test (i.e., the ego vehicle) and 
    
[^2]: 观察学习：基于视频的机器人操作学习方法综述

    Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation

    [https://arxiv.org/abs/2402.07127](https://arxiv.org/abs/2402.07127)

    

    

    机器人学习操作技能受到多样化、无偏的数据集的稀缺性的影响。尽管策划的数据集可以帮助解决问题，但在泛化性和现实世界的转移方面仍然存在挑战。与此同时，“野外”视频数据集的大规模存在通过自监督技术推动了计算机视觉的进展。将这一点应用到机器人领域，最近的研究探索了通过被动观察来学习丰富的在线视频中的操作技能。这种基于视频的学习范式显示出了有希望的结果，它提供了可扩展的监督方法，同时降低了数据集的偏见。本综述回顾了视频特征表示学习技术、物体可行性理解、三维手部/身体建模和大规模机器人资源等基础知识，以及从不受控制的视频演示中获取机器人操作技能的新兴技术。我们讨论了仅从观察大规模人类视频中学习如何增强机器人的泛化性和样本效率。

    Robot learning of manipulation skills is hindered by the scarcity of diverse, unbiased datasets. While curated datasets can help, challenges remain in generalizability and real-world transfer. Meanwhile, large-scale "in-the-wild" video datasets have driven progress in computer vision through self-supervised techniques. Translating this to robotics, recent works have explored learning manipulation skills by passively watching abundant videos sourced online. Showing promising results, such video-based learning paradigms provide scalable supervision while reducing dataset bias. This survey reviews foundations such as video feature representation learning techniques, object affordance understanding, 3D hand/body modeling, and large-scale robot resources, as well as emerging techniques for acquiring robot manipulation skills from uncontrolled video demonstrations. We discuss how learning only from observing large-scale human videos can enhance generalization and sample efficiency for roboti
    

