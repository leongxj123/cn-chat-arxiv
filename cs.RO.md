# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Vision-Guided Robotic System for Grasping Harvested Tomato Trusses in Cluttered Environments.](http://arxiv.org/abs/2309.17170) | 提出了一种用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统。该系统利用基于深度学习的视觉系统来识别穗果并确定适合抓取的位置，通过在线学习来排序抓取姿势，并实现无触觉传感器或几何模型的夹持抓取。实验表明，该系统具有100%的清理率和93%的一次性成功抓取率。 |
| [^2] | [Continual Learning through Human-Robot Interaction -- Human Perceptions of a Continual Learning Robot in Repeated Interactions.](http://arxiv.org/abs/2305.16332) | 本研究结合机器人和连续学习模型，通过人机交互的方式与60名参与者实验，结果表明使用连续学习可以提高机器人的能力，参与者更倾向于与其进行反复交互，并提供更多的反馈信息。 |

# 详细

[^1]: 用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统

    A Vision-Guided Robotic System for Grasping Harvested Tomato Trusses in Cluttered Environments. (arXiv:2309.17170v1 [cs.RO])

    [http://arxiv.org/abs/2309.17170](http://arxiv.org/abs/2309.17170)

    提出了一种用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统。该系统利用基于深度学习的视觉系统来识别穗果并确定适合抓取的位置，通过在线学习来排序抓取姿势，并实现无触觉传感器或几何模型的夹持抓取。实验表明，该系统具有100%的清理率和93%的一次性成功抓取率。

    

    目前，对于西红柿的称重和包装需要大量的人工操作。自动化的主要障碍在于开发一个可靠的用于已采摘的穗果的机器人抓取系统的困难。我们提出了一种方法来抓取堆放在装箱中的穗果，这是它们在采摘后常见的存储和运输方式。该方法包括一个基于深度学习的视觉系统，首先识别出装箱中的单个穗果，然后确定茎部的适合抓取的位置。为此，我们引入了一个具有在线学习能力的抓取姿势排序算法。在选择了最有前景的抓取姿势之后，机器人执行一种无需触觉传感器或几何模型的夹持抓取。实验室实验证明，配备了一个手眼一体的RGB-D相机的机器人操纵器从堆中捡起所有的穗果的清理率达到100%。93%的穗果在第一次尝试时成功抓取。

    Currently, truss tomato weighing and packaging require significant manual work. The main obstacle to automation lies in the difficulty of developing a reliable robotic grasping system for already harvested trusses. We propose a method to grasp trusses that are stacked in a crate with considerable clutter, which is how they are commonly stored and transported after harvest. The method consists of a deep learning-based vision system to first identify the individual trusses in the crate and then determine a suitable grasping location on the stem. To this end, we have introduced a grasp pose ranking algorithm with online learning capabilities. After selecting the most promising grasp pose, the robot executes a pinch grasp without needing touch sensors or geometric models. Lab experiments with a robotic manipulator equipped with an eye-in-hand RGB-D camera showed a 100% clearance rate when tasked to pick all trusses from a pile. 93% of the trusses were successfully grasped on the first try,
    
[^2]: 通过人机交互实现连续学习--人类在与机器人反复交互中对机器人连续学习的看法

    Continual Learning through Human-Robot Interaction -- Human Perceptions of a Continual Learning Robot in Repeated Interactions. (arXiv:2305.16332v1 [cs.RO])

    [http://arxiv.org/abs/2305.16332](http://arxiv.org/abs/2305.16332)

    本研究结合机器人和连续学习模型，通过人机交互的方式与60名参与者实验，结果表明使用连续学习可以提高机器人的能力，参与者更倾向于与其进行反复交互，并提供更多的反馈信息。

    

    为了在动态的实际环境中长期部署辅助机器人，机器人必须继续学习和适应其环境。研究人员已经开发了各种连续学习（CL）的计算模型，可以使机器人不断从有限的训练数据中学习，并避免遗忘先前的知识。虽然这些CL模型可以缓解静态、系统地收集的数据集上的遗忘，但人们目前尚不清楚在多次交互中连续学习的机器人是如何被人类用户所感知的。在本研究中，我们开发了一个系统，将目标识别的CL模型与Fetch移动操纵机器人进行整合，并允许人类参与者在多个会话中直接教授和测试机器人。我们开展了一项现场研究，60名参与者在300个会话中与我们的系统互动（每个参与者5次会话）。我们进行了一项两组实验的研究，并使用三种不同的CL模型（三个实验条件）来了解人类对连续学习机器人的看法。

    For long-term deployment in dynamic real-world environments, assistive robots must continue to learn and adapt to their environments. Researchers have developed various computational models for continual learning (CL) that can allow robots to continually learn from limited training data, and avoid forgetting previous knowledge. While these CL models can mitigate forgetting on static, systematically collected datasets, it is unclear how human users might perceive a robot that continually learns over multiple interactions with them. In this paper, we developed a system that integrates CL models for object recognition with a Fetch mobile manipulator robot and allows human participants to directly teach and test the robot over multiple sessions. We conducted an in-person study with 60 participants who interacted with our system in 300 sessions (5 sessions per participant). We conducted a between-participant study with three different CL models (3 experimental conditions) to understand huma
    

