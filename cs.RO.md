# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences](https://arxiv.org/abs/2402.17257) | RIME是一种针对嘈杂偏好的健壮PbRL算法，通过动态过滤去噪偏好和热启动奖励模型，极大增强了现有最先进PbRL方法的鲁棒性。 |

# 详细

[^1]: RIME: 具有嘈杂偏好的健壮偏好强化学习

    RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences

    [https://arxiv.org/abs/2402.17257](https://arxiv.org/abs/2402.17257)

    RIME是一种针对嘈杂偏好的健壮PbRL算法，通过动态过滤去噪偏好和热启动奖励模型，极大增强了现有最先进PbRL方法的鲁棒性。

    

    偏好强化学习（PbRL）通过利用人类偏好作为奖励信号，避免了对奖励设计的需求。然而，当前PbRL算法过度依赖来自领域专家的高质量反馈，导致缺乏鲁棒性。在本文中，我们提出了RIME，一种针对嘈杂偏好的健壮PbRL算法，用于有效地从嘈杂偏好中学习奖励。我们的方法结合了基于样本选择的鉴别器，动态过滤去噪偏好以进行健壮训练。为了减轻选择不正确造成的累积误差，我们提出热启动奖励模型，此外还能填补PbRL中从预训练到在线训练过渡时的性能差距。我们在机器人操纵和运动任务上的实验表明，RIME显著提升了当前最先进的PbRL方法的鲁棒性。消融研究进一步表明，热启动

    arXiv:2402.17257v1 Announce Type: cross  Abstract: Preference-based Reinforcement Learning (PbRL) avoids the need for reward engineering by harnessing human preferences as the reward signal. However, current PbRL algorithms over-reliance on high-quality feedback from domain experts, which results in a lack of robustness. In this paper, we present RIME, a robust PbRL algorithm for effective reward learning from noisy preferences. Our method incorporates a sample selection-based discriminator to dynamically filter denoised preferences for robust training. To mitigate the accumulated error caused by incorrect selection, we propose to warm start the reward model, which additionally bridges the performance gap during transition from pre-training to online training in PbRL. Our experiments on robotic manipulation and locomotion tasks demonstrate that RIME significantly enhances the robustness of the current state-of-the-art PbRL method. Ablation studies further demonstrate that the warm star
    

