# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Adversarial Attacks on Latent Diffusion Model](https://arxiv.org/abs/2310.04687) | 提出了一种改进 Latent Diffusion Model 的对抗攻击方法 ACE，其通过统一模式的额外误差来促使模型学习特定的偏差，从而胜过了目前最先进的方法 |
| [^2] | [Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations](https://arxiv.org/abs/2310.04566) | 本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。 |

# 详细

[^1]: 改进潜在扩散模型的对抗攻击

    Improving Adversarial Attacks on Latent Diffusion Model

    [https://arxiv.org/abs/2310.04687](https://arxiv.org/abs/2310.04687)

    提出了一种改进 Latent Diffusion Model 的对抗攻击方法 ACE，其通过统一模式的额外误差来促使模型学习特定的偏差，从而胜过了目前最先进的方法

    

    对 Latent Diffusion Model (LDM)，这种最先进的图像生成模型，进行对抗攻击已经被证明是有效防止 LDM 在未经授权的图像上进行恶意微调的保护手段。我们展示了这些攻击会对 LDM 预测的对抗样本的评分函数添加额外的误差。在这些对抗样本上进行微调的 LDM 学习通过一个偏差降低误差，从而遭受攻击并使用偏差预测评分函数。基于这一动态，我们提出了通过一致得分函数错误进行攻击（ACE）来改进 LDM 的对抗攻击。ACE 统一了添加到预测得分函数的额外误差的模式。这促使微调的 LDM 学习与对评分函数进行预测的偏差学习相同的模式。然后我们引入一个精心设计的模式来改进攻击。我们的方法在对 LDM 的对抗攻击中胜过了最先进的方法。

    arXiv:2310.04687v3 Announce Type: replace-cross  Abstract: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.
    
[^2]: Knolling Bot: 从整洁的示范中学习机器人对象排列

    Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations

    [https://arxiv.org/abs/2310.04566](https://arxiv.org/abs/2310.04566)

    本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。

    

    地址：arXiv:2310.04566v2  公告类型：replace-cross  摘要：解决家庭空间中散乱物品的整理挑战受到整洁性的多样性和主观性的复杂性影响。正如人类语言的复杂性允许同一理念的多种表达一样，家庭整洁偏好和组织模式变化广泛，因此预设物体位置将限制对新物体和环境的适应性。受自然语言处理（NLP）的进展启发，本文引入一种自监督学习框架，使机器人能够从整洁布局的示范中理解和复制整洁的概念，类似于使用会话数据集训练大语言模型（LLM）。我们利用一个Transformer神经网络来预测后续物体的摆放位置。我们展示了一个“整理”系统，利用机械臂和RGB相机在桌子上组织不同大小和数量的物品。

    arXiv:2310.04566v2 Announce Type: replace-cross  Abstract: Addressing the challenge of organizing scattered items in domestic spaces is complicated by the diversity and subjective nature of tidiness. Just as the complexity of human language allows for multiple expressions of the same idea, household tidiness preferences and organizational patterns vary widely, so presetting object locations would limit the adaptability to new objects and environments. Inspired by advancements in natural language processing (NLP), this paper introduces a self-supervised learning framework that allows robots to understand and replicate the concept of tidiness from demonstrations of well-organized layouts, akin to using conversational datasets to train Large Language Models(LLM). We leverage a transformer neural network to predict the placement of subsequent objects. We demonstrate a ``knolling'' system with a robotic arm and an RGB camera to organize items of varying sizes and quantities on a table. Our 
    

