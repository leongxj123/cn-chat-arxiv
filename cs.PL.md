# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Models for Code: Security Hardening and Adversarial Testing.](http://arxiv.org/abs/2302.05319) | 本研究针对大型语言模型在生成代码时缺乏安全意识，从安全加固和对抗测试的角度入手，提出了一项新的安全任务——受控代码生成，通过一种新型基于学习的方法SVEN，实现生成既安全又功能正确的代码，并对当前的LM进行对抗测试，强调了在LM的培训和评估中考虑安全因素的必要性。 |

# 详细

[^1]: 用于编码的大型语言模型：安全加固和对抗测试

    Large Language Models for Code: Security Hardening and Adversarial Testing. (arXiv:2302.05319v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2302.05319](http://arxiv.org/abs/2302.05319)

    本研究针对大型语言模型在生成代码时缺乏安全意识，从安全加固和对抗测试的角度入手，提出了一项新的安全任务——受控代码生成，通过一种新型基于学习的方法SVEN，实现生成既安全又功能正确的代码，并对当前的LM进行对抗测试，强调了在LM的培训和评估中考虑安全因素的必要性。

    

    大型语言模型(LMs)越来越多地预先在大规模代码库上进行预处理，用于生成代码。然而，LM缺乏安全意识，并经常生成不安全的代码。本研究沿着两个重要方向研究了LM的安全性:(i)安全加固，旨在增强LM在生成安全代码方面的可靠性;(ii)对抗测试，旨在在对抗性立场评估LM的安全性。我们通过制定一项称为受控代码生成的新安全任务来同时解决这两个问题。该任务是参数化的，将一个二进制属性作为输入，以指导LM生成安全或不安全的代码，同时保留LM生成功能正确代码的能力。我们提出了一种称为SVEN的新型基于学习的方法来解决这个任务。SVEN利用属性特定的连续向量来引导程序生成达到给定的属性，而不修改LM的权重。我们的训练过程通过可微分的投影损失来优化这些连续向量，实现端到端的训练。此外，我们使用SVEN进行对抗测试，并表明当前的LM容易受到攻击，在测试时修改它们的输入而保留功能。我们的工作强调需要在LM的培训和评估中考虑安全因素。

    Large language models (LMs) are increasingly pretrained on massive codebases and used to generate code. However, LMs lack awareness of security and are found to frequently produce unsafe code. This work studies the security of LMs along two important axes: (i) security hardening, which aims to enhance LMs' reliability in generating secure code, and (ii) adversarial testing, which seeks to evaluate LMs' security at an adversarial standpoint. We address both of these by formulating a new security task called controlled code generation. The task is parametric and takes as input a binary property to guide the LM to generate secure or unsafe code, while preserving the LM's capability of generating functionally correct code. We propose a novel learning-based approach called SVEN to solve this task. SVEN leverages property-specific continuous vectors to guide program generation towards the given property, without modifying the LM's weights. Our training procedure optimizes these continuous ve
    

