# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ComSD: Balancing Behavioral Quality and Diversity in Unsupervised Skill Discovery.](http://arxiv.org/abs/2309.17203) | ComSD提出了一种新方法，通过更合理的互信息估计和动态加权的内在奖励来平衡无监督技能发现中的行为质量和多样性。 |

# 详细

[^1]: ComSD: 在无监督技能发现中平衡行为质量和多样性

    ComSD: Balancing Behavioral Quality and Diversity in Unsupervised Skill Discovery. (arXiv:2309.17203v1 [cs.LG])

    [http://arxiv.org/abs/2309.17203](http://arxiv.org/abs/2309.17203)

    ComSD提出了一种新方法，通过更合理的互信息估计和动态加权的内在奖励来平衡无监督技能发现中的行为质量和多样性。

    

    无监督技能发现的理想方法能够在没有外部奖励的情况下产生多样且合格的技能，同时发现的技能集能够以各种方式高效地适应下游任务。本文提出了Contrastive multi-objectives Skill Discovery (ComSD)，通过更合理的互信息估计和动态加权的内在奖励来减轻发现的行为在质量和多样性之间的冲突。

    Learning diverse and qualified behaviors for utilization and adaptation without supervision is a key ability of intelligent creatures. Ideal unsupervised skill discovery methods are able to produce diverse and qualified skills in the absence of extrinsic reward, while the discovered skill set can efficiently adapt to downstream tasks in various ways. Maximizing the Mutual Information (MI) between skills and visited states can achieve ideal skill-conditioned behavior distillation in theory. However, it's difficult for recent advanced methods to well balance behavioral quality (exploration) and diversity (exploitation) in practice, which may be attributed to the unreasonable MI estimation by their rigid intrinsic reward design. In this paper, we propose Contrastive multi-objectives Skill Discovery (ComSD) which tries to mitigate the quality-versus-diversity conflict of discovered behaviors through a more reasonable MI estimation and a dynamically weighted intrinsic reward. ComSD proposes
    

