# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models.](http://arxiv.org/abs/2310.10378) | 本论文研究了多语言预训练语言模型中事实知识的跨语言一致性，提出了一种新的度量方法，并通过分析模型大小、语言配对等因素发现了影响一致性的因素。实验结果表明，增加模型大小可以提高准确性，但不会改善跨语言一致性。 |
| [^2] | [PlanFitting: Tailoring Personalized Exercise Plans with Large Language Models.](http://arxiv.org/abs/2309.12555) | PlanFitting是一个对话型人工智能，利用大型语言模型的生成能力帮助用户定制个性化的运动计划，并在用户研究中证明了它生成个性化、可操作和有据可依的运动计划的潜力。 |

# 详细

[^1]: 跨语言多语言模型中事实知识的跨语言一致性

    Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models. (arXiv:2310.10378v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.10378](http://arxiv.org/abs/2310.10378)

    本论文研究了多语言预训练语言模型中事实知识的跨语言一致性，提出了一种新的度量方法，并通过分析模型大小、语言配对等因素发现了影响一致性的因素。实验结果表明，增加模型大小可以提高准确性，但不会改善跨语言一致性。

    

    多语言大规模预训练语言模型（PLM）显示存储了大量的事实知识，但在不同语言之间存在较大的变化。为了确保不同语言背景的用户从同一个模型中获得一致的反馈，我们研究了各种多语言PLM中事实知识的跨语言一致性（CLC）。为此，我们提出了一种基于排序的一致性（RankC）度量，用于独立于准确性评估跨语言间的知识一致性。利用这个度量方法，我们对决定CLC的因素进行了深入分析，包括模型层面和语言对层面。在其他结果中，我们发现增加模型大小可以提高大多数语言中的事实探测准确性，但不能改善跨语言一致性。最后，我们通过模型编辑在PLMs中插入新的事实关联进行了一个CLC的案例研究。对一小部分事实进行了实验。

    Multilingual large-scale Pretrained Language Models (PLMs) have been shown to store considerable amounts of factual knowledge, but large variations are observed across languages. With the ultimate goal of ensuring that users with different language backgrounds obtain consistent feedback from the same model, we study the cross-lingual consistency (CLC) of factual knowledge in various multilingual PLMs. To this end, we propose a Ranking-based Consistency (RankC) metric to evaluate knowledge consistency across languages independently from accuracy. Using this metric, we conduct an in-depth analysis of the determining factors for CLC, both at model level and at language-pair level. Among other results, we find that increasing model size leads to higher factual probing accuracy in most languages, but does not improve cross-lingual consistency. Finally, we conduct a case study on CLC when new factual associations are inserted in the PLMs via model editing. Results on a small sample of facts 
    
[^2]: PlanFitting：利用大型语言模型定制个性化的运动计划

    PlanFitting: Tailoring Personalized Exercise Plans with Large Language Models. (arXiv:2309.12555v1 [cs.HC])

    [http://arxiv.org/abs/2309.12555](http://arxiv.org/abs/2309.12555)

    PlanFitting是一个对话型人工智能，利用大型语言模型的生成能力帮助用户定制个性化的运动计划，并在用户研究中证明了它生成个性化、可操作和有据可依的运动计划的潜力。

    

    个性化的运动计划对于确保足够的体育活动至关重要，但由于人们的复杂日程和考虑因素以及计划的创建通常需要与专家的反复沟通，这一过程变得具有挑战性。我们提出了PlanFitting，它是一个对话型人工智能，可以辅助个性化的运动计划。通过利用大型语言模型的生成能力，PlanFitting使用户能够用自然语言描述各种约束和查询，从而便于创建和优化适合其特定情况的每周运动计划，并保持基本原则的扎根。通过一项用户研究，参与者（N=18）使用PlanFitting生成个性化的运动计划，而专家规划者（N=3）对这些计划进行评估，我们确定了PlanFitting在生成个性化、可操作和有据可依的运动计划方面的潜力。我们还讨论了AI助手在创建计划方面的未来设计机遇。

    A personally tailored exercise regimen is crucial to ensuring sufficient physical activities, yet challenging to create as people have complex schedules and considerations and the creation of plans often requires iterations with experts. We present PlanFitting, a conversational AI that assists in personalized exercise planning. Leveraging generative capabilities of large language models, PlanFitting enables users to describe various constraints and queries in natural language, thereby facilitating the creation and refinement of their weekly exercise plan to suit their specific circumstances while staying grounded in foundational principles. Through a user study where participants (N=18) generated a personalized exercise plan using PlanFitting and expert planners (N=3) evaluated these plans, we identified the potential of PlanFitting in generating personalized, actionable, and evidence-based exercise plans. We discuss future design opportunities for AI assistants in creating plans that 
    

