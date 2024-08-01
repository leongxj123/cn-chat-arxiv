# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Self-Supervised Pre-Training of Graph Foundation Models: A Knowledge-Based Perspective](https://arxiv.org/abs/2403.16137) | 该论文从基于知识的角度全面调查和分析了图基础模型的自监督预训练任务，涉及微观和宏观知识，包括9个知识类别、25个预训练任务以及各种下游任务适应策略。 |
| [^2] | [Disentangled Condensation for Large-scale Graphs.](http://arxiv.org/abs/2401.12231) | 本文提出了用于大规模图的解缠结凝聚方法DisCo，通过节点和边的凝聚模块实现了对大规模图的高效缩凝，提高了可扩展性和压缩图的保真度。 |
| [^3] | [In-class Data Analysis Replications: Teaching Students while Testing Science.](http://arxiv.org/abs/2308.16491) | 这项研究揭示了课堂数据分析复制的可行性，以及这种方法对学生、教育者和科学家的成本与收益。同时，学生对数据的预期与实际情况存在差异。 |

# 详细

[^1]: 自监督预训练图基础模型的调查：基于知识的视角

    A Survey on Self-Supervised Pre-Training of Graph Foundation Models: A Knowledge-Based Perspective

    [https://arxiv.org/abs/2403.16137](https://arxiv.org/abs/2403.16137)

    该论文从基于知识的角度全面调查和分析了图基础模型的自监督预训练任务，涉及微观和宏观知识，包括9个知识类别、25个预训练任务以及各种下游任务适应策略。

    

    图自监督学习现在是预训练图基础模型的首选方法，包括图神经网络、图变换器，以及更近期的基于大型语言模型（LLM）的图模型。文章全面调查和分析了基于知识的视角下的图基础模型的预训练任务，包括微观（节点、链接等）和宏观知识（簇、全局结构等）。涵盖了共计9个知识类别和25个预训练任务，以及各种下游任务适应策略。

    arXiv:2403.16137v1 Announce Type: new  Abstract: Graph self-supervised learning is now a go-to method for pre-training graph foundation models, including graph neural networks, graph transformers, and more recent large language model (LLM)-based graph models. There is a wide variety of knowledge patterns embedded in the structure and properties of graphs which may be used for pre-training, but we lack a systematic overview of self-supervised pre-training tasks from the perspective of graph knowledge. In this paper, we comprehensively survey and analyze the pre-training tasks of graph foundation models from a knowledge-based perspective, consisting of microscopic (nodes, links, etc) and macroscopic knowledge (clusters, global structure, etc). It covers a total of 9 knowledge categories and 25 pre-training tasks, as well as various downstream task adaptation strategies. Furthermore, an extensive list of the related papers with detailed metadata is provided at https://github.com/Newiz430/
    
[^2]: 大规模图的解缠结凝聚

    Disentangled Condensation for Large-scale Graphs. (arXiv:2401.12231v1 [cs.SI])

    [http://arxiv.org/abs/2401.12231](http://arxiv.org/abs/2401.12231)

    本文提出了用于大规模图的解缠结凝聚方法DisCo，通过节点和边的凝聚模块实现了对大规模图的高效缩凝，提高了可扩展性和压缩图的保真度。

    

    图解缠结已经成为一种有趣的技术，为大规模图提供了一种更紧凑但信息丰富的小图，以节省大规模图学习的昂贵成本。尽管取得了有前途的结果，但先前的图解缠结方法常常采用纠缠的缩凝策略，同时涉及节点和边的缩凝，导致大量的GPU内存需求。这种纠缠的策略极大地阻碍了图解缠结的可扩展性，削弱了它对极大规模图的缩凝和高保真度压缩图的能力。因此，本文提出了用于大规模图的解缠结凝聚，简称为DisCo，以提供可扩展的图解缠结，适用于不同规模的图。DisCo的核心是两个互补的组件，即节点和边的凝聚模块，在解缠的方式下实现节点和边的凝聚。

    Graph condensation has emerged as an intriguing technique to provide Graph Neural Networks for large-scale graphs with a more compact yet informative small graph to save the expensive costs of large-scale graph learning. Despite the promising results achieved, previous graph condensation methods often employ an entangled condensation strategy that involves condensing nodes and edges simultaneously, leading to substantial GPU memory demands. This entangled strategy has considerably impeded the scalability of graph condensation, impairing its capability to condense extremely large-scale graphs and produce condensed graphs with high fidelity. Therefore, this paper presents Disentangled Condensation for large-scale graphs, abbreviated as DisCo, to provide scalable graph condensation for graphs of varying sizes. At the heart of DisCo are two complementary components, namely node and edge condensation modules, that realize the condensation of nodes and edges in a disentangled manner. In the 
    
[^3]: 课堂数据分析复制：教学生，同时测试科学

    In-class Data Analysis Replications: Teaching Students while Testing Science. (arXiv:2308.16491v1 [cs.CY])

    [http://arxiv.org/abs/2308.16491](http://arxiv.org/abs/2308.16491)

    这项研究揭示了课堂数据分析复制的可行性，以及这种方法对学生、教育者和科学家的成本与收益。同时，学生对数据的预期与实际情况存在差异。

    

    科学正面临可重复性危机。先前的工作提出将数据分析复制纳入课堂作为潜在解决方案。然而，尽管潜在的好处，目前尚不清楚这一方法是否可行，如果可行，涉及的利益相关者-学生、教育者和科学家-应该期望什么。学生能够在课堂上进行数据分析复制吗？教育者的成本与收益如何？这个解决方案如何帮助评估和改进科学的现状？本研究在EPFL教授的应用数据分析课程（CS-401）的项目部分中纳入了数据分析复制（N=354名学生）。在此报告中，我们基于课程期间进行的调查提前进行注册的发现。首先，我们证明学生可以复制先前发表的科学论文，大部分是定性的，有些是完全一样的。我们发现学生对数据的预期与实际情况存在差异

    Science is facing a reproducibility crisis. Previous work has proposed incorporating data analysis replications into classrooms as a potential solution. However, despite the potential benefits, it is unclear whether this approach is feasible, and if so, what the involved stakeholders-students, educators, and scientists-should expect from it. Can students perform a data analysis replication over the course of a class? What are the costs and benefits for educators? And how can this solution help benchmark and improve the state of science?  In the present study, we incorporated data analysis replications in the project component of the Applied Data Analysis course (CS-401) taught at EPFL (N=354 students). Here we report pre-registered findings based on surveys administered throughout the course. First, we demonstrate that students can replicate previously published scientific papers, most of them qualitatively and some exactly. We find discrepancies between what students expect of data an
    

