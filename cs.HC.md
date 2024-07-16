# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Android in the Zoo: Chain-of-Action-Thought for GUI Agents](https://arxiv.org/abs/2403.02713) | 该研究提出了一个名为CoAT的Chain-of-Action-Thought模型，通过考虑先前动作描述、当前屏幕情况以及未来动作思考，显著提高了智能手机GUI代理的任务执行效果。 |
| [^2] | [Large Language Models and Games: A Survey and Roadmap](https://arxiv.org/abs/2402.18659) | 这项研究调查了大型语言模型在游戏领域中的多种应用及其角色，指出了未开发领域和未来发展方向，同时探讨了在游戏领域中大型语言模型的潜力和限制。 |

# 详细

[^1]: Android在动物园中: GUI代理的动作思维链

    Android in the Zoo: Chain-of-Action-Thought for GUI Agents

    [https://arxiv.org/abs/2403.02713](https://arxiv.org/abs/2403.02713)

    该研究提出了一个名为CoAT的Chain-of-Action-Thought模型，通过考虑先前动作描述、当前屏幕情况以及未来动作思考，显著提高了智能手机GUI代理的任务执行效果。

    

    大型语言模型（LLM）导致智能手机上的大量自主GUI代理激增，这些代理通过预测API的一系列动作来完成由自然语言触发的任务。尽管该任务高度依赖于过去的动作和视觉观察，但现有研究通常很少考虑中间截图和屏幕操作传递的语义信息。为了解决这一问题，本文提出了动作思维链（CoAT），它考虑了先前动作的描述、当前屏幕，更重要的是分析应当执行的动作以及选择的动作带来的结果。我们证明，在使用现成LLM进行零次学习的情况下，CoAT相比于标准上下文建模显著提高了目标的完成情况。为了进一步促进这一研究领域的发展，我们构建了一个名为Android-In-The-Zoo（AitZ）的基准测试集，其中包含18,643个屏幕动作对。

    arXiv:2403.02713v1 Announce Type: new  Abstract: Large language model (LLM) leads to a surge of autonomous GUI agents for smartphone, which completes a task triggered by natural language through predicting a sequence of actions of API. Even though the task highly relies on past actions and visual observations, existing studies typical consider little semantic information carried out by intermediate screenshots and screen operations. To address this, this work presents Chain-of-Action-Thought (dubbed CoAT), which takes the description of the previous actions, the current screen, and more importantly the action thinking of what actions should be performed and the outcomes led by the chosen action. We demonstrate that, in a zero-shot setting upon an off-the-shell LLM, CoAT significantly improves the goal progress compared to standard context modeling. To further facilitate the research in this line, we construct a benchmark Android-In-The-Zoo (AitZ), which contains 18,643 screen-action pa
    
[^2]: 大型语言模型与游戏：调研与路线图

    Large Language Models and Games: A Survey and Roadmap

    [https://arxiv.org/abs/2402.18659](https://arxiv.org/abs/2402.18659)

    这项研究调查了大型语言模型在游戏领域中的多种应用及其角色，指出了未开发领域和未来发展方向，同时探讨了在游戏领域中大型语言模型的潜力和限制。

    

    近年来，大型语言模型（LLMs）的研究急剧增加，并伴随着公众对该主题的参与。尽管起初是自然语言处理中的一小部分，LLMs在广泛的应用和领域中展现出显著潜力，包括游戏。本文调查了LLMs在游戏中及为游戏提供支持的各种应用的最新技术水平，并明确了LLMs在游戏中可以扮演的不同角色。重要的是，我们讨论了尚未开发的领域和LLMs在游戏中未来应用的有前途的方向，以及在游戏领域中LLMs的潜力和限制。作为LLMs和游戏交叉领域的第一份综合调查和路线图，我们希望本文能够成为这一激动人心的新领域的开创性研究和创新的基础。

    arXiv:2402.18659v1 Announce Type: cross  Abstract: Recent years have seen an explosive increase in research on large language models (LLMs), and accompanying public engagement on the topic. While starting as a niche area within natural language processing, LLMs have shown remarkable potential across a broad range of applications and domains, including games. This paper surveys the current state of the art across the various applications of LLMs in and for games, and identifies the different roles LLMs can take within a game. Importantly, we discuss underexplored areas and promising directions for future uses of LLMs in games and we reconcile the potential and limitations of LLMs within the games domain. As the first comprehensive survey and roadmap at the intersection of LLMs and games, we are hopeful that this paper will serve as the basis for groundbreaking research and innovation in this exciting new field.
    

