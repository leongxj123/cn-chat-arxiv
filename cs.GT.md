# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimality of weighted contracts for multi-agent contract design with a budget](https://arxiv.org/abs/2402.15890) | 主体与多个代理的合同设计中，加权合同是最优的选择，可以通过为代理分配正权重和优先级水平来实现最大化代理的成功概率。 |
| [^2] | [States as Strings as Strategies: Steering Language Models with Game-Theoretic Solvers](https://arxiv.org/abs/2402.01704) | 本研究提出了一种在语言模型中引入博弈论思想的方法，通过绑定博弈论的符号逻辑，使得语言模型能够通过博弈论求解器提供更加稳定和理性的对话策略。 |

# 详细

[^1]: 带预算的多代理合同设计中加权合同的最优性

    Optimality of weighted contracts for multi-agent contract design with a budget

    [https://arxiv.org/abs/2402.15890](https://arxiv.org/abs/2402.15890)

    主体与多个代理的合同设计中，加权合同是最优的选择，可以通过为代理分配正权重和优先级水平来实现最大化代理的成功概率。

    

    我们研究了一个主体与多个代理之间的合同设计问题。每个代理参与一个独立任务，结果为成功或失败，代理可以付出代价努力提高成功的概率，主体有固定预算，可以为代理提供与结果相关的奖励。关键是，我们假设主体只关心最大化代理的成功概率，而不关心预算的支出量。我们首先证明了对于某些目标，合同只有当它是成功一切的合同才是最优的。这个结果的一个直接推论是，在这种设定下，计件合同和奖金池合同从来不是最优的。然后我们证明，对于任何目标，存在一个最优的基于优先级加权的合同，这个合同为代理分配正权重和优先级水平，并将预算分配给最高优先级的成功代理。

    arXiv:2402.15890v1 Announce Type: new  Abstract: We study a contract design problem between a principal and multiple agents. Each agent participates in an independent task with binary outcomes (success or failure), in which it may exert costly effort towards improving its probability of success, and the principal has a fixed budget which it can use to provide outcome-dependent rewards to the agents. Crucially, we assume the principal cares only about maximizing the agents' probabilities of success, not how much of the budget it expends. We first show that a contract is optimal for some objective if and only if it is a successful-get-everything contract. An immediate consequence of this result is that piece-rate contracts and bonus-pool contracts are never optimal in this setting. We then show that for any objective, there is an optimal priority-based weighted contract, which assigns positive weights and priority levels to the agents, and splits the budget among the highest-priority suc
    
[^2]: 作为策略的状态字符串：用博弈论求解器引导语言模型

    States as Strings as Strategies: Steering Language Models with Game-Theoretic Solvers

    [https://arxiv.org/abs/2402.01704](https://arxiv.org/abs/2402.01704)

    本研究提出了一种在语言模型中引入博弈论思想的方法，通过绑定博弈论的符号逻辑，使得语言模型能够通过博弈论求解器提供更加稳定和理性的对话策略。

    

    博弈论是研究理性主体间战略互动的数学模型。语言是人类互动的重要方式，但在历史上一直很难通过数学方法对对话及其战略动机建模。与语言互动相关的玩家、策略和回报的适当模型（即对游戏论常规符号逻辑的约束）将使现有的博弈论算法能够在语言领域提供战略解决方案。换句话说，这种约束可以为在对话中计算稳定、理性的对话策略提供一条途径。大型语言模型（LLM）可能已经达到了其生成能力足以实现自然对话真实、类似人类的模拟的程度。通过以不同的方式提示它们，我们可以将其响应引导到不同的输出话语。利用自然语言的表达能力，LLM还可以帮助我们快速生成新的对话。

    Game theory is the study of mathematical models of strategic interactions among rational agents. Language is a key medium of interaction for humans, though it has historically proven difficult to model dialogue and its strategic motivations mathematically. A suitable model of the players, strategies, and payoffs associated with linguistic interactions (i.e., a binding to the conventional symbolic logic of game theory) would enable existing game-theoretic algorithms to provide strategic solutions in the space of language. In other words, a binding could provide a route to computing stable, rational conversational strategies in dialogue. Large language models (LLMs) have arguably reached a point where their generative capabilities can enable realistic, human-like simulations of natural dialogue. By prompting them in various ways, we can steer their responses towards different output utterances. Leveraging the expressivity of natural language, LLMs can also help us quickly generate new di
    

