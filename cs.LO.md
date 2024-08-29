# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enforcing Temporal Constraints on Generative Agent Behavior with Reactive Synthesis](https://arxiv.org/abs/2402.16905) | 提出了一种利用形式逻辑为基础的程序合成和LLM内容生成相结合的方法，通过使用时间流逻辑（TSL）对生成式代理施加时间约束，从而提高了代理行为的保证水平、系统的解释性和代理的模块化构建能力。 |

# 详细

[^1]: 利用反应合成对生成式代理行为施加时间约束

    Enforcing Temporal Constraints on Generative Agent Behavior with Reactive Synthesis

    [https://arxiv.org/abs/2402.16905](https://arxiv.org/abs/2402.16905)

    提出了一种利用形式逻辑为基础的程序合成和LLM内容生成相结合的方法，通过使用时间流逻辑（TSL）对生成式代理施加时间约束，从而提高了代理行为的保证水平、系统的解释性和代理的模块化构建能力。

    

    大型语言模型（LLM）的流行引发了对创建交互代理新方法的探索。然而，在互动过程中管理这些代理的时间行为仍然具有挑战性。我们提出了一种将形式逻辑为基础的程序合成与LLM内容生成相结合的方法，以创建遵守时间约束的生成式代理。我们的方法使用时间流逻辑（Temporal Stream Logic，TSL）生成一个自动机，对代理施加时间结构，并将每个动作的细节留给LLM。通过使用TSL，我们能够增强生成代理，使用户在行为上有更高的保证水平，系统更易解释，并且更能以模块化方式构建代理。我们评估了我们的方法……

    arXiv:2402.16905v1 Announce Type: new  Abstract: The surge in popularity of Large Language Models (LLMs) has opened doors for new approaches to the creation of interactive agents. However, managing the temporal behavior of such agents over the course of an interaction remains challenging. The stateful, long-term horizon and quantitative reasoning required for coherent agent behavior does not fit well into the LLM paradigm. We propose a combination of formal logic-based program synthesis and LLM content generation to create generative agents that adhere to temporal constraints. Our approach uses Temporal Stream Logic (TSL) to generate an automaton that enforces a temporal structure on an agent and leaves the details of each action for a moment in time to an LLM. By using TSL, we are able to augment the generative agent where users have a higher level of guarantees on behavior, better interpretability of the system, and more ability to build agents in a modular way. We evaluate our appro
    

