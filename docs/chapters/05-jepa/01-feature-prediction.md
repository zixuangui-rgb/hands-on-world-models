# 5.1　为什么是 JEPA：从世界模型到预测表征

探索机器智能的边界，一直是人工智能研究者长期追求的目标。

机器非常擅长计算，拥有远超人类的计算能力。随着以 ChatGPT 为代表的大语言模型兴起，机器在目标、规则和评价标准较为明确的任务上展现出了强大执行能力，并且这种能力还在以非常快的速度提升。

但是，即便机器智能在一些特定的任务上拥有非常强的表现，其局限性依然非常明显。

现实世界并不是一道已经出好的题。环境不会告诉机器什么重要、规则是什么，也不会为每个行为即时评分。机器若要自主行动，就必须自己理解环境、预测后果，并随变化调整计划。对人类和动物来说，这就像是天生的能力，但是对于机器而言，这并不简单。

Yann LeCun 认为，这是机器走向自主智能必须补上的关键缺口。2022 年，他在 [_A Path Towards Autonomous Machine Intelligence_](https://openreview.net/forum?id=BZ5a1r-kVsf) 中系统提出了一套以世界模型为核心的智能架构，并将 JEPA 作为其中的重要组成部分。

**JEPA 不是一个完整的智能系统，而是一种学习内部世界状态的架构思想。它不要求模型还原观测的全部细节，而是在表示空间中进行预测，从而保留对理解和预测世界变化有用的信息。它试图回答的是：机器能否从感知经验中自主学到一种能够描述环境、预测变化，并最终支持规划与行动的内部状态？**

## 5.1.1　从完整架构出发：机器究竟要预测什么？

### 一套完整的自主智能架构

LeCun 在文章中画出了一套自主智能体的完整蓝图，并将 JEPA 放进其中，作为学习世界模型的一种方案。

这套架构要形成一个闭环：**感知现在 → 预想未来 → 评价结果 → 选择行动 → 再次感知。**

JEPA 主要处理其中的“如何表示和预测世界”，并不独自决定目标或动作。

![LeCun 2022 年提出的自主机器智能架构](/jepa/lecun-2022-autonomous-intelligence-architecture.png)

_图 5.1　LeCun 提出的自主机器智能架构。图中 Cost 由 Intrinsic Cost 与 Critic 两部分组成；为保持图面清晰，Configurator 从其他模块接收信息的箭头没有画出。来源：[LeCun, 2022, Figure 2](https://openreview.net/forum?id=BZ5a1r-kVsf)；六个模块的简要说明也可参见 [Meta 官方介绍](https://ai.meta.com/blog/yann-lecun-advances-in-ai-research/)。_

整套架构包含六个模块。每个模块只回答一个问题：

| 模块                  | 作用                                           | 它回答的问题       |
| --------------------- | ---------------------------------------------- | ------------------ |
| **Perception**        | 从传感器输入中估计当前状态                     | 现在发生了什么？   |
| **World Model**       | 补全缺失信息，并预测自然演化或动作造成的未来   | 接下来可能怎样？   |
| **Cost**              | 用固有代价与可学习的 Critic 评价状态           | 哪种结果更好？     |
| **Actor**             | 提出并优化候选动作序列                         | 接下来怎么做？     |
| **Short-term Memory** | 保存过去、当前与预测的状态及其代价             | 刚才发生了什么？   |
| **Configurator**      | 根据当前任务调整感知、世界模型、代价与动作模块 | 此刻应该关注什么？ |

它的工作方式并不复杂。Actor 先提出几组候选动作；World Model 分别预演这些动作可能带来的后果；Cost 比较这些后果；Actor 执行当前看来最合适的一步。新的观测到来后，系统重新估计状态、重新预测、重新规划。

这张图是一份**研究蓝图**，不是已经完成的系统。论文明确留下了许多尚未解决的问题，例如怎样训练 Configurator、Critic 和长期可用的世界模型。

### 为什么“预测”是必要的？

在这套架构的规划路径中，Perception 只能告诉机器“现在怎样”，Cost 只能评价“这个状态好不好”，Actor 只能提出“可以怎么做”。真正把动作与后果连接起来的，是 World Model：

```text
候选动作  →  World Model  →  可能的未来状态  →  Cost
   ↑                                                │
   └──────────── 选择预测代价更低的动作 ────────────┘
```

用一个最简化的式子表示，就是：

$$
a^* = \arg\min_a C\!\left(W(s,a)\right)
$$

其中，\(s\) 是当前状态，\(a\) 是候选动作，\(W(s,a)\) 是世界模型预测的后果，\(C\) 负责评价后果。机器不必把每种动作都在现实中试一遍，而是可以先在内部预演，再选择行动。

::: tip 预测为什么关键
没有这类预测，机器只能依靠已经学会的反应，或行动之后再看结果；有了预测，它才能在行动之前比较不同未来。**在这套架构中，规划就是利用预测来选择行动。**
:::

这里的“预测”不只意味着预测下一帧。按照这套架构，世界模型有两项任务：

1. 根据已经看到的内容，补全当前状态中缺失的空间或时间信息；
2. 根据自然变化或候选动作，预测一个或多个可能的未来状态。

因此，预测是世界模型服务于推理和规划的接口。但新的问题也随之出现：真实观测包含海量信息，世界模型究竟应该预测其中的什么？

### 预测观测，还是预测表示？

以驾驶为例。周围车辆会怎样运动，直接影响下一步操作；路边每片树叶会怎样摆动，通常与驾驶无关，也很难被精确确定。如果要求模型生成下一帧的全部像素，两类信息都会进入同一个预测任务。

这并不说明生成式预测没有价值。需要生成画面或进行精细仿真时，观测细节本来就是目标。JEPA 做的是另一种取舍：**不直接预测完整观测，而是预测由编码器提取的目标表示。**

| 预测目标     | 模型要回答什么                   | 主要取舍                                       |
| ------------ | -------------------------------- | ---------------------------------------------- |
| **完整观测** | 目标画面或信号具体是什么样       | 保留并生成细节，也要承担全部观测的信息量       |
| **抽象表示** | 从当前信息可以推断出哪些目标状态 | 可以省去不必重建的细节，但表示是否有用仍需检验 |

::: warning 不可预测，不等于不重要
另一辆车是否突然转向同样难以确定，却绝不能被忽略。可以抽象掉的是与任务无关的偶然细节；与行动有关的不确定性仍要被表达。LeCun 的原始蓝图为此引入潜变量 \(z\)，用来表示当前信息无法确定的多种可能。后来的具体 JEPA 实现并不都采用同一种做法。
:::

JEPA 由此出现：它仍然通过预测来学习，但把目标从观测本身移到了表示空间。它希望得到的不是一幅精确的未来图像，而是一种**有信息、可预测，并可能服务于后续规划的内部状态**。下一节将进一步拆开这个过程，说明 Joint Embedding Predictive Architecture 这个名字究竟对应哪些计算。

## 5.1.2　JEPA 的核心：在表示空间中预测

JEPA 的核心计算关系可以概括为：给定一部分信息，JEPA 先把它编码成 **context 表示**，再用 Predictor 预测另一部分信息的 **target 表示**。

### JEPA 架构示意图

![JEA、生成式架构与 JEPA 的结构对比](/jepa/jepa-architecture-comparison.png)

![JEA、生成式架构与 JEPA 的纵向结构对比](/jepa/jepa-architecture-comparison-mobile.png)

_图 5.2　三类自监督学习架构的对比。JEA 直接比较两侧表示；生成式架构预测原始目标 $y$；JEPA 预测目标的表示 $s_y$。来源：[I-JEPA 论文 Figure 2](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf)。_

JEPA 这个名字描述的正是图中最右侧的计算关系：

| 名称                | 含义                                                                                |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Joint Embedding** | $x$ 和 $y$ 分别经过编码器得到表示；模型在表示空间中学习二者的依赖关系               |
| **Predictive**      | Predictor 根据 $x$ 的表示生成对 $y$ 表示的预测，而不是直接令两侧表示相等            |
| **Architecture**    | 它规定的是一种学习结构，不限定输入模态、网络类型或 context 与 target 的具体构造方式 |

用最小的一组式子表示，就是：

$$
s_x = E_x(x), \qquad s_y = E_y(y)
$$

$$
\hat{s}_y = P(s_x,c,z), \qquad \mathcal{L}_{\mathrm{pred}} = D(\hat{s}_y,s_y)
$$

其中，$x$ 是已经给出的 context，$y$ 是要预测的 target。两个编码器 $E_x$ 和 $E_y$ 不必具有相同结构，也不必共享参数；真正需要与 $s_y$ 比较的是 Predictor 的输出 $\hat{s}_y$，而不是 $s_x$ 本身。

$c$ 表示模型已经知道的条件，例如 target 的位置、预测的时间间隔或计划执行的动作；$z$ 是可选的潜变量，用来表示 target 中存在、却无法由 context 确定的信息。这里把二者分开书写，是为了避免把“已知条件”和“未知因素”混为一谈。LeCun 在 2022 年给出的通用 JEPA 使用的是更简洁的写法 $P(s_x,z)$。

### 从 context 到 target：JEPA 如何完成一次预测

上面的两行公式，可以拆成一次训练中的四个步骤。

1. **构造预测关系。** 从数据中取出 context $x$ 和 target $y$。它们可以是同一幅图像的不同区域、同一段视频的不同片段，也可以是当前状态与未来状态。若 target 的位置、时间间隔或计划执行的动作已经给定，就把它们作为条件 $c$。
2. **分别编码两端。** Context Encoder 将已知信息编码为 $s_x$；Target Encoder 将真实 target 编码为 $s_y$。$s_y$ 是训练时的参照，Predictor 并不会直接看到原始的 $y$。
3. **预测 target 表示。** Predictor 根据 $s_x$、已知条件 $c$ 和可选的潜变量 $z$，产生预测 $\hat{s}_y$。它输出的是 target 的表示，而不是 target 本身。
4. **在表示空间中比较。** 距离 $D(\hat{s}_y,s_y)$ 衡量预测与参照是否一致，并成为训练信号。至于两个编码器具体怎样更新、怎样避免得到无信息的表示，将在 5.2 中展开。

这里最关键的区别是：**预测误差比较的是 $\hat{s}_y$ 与 $s_y$，不是 $\hat{y}$ 与 $y$。** 因此，原始 target 中的一项差异是否需要被预测，取决于它是否改变了 $s_y$。

<div class="jepa-outcome-grid">
  <section class="jepa-outcome-card is-predictable">
    <span class="jepa-outcome-label">能够确定</span>
    <strong>保留并预测</strong>
    <p>如果一种变化被保留在 <code>s_y</code> 中，并且能够由 <code>s_x</code> 和已知条件 <code>c</code> 推出，Predictor 就需要学会这种关系。</p>
  </section>
  <section class="jepa-outcome-card is-ignored">
    <span class="jepa-outcome-label">不必区分</span>
    <strong>在表示中合并</strong>
    <p>如果不同的 target 被编码为相同或相近的 <code>s_y</code>，它们之间的差异便不会进入预测误差，也不需要被恢复。</p>
  </section>
  <section class="jepa-outcome-card is-uncertain">
    <span class="jepa-outcome-label">重要但未知</span>
    <strong>保留多种可能</strong>
    <p>如果一种差异需要保留，却无法由 context 确定，通用 JEPA 可以用不同的 <code>z</code> 表达多个可能的 target 表示。</p>
  </section>
</div>

仍以车辆驶近岔路口为例：车辆的位置和速度可以由已有信息推断，属于第一种情况；路面纹理或树叶的具体摆动可以不进入目标表示，属于第二种情况；另一辆车最终左转还是右转无法事先确定，却可能直接影响行动，属于第三种情况。如果转向已经由智能体选定，它就是条件 $c$；如果它仍是未知因素，才需要由 $z$ 表达。

[LeCun 2022 年的通用 JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) 正是通过两种机制处理“一份 context 对应多个 target”的情况：不需要区分的 target 可以通过编码器的不变性得到相同表示；仍需区分的可能结果，则可以通过改变 $z$ 得到不同的预测表示。这是一种通用设计，[I-JEPA](https://arxiv.org/abs/2301.08243) 和 [V-JEPA](https://arxiv.org/abs/2404.08471) 等具体实现并没有完整采用其中显式搜索多个结果的潜变量机制。

训练时，Target Encoder 用真实的 $y$ 提供参照表示 $s_y$。训练完成后保留哪些模块，则取决于用途：表征学习通常使用编码器；预测未来状态还需要 Context Encoder 与 Predictor。无论哪种用途，JEPA 的 Predictor 都不会因此自动变成图像或视频生成器。

::: warning 不可预测，不等于不重要
JEPA 提供了忽略差异和表达不确定性的机制，却不会自动知道二者的正确边界。预测任务、数据和训练约束都会影响表示最终保留什么；如果所有输入都被编码成同一个常量，预测误差甚至也可以很小。怎样让表示既有信息又可预测，就是 5.2 要处理的**表示坍缩**问题。
:::

至此，JEPA 的计算过程可以概括为：**构造 context 与 target，分别编码，在表示空间中完成预测，再用目标表示提供训练信号。** 下一节将继续讨论，这套计算最终希望形成怎样的内部状态。

## 5.1.3　从预测表征到行动规划：JEPA 研究现状

JEPA 的基本计算关系并没有改变。真正发生变化的是**预测问题**：target 从静态图像中的区域扩展到视频中的时空状态，动作随后成为 Predictor 的条件，预测结果才开始被用于规划。

### 时间线：从 JEPA 蓝图到行动规划

<div class="jepa-history" role="list" aria-label="JEPA 重要工作时间线">
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2022</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>JEPA 被提出为世界模型的研究蓝图</strong>
        <span class="jepa-history-tag is-blueprint">LeCun</span>
      </div>
      <p><a href="https://openreview.net/forum?id=BZ5a1r-kVsf">A Path Towards Autonomous Machine Intelligence</a> 提出：世界模型应预测抽象表示，而不是还原观测中的全部细节；H-JEPA 则进一步设想在多个抽象层级和时间尺度上进行预测。</p>
      <p class="jepa-history-boundary"><b>边界：</b>这是一份研究蓝图，不是已经完成的系统。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2023</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>先在静态图像上验证表示预测</strong>
        <span class="jepa-history-tag is-representation">I-JEPA</span>
      </div>
      <p><code>可见区域 → 被遮挡区域的表示</code>。<a href="https://arxiv.org/abs/2301.08243">I-JEPA</a> 给出了可扩展的图像实验，说明这种目标可以学到有用的静态视觉表征。</p>
      <p class="jepa-history-boundary"><b>边界：</b>它不包含时间和动作，也不描述环境如何变化。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2024</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>从静态场景走向视频中的时间变化</strong>
        <span class="jepa-history-tag is-video">V-JEPA</span>
      </div>
      <p><code>可见时空区域 → 被遮挡时空区域的表示</code>。<a href="https://arxiv.org/abs/2404.08471">V-JEPA</a> 将预测扩展到视频，让模型从无动作标签的视频中学习时间变化。</p>
      <p class="jepa-history-boundary"><b>边界：</b>看到画面如何变化，不等于能够区分不同动作造成的后果。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2025</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>从大规模视频预训练走向动作与规划</strong>
        <span class="jepa-history-tag is-action">V-JEPA 2 / 2-AC</span>
      </div>
      <p><a href="https://arxiv.org/abs/2506.09985">V-JEPA 2</a> 扩大了视频预训练的模型与数据规模；随后，V-JEPA 2-AC 冻结视频 Encoder，用机器人轨迹训练动作条件 Predictor，并将预测结果接入 MPC 选择动作。</p>
      <p class="jepa-history-boundary"><b>边界：</b>基础 V-JEPA 2 不接收动作；动作预测来自后训练的 2-AC。现有规划仍依赖外部目标、代价和 Planner，主要证据来自受限任务与较短范围。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2025–26</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>主线走到行动规划，研究图景转向多线并行</strong>
        <span class="jepa-history-tag is-frontier">多线并行</span>
      </div>
      <p>从 2025 年末的 <a href="https://arxiv.org/abs/2511.08544">LeJEPA</a>，到 2026 年的 <a href="https://arxiv.org/abs/2603.14482">V-JEPA 2.1</a>、<a href="https://arxiv.org/abs/2603.19312">LeWorldModel</a> 与 <a href="https://arxiv.org/abs/2604.03208">HWM</a>，不同工作开始分别处理训练稳定性、状态表征、端到端动力学和长程规划等瓶颈。</p>
      <p class="jepa-history-boundary"><b>边界：</b>这些工作不是依次替代的“下一代 JEPA”，而是下面几条并行研究线的代表。</p>
    </div>
  </article>
</div>

_时间线回答的是“JEPA 如何从表示学习走到受限规划”；到 2026 年，还需要横向观察不同工作正在补哪一个瓶颈。_

### 截至 2026：五条前沿路线并行推进

截至 2026 年，JEPA 的路线经过发展已经开始形成几条平行的研究线，包括：状态能否稳定学出？模型应该怎样组织？Predictor 能否描述动作造成的未来？表示是否适合行动？Planner 又怎样利用预测想得更远？

此处只做概述，后续章节将会详细展开。

<div class="jepa-frontier-grid" role="list" aria-label="2026 年 JEPA 的五条并行研究线">
  <article class="jepa-frontier-card is-training" role="listitem">
    <span class="jepa-frontier-index">01</span>
    <strong>防塌缩：怎样学出有信息的状态？</strong>
    <p>如果所有输入都被编码成同一个表示，预测误差也可能很小。I-JEPA 与 V-JEPA 采用缓慢更新的 Target Encoder、停止梯度和 Predictor 等非对称机制稳定训练；NeurIPS 2024 的<a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/04a80267ad46fc730011f8760f265054-Abstract-Conference.html">防塌缩 C-JEPA</a>和 <a href="https://arxiv.org/abs/2511.08544">LeJEPA</a> 则开始用显式正则直接约束表示的方差与分布。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>不塌缩只是最低条件。表示彼此不同，不等于它保留了正确的语义、动力学和动作信息。</p>
  </article>
  <article class="jepa-frontier-card is-architecture" role="listitem">
    <span class="jepa-frontier-index">02</span>
    <strong>架构：Encoder 与 Predictor 应该怎样训练？</strong>
    <p>V-JEPA 2-AC 冻结大规模视频 Encoder，只训练动作条件 Predictor；<a href="https://arxiv.org/abs/2603.19312">LeWorldModel</a> 则让 Encoder 与 Predictor 端到端一起学习。前者复用已经学到的视觉表征，后者允许内部状态与环境动力学共同适配。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>两条路线尚无统一胜者。通用视觉能力、任务适应性、训练稳定性、数据效率与计算成本仍需取舍。</p>
  </article>
  <article class="jepa-frontier-card is-dynamics" role="listitem">
    <span class="jepa-frontier-index">03</span>
    <strong>动力学：怎样预测动作造成的未来？</strong>
    <p>视频只能告诉模型世界发生了变化，动作条件模型还要分辨变化由什么行为造成。一些工作利用少量交互轨迹学习状态转移，另一些工作尝试从无动作标签视频中<a href="https://arxiv.org/abs/2601.05230">发现潜在动作</a>。研究也从一步预测转向多步滚动；当同一状态对应多个合理未来时，在常见的均方误差目标下，确定性 Predictor 往往只能给出平均结果。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>动作数据稀缺，多步预测会累积误差；怎样稳定表达随机、分叉的未来，仍处在很早期的探索阶段。</p>
  </article>
  <article class="jepa-frontier-card is-interface" role="listitem">
    <span class="jepa-frontier-index">04</span>
    <strong>表示：怎样让内部状态真正适合行动？</strong>
    <p>精细控制需要保留位置、边界等局部信息，<a href="https://arxiv.org/abs/2603.14482">V-JEPA 2.1</a> 因此加强了稠密预测与深层监督。但信息更丰富还不够：2026 年 TMLR 接收的一项<a href="https://openreview.net/forum?id=cHZn5Gdh8e">系统研究</a>表明，更好的多步预测并不会自动带来更高的规划成功率；<a href="https://arxiv.org/abs/2607.25337">Temporal-Distance JEPA</a> 等早期预印本开始进一步学习可达性或时间距离。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>状态有信息、预测得准确与适合规划是三层不同的要求。目前还没有一种表示空间结构或代价函数能够适用于所有任务。</p>
  </article>
  <article class="jepa-frontier-card is-planning" role="listitem">
    <span class="jepa-frontier-index">05</span>
    <strong>Planner：怎样把短期预测变成长程行动？</strong>
    <p>Predictor 只回答“执行这些动作可能发生什么”，Planner 才负责提出并选择动作。短期 MPC 通常直接搜索动作序列；<a href="https://arxiv.org/abs/2604.03208">HWM</a> 尝试先在较长时间尺度上寻找潜在子目标，再用短期模型寻找具体动作；<a href="https://arxiv.org/abs/2606.09311">FF-JEPA</a> 则初步探索由模型直接预测潜在子目标。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>HWM 仍使用给定的最终目标和外部搜索；FF-JEPA 尝试去掉推理时的显式目标图像，但目前只有 Push-T 上的初步结果。开放世界中的长期自主规划还没有实现。</p>
  </article>
</div>

::: info 更多任务与预测方式
除上述五条路线外，还有一批工作在探索 JEPA 能用于什么任务，又可以怎样构造预测目标。例如，[VL-JEPA](https://arxiv.org/abs/2512.10942) 将这种方法扩展到视觉—语言任务，[对象级 C-JEPA](https://arxiv.org/abs/2602.11389) 在对象层级建模关系，[UniJEPA](https://arxiv.org/abs/2608.07409) 则尝试统一图像与视频表示；音频、点云等模态中也出现了类似探索。这些工作未必属于同一条发展主线，但都在扩展 JEPA 的问题边界：context 与 target 可以怎样定义，模型可以预测什么，以及这种预测能够支持哪些任务。
:::

_截至 2026 年 8 月，LeJEPA、LeWorldModel、V-JEPA 2.1、HWM、FF-JEPA 和 Temporal-Distance JEPA 等多项代表工作仍是预印本。它们展示的是正在形成的研究方向，而不是已经公认的最终架构。_

## 主要参考资料

- [LeCun, 2022, A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)——JEPA、H-JEPA 与完整自主智能架构的原始蓝图。
- [I-JEPA](https://arxiv.org/abs/2301.08243)、[IWM](https://arxiv.org/abs/2403.00504)、[V-JEPA](https://arxiv.org/abs/2404.08471)、[V-JEPA 2](https://arxiv.org/abs/2506.09985) 与 [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)——核对图像、条件预测、视频、动作条件预测和稠密局部表征的发展。
- [NeurIPS 2024 防塌缩 C-JEPA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/04a80267ad46fc730011f8760f265054-Abstract-Conference.html)、[LeJEPA](https://arxiv.org/abs/2511.08544)、[LeWorldModel](https://arxiv.org/abs/2603.19312) 与 [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)——核对显式防塌缩、端到端世界模型及其理论边界。
- [Learning Latent Action World Models in the Wild](https://arxiv.org/abs/2601.05230)、[What Drives Success in Physical Planning with JEPA World Models?](https://openreview.net/forum?id=cHZn5Gdh8e)、[Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337)、[HWM](https://arxiv.org/abs/2604.03208) 与 [FF-JEPA](https://arxiv.org/abs/2606.09311)——核对潜在动作、规划接口与长程规划的早期探索。
- [对象级 C-JEPA](https://arxiv.org/abs/2602.11389)、[VL-JEPA](https://arxiv.org/abs/2512.10942) 与 [UniJEPA](https://arxiv.org/abs/2608.07409)——核对对象关系、视觉语言和统一建模等横向扩展。
- [Stanford CS25：Joint Embedding Predictive World Models](https://web.stanford.edu/class/cs25/)（[slides](https://drive.google.com/file/d/1bF5Yfzf-FG5iNIAgsXn2DwVD3l3ymvZW/view)）——采用“状态表征—世界变化—动作后果”的教学骨架。
- [LeCun, Les Houches 2022, Lecture 3](https://leshouches2022.github.io/SLIDES/lecun-20220720-leshouches-03.pdf)——核对 JEPA、latent variable、H-JEPA 的原始概念关系。
- [LeCun & Manyika, 2026, Learning Abstractions](https://www.amacad.org/publication/daedalus/learning-abstractions-conversation-yann-lecun)——核对“预测充分的抽象表示”与多层、长时间尺度预测的近期表述。
- [LeCun, NUS 2025](https://drive.google.com/file/d/1j3A8SYtkJqui7CotCMuvES2LpR17JYrQ/view)——参考“寻找可预测状态变量”到动作条件世界模型的宏观过渡。
- [LeCun, Harvard 2024](https://cmsa.fas.harvard.edu/media/lecun-20240328-harvard_reduced.pdf)——核对 JEPA、世界模型与完整 Objective-Driven AI 架构的边界。
- [I-JEPA, CVPR 2023 slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/21019.pdf)——核对 I-JEPA 的最小训练结构与实验设置。
- [NYU Embodied Learning and Vision：World Models and Planning](https://elvcourse.org/course-public/2026-spring/lectures/week05_wm_planning.pdf)——用独立课程视角校正 JEPA 在整个世界模型谱系中的位置。

[回到第 5 章](./index.md)
