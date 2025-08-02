# Chapter 24: 多阶段排序系统

搜索引擎需要在毫秒级延迟内从数十亿文档中找出最相关的结果。单一排序模型难以兼顾效率与精度，因此现代搜索系统普遍采用多阶段级联架构。本章深入探讨如何设计一个高效的多阶段排序系统，包括各阶段的职责划分、特征传递机制、算力分配策略以及模型压缩技术。我们将学习如何在有限的计算资源下最大化排序质量，以及如何通过知识蒸馏等技术优化各阶段的模型。

## 24.1 召回-粗排-精排-重排的级联架构

多阶段排序系统的核心思想是逐步缩小候选集，在每个阶段使用适合的算法复杂度。这种漏斗式架构让我们能够在严格的延迟约束下，对海量文档进行高质量排序。典型的四阶段架构包括：

### 24.1.1 各阶段的职责与设计原则

**召回阶段（Recall）**：
- 目标：从全量索引中快速筛选出数千到数万个候选文档
- 方法：倒排索引、向量检索、布尔查询等轻量级方法
- 设计原则：高召回率优于精确率，避免遗漏相关文档
- 典型延迟：10-50ms
- 关键指标：
  - 召回率 @ k（通常要求 > 90%）
  - 候选集多样性（避免同质化）
  - 查询词覆盖率

**多路召回融合**：
现代系统通常采用多路召回策略，每路负责不同的相关性信号：
- 词匹配召回：基于倒排索引的精确和模糊匹配
- 语义召回：基于向量相似度的语义匹配
- 个性化召回：基于用户历史的协同过滤
- 热门召回：基于全局流行度的候选

融合策略需要考虑：
- 去重机制：相同文档的不同召回路径
- 分数归一化：不同召回源的分数尺度对齐
- 配额分配：每路召回的文档数量上限

**粗排阶段（Coarse Ranking）**：
- 目标：将候选集缩减至数百个文档
- 方法：简单的线性模型或浅层神经网络
- 设计原则：使用少量高效特征，如 BM25、静态质量分等
- 典型延迟：5-20ms
- 架构选择：
  - 线性模型：logistic regression，计算效率最高
  - 浅层网络：2-3 层 MLP，表达能力更强
  - 树模型：GBDT/LightGBM，特征交互能力强
  - 双塔模型：预计算文档向量，在线只需点积

**精排阶段（Fine Ranking）**：
- 目标：产生高质量的 Top-K 结果（通常 K=10-100）
- 方法：复杂的深度学习模型，如 BERT、T5 等
- 设计原则：使用丰富的语义特征，追求排序精度
- 典型延迟：20-100ms
- 模型演进：
  - 早期：Learning to Rank (LTR) 模型，如 LambdaMART
  - 中期：深度匹配模型，如 DRMM、KNRM
  - 现代：预训练语言模型，如 BERT、T5
  - 前沿：生成式排序模型，如 RankGPT

**重排阶段（Re-ranking）**：
- 目标：考虑多样性、新鲜度、个性化等业务目标
- 方法：基于规则或轻量级模型的后处理
- 设计原则：平衡相关性与其他目标，优化用户体验
- 典型延迟：1-5ms
- 优化维度：
  - 结果多样性：MMR (Maximal Marginal Relevance) 算法
  - 时效性平衡：boost 最新内容
  - 站点分散：避免单一来源垄断
  - 个性化调整：基于用户画像微调

### 24.1.2 阶段间的数据流与候选集传递

在 OCaml 中，我们可以定义阶段间的接口：

```ocaml
module type RankingStage = sig
  type input_doc
  type output_doc
  type feature_set
  
  val process : 
    input_doc list -> 
    query:Query.t ->
    budget:Latency.t ->
    (output_doc list * feature_set)
    
  val pass_features : feature_set -> Feature.collection
end
```

关键设计考虑：
- **候选集大小递减**：召回 10K → 粗排 1K → 精排 100 → 重排 10
- **特征逐步丰富**：每个阶段可以继承并增强前序阶段的特征
- **元数据传递**：文档 ID、初步分数、计算的中间结果等
- **批处理优化**：合并多个查询的候选集进行批量计算

**数据流设计模式**：

1. **管道模式（Pipeline）**：
   - 每个阶段顺序执行
   - 前一阶段的输出是后一阶段的输入
   - 适合延迟敏感的在线服务

2. **异步模式（Async）**：
   - 精排可以异步处理部分结果
   - 先返回粗排结果，精排完成后更新
   - 适合对延迟要求极高的场景

3. **并行模式（Parallel）**：
   - 多路召回并行执行
   - 不同特征提取并行计算
   - 需要最终的同步点

**候选集管理策略**：

```ocaml
module CandidateSet = struct
  type t = {
    docs : Document.t array;
    scores : float array;
    features : Feature.t list;
    stage_metadata : (string * string) list;
  }
  
  val truncate : t -> int -> t
  val merge : t list -> t
  val sort_by_score : t -> t
end
```

关键优化点：
- 使用数组而非列表，提升缓存局部性
- 分数和文档分离存储，便于 SIMD 优化
- 延迟特征计算，按需加载
- 支持增量式特征更新

### 24.1.3 降级策略与容错设计

系统必须处理各种异常情况：

**超时降级**：
- 粗排超时 → 直接使用召回结果
- 精排超时 → 使用粗排结果
- 设置每阶段的 SLA (Service Level Agreement)

```ocaml
module DegradationPolicy = struct
  type action = 
    | SkipStage of stage_name
    | ReduceCandidates of int
    | UseBackupModel of model_id
    | ReturnPartialResults
    
  type trigger = 
    | LatencyExceeded of Time.span
    | ErrorRate of float
    | LoadThreshold of float
    
  val decide : trigger -> context -> action
end
```

**容量降级**：
- 动态调整各阶段的候选集大小
- 高负载时减少精排文档数量
- 使用断路器模式防止级联故障

降级决策矩阵：
| 系统负载 | 召回数量 | 粗排数量 | 精排数量 | 模型选择 |
|---------|---------|---------|---------|----------|
| 低 (<30%) | 10000 | 1000 | 100 | 完整模型 |
| 中 (30-70%) | 5000 | 500 | 50 | 完整模型 |
| 高 (70-90%) | 2000 | 200 | 20 | 轻量模型 |
| 过载 (>90%) | 1000 | 100 | 10 | 降级模型 |

**模型降级**：
- 维护多个版本的模型（轻量级备份）
- A/B 测试新模型时的回滚机制
- 特征缺失时的 fallback 策略

```ocaml
module ModelRegistry = struct
  type model_variant = 
    | Primary of { version: string; size: int }
    | Lightweight of { speedup: float; quality_loss: float }
    | Fallback of { min_features: string list }
    
  val select_model : 
    latency_budget:Time.span -> 
    available_features:string list ->
    model_variant
end
```

**容错机制设计**：

1. **断路器模式**：
   - 错误计数超过阈值时自动熔断
   - 定期尝试恢复（半开状态）
   - 支持手动强制开关

2. **重试策略**：
   - 指数退避：1ms → 2ms → 4ms
   - 最大重试次数：通常 2-3 次
   - 仅对幂等操作重试

3. **隔离策略**：
   - 线程池隔离：每个阶段独立线程池
   - 请求隔离：VIP 请求独立通道
   - 资源隔离：缓存、内存分区管理

4. **监控告警**：
   - 实时指标：延迟 P50/P99、错误率
   - 降级事件：记录触发原因和影响
   - 自动扩缩容：基于负载预测

## 24.2 特征工程与跨阶段特征传递

### 24.2.1 轻量级特征 vs. 重量级特征的分配

特征按计算成本可分为：

**静态特征**（预计算）：
- PageRank、文档质量分
- 文档长度、更新时间
- 站点权威度
- 适用阶段：召回、粗排
- 存储策略：
  - 直接存储在倒排索引的 payload 中
  - 使用定点数压缩（如 uint8 映射到 0-1）
  - 更新频率：每日批量更新

**查询相关特征**（实时计算）：
- BM25、TF-IDF 分数
- 查询词覆盖率
- 短语匹配、近似度
- 适用阶段：粗排为主
- 优化技巧：
  - 使用倒排索引的统计信息加速计算
  - 预计算查询无关部分（如文档长度归一化因子）
  - SIMD 指令并行计算多个文档

**深度语义特征**（计算密集）：
- BERT embedding 相似度
- 跨注意力分数
- 上下文相关性
- 适用阶段：精排
- 计算优化：
  - 批量推理减少开销
  - 模型量化（FP16/INT8）
  - 使用 TensorRT 等推理加速库
  - 文档端 embedding 预计算

**交互特征**（用户相关）：
- 点击历史、停留时间
- 个性化偏好
- 会话上下文
- 适用阶段：精排、重排
- 实现考虑：
  - 用户特征服务独立部署
  - 会话级缓存减少查询
  - 隐私保护和数据脱敏

**特征成本分析表**：

| 特征类型 | 计算成本 | 存储成本 | 更新频率 | 适用阶段 |
|---------|---------|---------|---------|----------|
| 静态质量分 | O(1) | 4B/doc | 每日 | 全阶段 |
| BM25 | O(k) | 0 | 实时 | 召回/粗排 |
| 词向量相似度 | O(d) | 4d B/doc | 离线 | 粗排 |
| BERT 分数 | O(n²) | 0 | 实时 | 精排 |
| 用户 CTR | O(1) | 可变 | 准实时 | 精排/重排 |

其中 k=查询词数，d=向量维度，n=序列长度

**特征分配原则**：
1. **计算/存储权衡**：高频特征倾向预计算，低频特征实时计算
2. **精度递增**：早期阶段使用粗粒度特征，后期使用细粒度特征
3. **个性化递增**：早期通用特征，后期个性化特征
4. **复用最大化**：设计可跨阶段复用的特征

### 24.2.2 特征缓存与复用策略

```ocaml
module FeatureCache = struct
  type cache_key = {
    doc_id : Document.id;
    query_hash : int;
    feature_version : int;
  }
  
  type cache_policy = 
    | TTL of Time.span
    | LRU of int
    | Adaptive of (unit -> policy)
    
  type cache_stats = {
    hit_rate : float;
    avg_latency : Time.span;
    memory_usage : int;
  }
end
```

缓存策略：
- **文档级缓存**：静态特征长期缓存
- **查询级缓存**：热门查询的特征结果
- **会话级缓存**：用户会话内的特征复用
- **分布式缓存**：跨机器共享计算结果

**多级缓存架构**：

```ocaml
module HierarchicalCache = struct
  type level = 
    | ProcessLocal    (* 最快，容量小 *)
    | MachineLocal    (* 快，容量中等 *)
    | Distributed     (* 慢，容量大 *)
    
  type lookup_result = 
    | Hit of (value * level)
    | Miss
    
  val lookup : key -> lookup_result
  val populate : key -> value -> level -> unit
end
```

**缓存策略优化**：

1. **热度感知**：
   - 统计查询频率，优先缓存高频特征
   - 使用 Count-Min Sketch 估计频率
   - 自适应调整不同特征的 TTL

2. **预取机制**：
   - 基于查询 pattern 预测可能需要的特征
   - 在空闲时预热常用特征
   - 会话开始时批量加载用户特征

3. **压缩存储**：
   - 特征量化：float32 → int8
   - 稀疏表示：只存储非零值
   - Delta 编码：存储与基准值的差异

4. **失效策略**：
   ```ocaml
   module Invalidation = struct
     type strategy = 
       | Immediate      (* 立即失效 *)
       | Eventual      (* 最终一致性 *)
       | Versioned     (* 版本控制 *)
       
     val invalidate : cache_key -> strategy -> unit
   end
   ```

**复用模式设计**：

1. **特征继承链**：
   ```
   召回特征 → 粗排继承 + 扩展 → 精排继承 + 深化
   ```

2. **特征池化**：
   - 维护全局特征池
   - 各阶段按需提取
   - 避免重复计算

3. **增量计算**：
   - 基于已有特征计算新特征
   - 如：BM25 基础上计算 BM25F
   - 向量运算的部分结果复用

### 24.2.3 跨阶段特征增强机制

特征在各阶段间的流动和增强：

1. **特征继承**：
   - 召回阶段的倒排索引位置 → 粗排的位置特征
   - 粗排的 BM25 分数 → 精排的基础相关性信号

2. **特征组合**：
   - 多路召回的分数融合
   - 跨阶段分数的非线性组合

3. **特征衍生**：
   - 从粗排分数分布推导文档区分度
   - 从精排预测结果生成置信度特征

**特征传递协议**：

```ocaml
module FeaturePassthrough = struct
  type feature_bundle = {
    stage_scores : (string * float) list;
    raw_features : Feature.collection;
    derived_features : Feature.collection;
    computation_time : (string * Time.span) list;
  }
  
  val encode : feature_bundle -> bytes
  val decode : bytes -> feature_bundle
  val merge : feature_bundle list -> feature_bundle
end
```

**增强机制设计**：

1. **统计特征增强**：
   ```ocaml
   module StatisticalEnhancement = struct
     (* 基于候选集分布的特征 *)
     type distribution_features = {
       score_mean : float;
       score_std : float;
       rank_percentile : float;
       z_score : float;
     }
     
     val compute : scores:float array -> int -> distribution_features
   end
   ```

2. **交叉特征生成**：
   - 召回方式 × 查询类型
   - 静态质量 × 查询相关性
   - 时效性 × 用户偏好

3. **元特征（Meta-features）**：
   - 各阶段耗时
   - 特征覆盖率
   - 模型置信度
   - 候选集大小

**特征工程最佳实践**：

1. **特征标准化**：
   ```ocaml
   module Normalization = struct
     type method_ = 
       | MinMax of { min: float; max: float }
       | ZScore of { mean: float; std: float }
       | Quantile of { percentiles: float array }
       | Log1p  (* log(1 + x) for long-tail features *)
   end
   ```

2. **特征选择**：
   - 基于互信息的特征重要性
   - 前向/后向特征选择
   - L1 正则化自动选择
   - 在线特征价值评估

3. **特征监控**：
   - 特征分布偏移检测
   - 缺失率告警
   - 计算延迟跟踪
   - A/B 测试新特征

**跨阶段协同优化**：

```ocaml
module CrossStageOptimization = struct
  type feedback = {
    stage : string;
    quality_delta : float;
    latency_delta : Time.span;
  }
  
  (* 基于下游反馈调整上游策略 *)
  val adjust_upstream : 
    feedback list -> 
    current_config -> 
    optimized_config
end
```

关键技术：
- 反向传播梯度到早期阶段
- 联合训练多阶段模型
- 端到端优化目标函数
- 知识蒸馏改进早期阶段

## 24.3 动态算力分配与延迟预算管理

### 24.3.1 延迟预算的分解与监控

```ocaml
module LatencyBudget = struct
  type budget = {
    total : Time.span;
    recall : Time.span;
    coarse : Time.span; 
    fine : Time.span;
    rerank : Time.span;
    buffer : Time.span;
  }
  
  type monitor = {
    track : stage -> Time.span -> unit;
    remaining : unit -> Time.span;
    exceeded : unit -> bool;
  }
end
```

预算分配原则：
- **前紧后松**：为后续阶段预留缓冲
- **动态调整**：根据实际执行情况调整
- **优先级**：核心阶段（精排）优先保障

### 24.3.2 自适应候选集大小调整

根据查询复杂度和系统负载动态调整：

```ocaml
module AdaptiveRanking = struct
  type strategy = 
    | Fixed of int
    | LinearScaling of { base: int; factor: float }
    | LoadBased of { 
        low: int; 
        normal: int; 
        high: int;
        current_load: unit -> Load.level
      }
    | QueryComplexity of (Query.t -> int)
end
```

调整因素：
- **查询类型**：导航查询 vs. 信息查询
- **结果分布**：高区分度 vs. 低区分度
- **系统负载**：CPU、内存、网络状况
- **用户等级**：VIP 用户获得更多算力

### 24.3.3 尾延迟优化技术

P99 延迟优化策略：

1. **请求冗余**：
   - 同时向多个副本发送请求
   - 使用最快返回的结果

2. **提前终止**：
   - 设置激进的超时
   - 部分结果降级返回

3. **负载均衡**：
   - 避免热点
   - 考虑机器异构性

4. **预热与预取**：
   - 模型预热
   - 热数据预加载

## 24.4 知识蒸馏在轻量级模型中的应用

知识蒸馏是优化多阶段排序系统的关键技术，通过将复杂模型（教师）的知识转移到简单模型（学生），我们可以在早期阶段使用更强大的模型。

### 24.4.1 教师-学生模型架构

```ocaml
module Distillation = struct
  type teacher_model = {
    predict : Document.t -> Query.t -> float;
    get_logits : Document.t -> Query.t -> float array;
    get_attention : Document.t -> Query.t -> Attention.map;
  }
  
  type student_model = {
    architecture : [ `Linear | `ShallowNN | `MobileBERT ];
    params : int; (* 参数量 *)
    latency : Time.span;
  }
  
  type training_config = {
    temperature : float;
    alpha : float; (* 蒸馏损失权重 *)
    hard_label_weight : float;
  }
end
```

蒸馏目标的选择：
- **输出分数蒸馏**：学习教师模型的排序分数
- **排序列表蒸馏**：学习相对顺序而非绝对分数
- **特征蒸馏**：学习中间层表示
- **注意力蒸馏**：学习注意力分布模式

### 24.4.2 蒸馏策略选择

不同阶段的蒸馏策略：

**粗排模型蒸馏**：
- 教师：精排使用的 BERT-Large
- 学生：双线性模型或 2 层 MLP
- 重点：保持排序一致性
- 训练数据：大规模点击日志

**召回模型增强**：
- 教师：粗排模型
- 学生：向量检索模型
- 重点：学习语义相似度
- 方法：对比学习 + 蒸馏

**重排模型压缩**：
- 教师：多目标融合的复杂模型
- 学生：线性组合或决策树
- 重点：业务规则的自动学习

### 24.4.3 在线蒸馏与离线蒸馏的权衡

**离线蒸馏**：
```ocaml
module OfflineDistillation = struct
  type pipeline = {
    collect_data : unit -> Dataset.t;
    run_teacher : Dataset.t -> Predictions.t;
    train_student : Dataset.t -> Predictions.t -> Model.t;
    validate : Model.t -> Metrics.t;
  }
end
```

优点：
- 可以使用最强的教师模型
- 训练过程稳定可控
- 便于大规模并行处理

缺点：
- 数据分布可能过时
- 无法捕捉实时变化
- 需要定期重新训练

**在线蒸馏**：
```ocaml
module OnlineDistillation = struct
  type strategy = 
    | Synchronous of { delay: Time.span }
    | Asynchronous of { buffer_size: int }
    | Streaming of { window: Time.span }
    
  type feedback_loop = {
    collect_predictions : unit -> teacher_pred list;
    update_student : teacher_pred list -> unit;
    swap_model : unit -> unit;
  }
end
```

优点：
- 及时适应分布变化
- 持续学习新模式
- 减少离线训练成本

挑战：
- 系统复杂度增加
- 稳定性保障困难
- 计算资源竞争

### 24.4.4 蒸馏效果评估

评估指标体系：

1. **一致性指标**：
   - Kendall's τ（排序相关性）
   - NDCG@k 相对下降
   - Top-k 重合度

2. **效率指标**：
   - 推理延迟
   - 模型大小
   - 内存占用

3. **业务指标**：
   - 点击率变化
   - 用户满意度
   - 长尾查询表现

## 本章小结

多阶段排序系统是现代搜索引擎的核心架构模式。通过召回、粗排、精排、重排的级联设计，我们可以在毫秒级延迟内处理海量文档。关键要点包括：

1. **级联架构设计**：每个阶段承担不同职责，通过逐步缩小候选集实现效率与质量的平衡
2. **特征工程策略**：根据计算成本合理分配特征，通过缓存和传递机制优化整体性能
3. **动态资源管理**：基于查询复杂度和系统负载自适应调整算力分配，优化尾延迟
4. **知识蒸馏应用**：将复杂模型的知识转移到轻量级模型，提升早期阶段的排序质量

关键公式：
- 候选集缩减比例：$r_i = |C_i| / |C_{i-1}|$，典型值 $r \approx 0.1$
- 延迟预算分解：$T_{total} = \sum_{i} T_i + T_{buffer}$
- 蒸馏损失：$L = \alpha L_{KD} + (1-\alpha) L_{hard}$

## 练习题

### 基础题

**练习 24.1**：设计一个三阶段排序系统，第一阶段从 100 万文档中召回 1000 个，第二阶段选出 100 个，第三阶段产生最终的 10 个结果。如果总延迟预算是 100ms，如何分配各阶段的时间预算？

<details>
<summary>Hint</summary>
考虑各阶段的计算复杂度和候选集大小，召回阶段虽然处理文档最多但使用简单算法，精排阶段处理文档少但模型复杂。
</details>

<details>
<summary>答案</summary>

建议的时间分配：
- 召回阶段：30ms（倒排索引查找为主）
- 粗排阶段：20ms（简单模型，1000 个文档）
- 精排阶段：40ms（复杂模型，100 个文档）
- 缓冲时间：10ms（应对延迟抖动）

关键考虑：
- 召回使用倒排索引，时间复杂度 O(k)，k 为匹配文档数
- 粗排可以批量处理，利用向量化计算
- 精排是主要瓶颈，需要最多时间预算
</details>

**练习 24.2**：在粗排阶段，你有以下特征可选：BM25 分数（1μs/doc）、PageRank（预计算）、BERT 相似度（100μs/doc）、点击率（5μs/doc）。如果要在 20ms 内处理 1000 个文档，应该选择哪些特征？

<details>
<summary>Hint</summary>
计算每个特征的总耗时，考虑特征的区分能力和计算成本的权衡。
</details>

<details>
<summary>答案</summary>

时间预算分析：
- BM25：1μs × 1000 = 1ms ✓
- PageRank：0ms（预计算）✓
- BERT：100μs × 1000 = 100ms ✗（超出预算）
- 点击率：5μs × 1000 = 5ms ✓

应选择：PageRank + BM25 + 点击率，总计 6ms，留有充足余量。BERT 相似度应该留给精排阶段使用。
</details>

**练习 24.3**：实现一个简单的特征缓存系统，支持 TTL 和 LRU 两种淘汰策略。缓存键包含文档 ID 和查询哈希值。

<details>
<summary>Hint</summary>
考虑使用哈希表存储缓存项，LRU 需要维护访问顺序，TTL 需要记录时间戳。
</details>

<details>
<summary>答案</summary>

缓存系统设计要点：
1. 数据结构：HashMap + 双向链表（LRU）或优先队列（TTL）
2. 缓存键：(doc_id, query_hash) 的组合
3. 操作复杂度：get/put 均为 O(1)
4. 并发控制：分片锁或无锁数据结构
5. 监控指标：命中率、延迟、内存使用
</details>

### 挑战题

**练习 24.4**：设计一个自适应的候选集大小调整算法，根据以下信号动态决定各阶段的候选集大小：(1) 查询复杂度（词数、稀有度）(2) 初步结果的分数分布 (3) 系统当前负载。

<details>
<summary>Hint</summary>
可以使用机器学习方法学习查询特征到最优候选集大小的映射，或设计基于规则的启发式算法。
</details>

<details>
<summary>答案</summary>

自适应算法框架：

1. **查询复杂度评分**：
   - 简单查询（1-2 个常见词）：基础大小 × 0.5
   - 复杂查询（长尾词、多词组合）：基础大小 × 2

2. **分数分布分析**：
   - 高区分度（top 分数显著高于其他）：减小候选集
   - 低区分度（分数接近）：增大候选集
   - 使用分数标准差或 Gini 系数量化

3. **负载感知调整**：
   - CPU 使用率 > 80%：所有候选集 × 0.7
   - 延迟 P99 超标：触发降级模式
   - 队列长度监控：动态调整并发度

4. **组合策略**：
   ```
   final_size = base_size × 
     query_factor × 
     distribution_factor × 
     load_factor
   ```
</details>

**练习 24.5**：你负责将一个 BERT-Large 模型（340M 参数）蒸馏到一个适合粗排的轻量级模型。目标是延迟不超过 0.1ms/doc，模型大小不超过 10MB。设计蒸馏方案，包括学生模型架构、训练策略和评估方法。

<details>
<summary>Hint</summary>
考虑使用双塔架构分离查询和文档编码，使用简单的交互函数，可以预计算文档向量。
</details>

<details>
<summary>答案</summary>

蒸馏方案设计：

1. **学生模型架构**：
   - 双塔结构：查询塔 + 文档塔
   - 每塔 2 层 MLP，隐藏层 128 维
   - 最终 64 维向量，点积计算相似度
   - 参数量：约 50K，模型大小 < 1MB

2. **训练策略**：
   - 教师信号：BERT-Large 的 CLS 向量相似度
   - 损失函数：MSE(student_score, teacher_score) + 排序损失
   - 数据增强：查询改写、文档摘要
   - 课程学习：从简单查询到复杂查询

3. **优化技巧**：
   - 量化：INT8 推理
   - 文档向量预计算并建索引
   - SIMD 向量化计算
   - 批量推理减少开销

4. **评估方法**：
   - 离线：Kendall's τ > 0.8
   - 在线：A/B 测试 CTR 下降 < 1%
   - 延迟：P99 < 0.1ms/doc
</details>

**练习 24.6**：设计一个端到端的在线蒸馏系统，能够实时收集精排模型的预测结果，并用于更新粗排模型。考虑数据流设计、模型更新策略、一致性保证等方面。

<details>
<summary>Hint</summary>
考虑使用消息队列解耦数据收集和模型训练，使用版本控制管理模型更新。
</details>

<details>
<summary>答案</summary>

在线蒸馏系统架构：

1. **数据流设计**：
   - 精排服务 → Kafka → 训练集Buffer
   - 采样策略：重要查询（高流量）100%，长尾 10%
   - 数据格式：(query, doc, teacher_score, features)
   - 去重机制：布隆过滤器

2. **模型更新流程**：
   - 增量训练：每小时触发，使用最近 N 小时数据
   - 全量训练：每天一次，防止遗忘
   - 模型验证：留出集 + 在线小流量
   - 版本管理：模型 ID = timestamp + hash

3. **一致性保障**：
   - 特征版本对齐
   - 模型切换原子性（双Buffer）
   - 回滚机制：保留最近 3 个版本
   - 监控告警：效果指标实时跟踪

4. **系统优化**：
   - 训练与服务分离部署
   - GPU 训练，CPU 推理
   - 模型压缩后分发
   - 增量更新减少传输
</details>

**练习 24.7**：某搜索系统发现长尾查询（占 20% 流量）的排序质量较差。设计一个方案，在不显著增加整体延迟的前提下，提升长尾查询的排序质量。

<details>
<summary>Hint</summary>
考虑为长尾查询分配更多计算资源，或使用专门优化的模型。
</details>

<details>
<summary>答案</summary>

长尾查询优化方案：

1. **查询分类**：
   - 基于查询频率的分类器
   - 特征：词频、历史出现次数、查询长度
   - 在线识别，缓存分类结果

2. **差异化处理**：
   - 头部查询：标准流程，激进缓存
   - 长尾查询：增加候选集 2x，使用更强模型
   - 计算预算重分配：头部省下的资源给长尾

3. **专门优化**：
   - 长尾专用召回通道（语义召回权重更高）
   - 针对长尾训练的排序模型
   - 更多上下文特征（会话、用户画像）

4. **成本控制**：
   - 长尾查询结果缓存时间更短
   - 使用异步精排，先返回粗排结果
   - 边缘机房处理常见查询，中心机房处理长尾
</details>

## 常见陷阱与错误

1. **候选集大小设置不当**
   - 错误：各阶段缩减比例不均匀（如 10000→100→50→10）
   - 正确：保持相对均匀的缩减比例（如 10000→1000→100→10）

2. **特征重复计算**
   - 错误：每个阶段独立计算所有特征
   - 正确：建立特征传递机制，避免重复计算

3. **延迟预算过于乐观**
   - 错误：按平均延迟分配预算
   - 正确：考虑 P99 延迟，预留 20-30% 缓冲

4. **忽视降级处理**
   - 错误：超时直接返回空结果
   - 正确：设计优雅降级方案，返回部分结果

5. **蒸馏目标选择不当**
   - 错误：只关注最终分数，忽视排序顺序
   - 正确：使用排序损失，关注相对顺序

6. **模型更新不当**
   - 错误：直接替换线上模型
   - 正确：小流量验证，渐进式切换

## 最佳实践检查清单

### 架构设计
- [ ] 明确定义各阶段的职责和 SLA
- [ ] 设计合理的候选集缩减策略
- [ ] 实现完善的降级和容错机制
- [ ] 支持动态配置调整

### 特征工程
- [ ] 按计算成本对特征分级
- [ ] 实现特征缓存和复用机制
- [ ] 设计跨阶段特征传递协议
- [ ] 监控特征计算耗时

### 性能优化
- [ ] 实施延迟预算管理
- [ ] 优化热点路径
- [ ] 使用批处理提升吞吐
- [ ] 实现请求级别的超时控制

### 模型管理
- [ ] 建立模型版本管理体系
- [ ] 实现 A/B 测试框架
- [ ] 设计模型回滚机制
- [ ] 持续监控模型效果

### 监控运维
- [ ] 监控各阶段延迟分布
- [ ] 跟踪候选集大小变化
- [ ] 告警异常流量模式
- [ ] 记录降级事件
