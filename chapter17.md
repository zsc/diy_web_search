# Chapter 17: 查询理解系统

现代搜索引擎的成功不仅依赖于海量数据的索引能力，更取决于对用户查询意图的深刻理解。随着大语言模型（LLM）的兴起，查询理解系统正在经历一次架构范式的转变——从基于规则和传统 NLP 的管道，演进为以 LLM 为核心的智能理解系统。本章将深入探讨如何设计一个生产级的查询理解系统，涵盖 LLM 服务集成、意图识别、实体抽取到会话状态管理的完整架构。

## 2. LLM 服务的接口设计

### 2.1 服务抽象层设计

在生产环境中集成 LLM 服务，首要任务是设计一个健壮的抽象层，隔离底层模型的具体实现细节。

```ocaml
module type LLM_SERVICE = sig
  type model_id = 
    | GPT4 
    | Claude3 
    | Llama3 
    | Custom of string
    
  type request = {
    prompt: string;
    model: model_id;
    max_tokens: int;
    temperature: float;
    metadata: (string * string) list;
  }
  
  type response = {
    content: string;
    usage: token_usage;
    latency_ms: int;
    model_used: model_id;
  }
  
  and token_usage = {
    prompt_tokens: int;
    completion_tokens: int;
    total_cost: float;
  }
  
  val complete : request -> response Lwt.t
  val complete_with_retry : 
    request -> max_retries:int -> response Lwt.t
  val stream : request -> response Lwt_stream.t
end
```

这种设计允许我们：
- **模型无关性**：通过 `model_id` 类型支持多种模型
- **成本追踪**：`token_usage` 记录每次调用的成本
- **性能监控**：`latency_ms` 用于延迟分析
- **灵活扩展**：`metadata` 支持传递额外参数

**多模型路由策略**

实际系统需要根据查询特征动态选择模型：

```ocaml
module type MODEL_ROUTER = sig
  type routing_strategy =
    | CostOptimized    (* 优先低成本模型 *)
    | QualityFirst     (* 优先高质量模型 *)
    | LatencySensitive (* 优先低延迟模型 *)
    | LoadBalanced     (* 负载均衡 *)
    
  type query_features = {
    estimated_complexity: float;
    user_tier: [ `Free | `Premium | `Enterprise ];
    expected_latency_ms: int option;
    domain: string option;
  }
  
  val select_model : 
    features:query_features -> 
    strategy:routing_strategy -> 
    model_id
    
  val estimate_cost : 
    model:model_id -> 
    prompt_length:int -> 
    float
end
```

路由决策考虑因素：
- **查询复杂度**：简单查询用轻量模型，复杂查询用强模型
- **用户等级**：付费用户获得更好的模型服务
- **延迟要求**：实时场景优先快速模型
- **成本预算**：在质量可接受范围内优化成本

### 2.2 Prompt 管理架构

Prompt 工程是 LLM 应用的核心，需要系统化的管理机制。

```ocaml
module type PROMPT_MANAGER = sig
  type template = {
    id: string;
    version: int;
    content: string;
    variables: string list;
    examples: (string * string) list option;
  }
  
  type rendering_context = {
    query: string;
    user_context: (string * string) list;
    system_context: (string * string) list;
    few_shot_examples: (string * string) list;
  }
  
  val register_template : template -> unit
  val render : 
    template_id:string -> 
    context:rendering_context -> 
    string
    
  val get_active_version : string -> template option
  val rollback_version : string -> int -> unit
end
```

**模板版本控制**

生产环境的 prompt 需要版本管理：
- **A/B 测试**：并行测试不同版本的 prompt
- **渐进发布**：新版本逐步扩大流量
- **快速回滚**：问题版本一键回退

**动态示例选择**

Few-shot 示例的选择对效果影响巨大：

```ocaml
module type EXAMPLE_SELECTOR = sig
  type selection_strategy =
    | Random of int
    | Similarity_based of { 
        embedder: string -> float array;
        top_k: int 
      }
    | Domain_specific of {
        domain_classifier: string -> string;
        examples_per_domain: int
      }
    
  val select_examples : 
    query:string -> 
    strategy:selection_strategy -> 
    example_pool:(string * string) list ->
    (string * string) list
end
```

### 2.3 容错与降级策略

LLM 服务的不稳定性要求完善的容错机制。

```ocaml
module type RESILIENCE_LAYER = sig
  type fallback_strategy =
    | UseCache of { ttl_seconds: int }
    | SwitchModel of model_id
    | DegradeToRule of (string -> string option)
    | ReturnDefault of string
    
  type circuit_breaker_config = {
    failure_threshold: int;
    timeout_ms: int;
    reset_timeout_ms: int;
  }
  
  val with_fallback : 
    primary:(unit -> 'a Lwt.t) ->
    fallback:fallback_strategy ->
    'a Lwt.t
    
  val with_circuit_breaker :
    config:circuit_breaker_config ->
    (unit -> 'a Lwt.t) ->
    'a Lwt.t
end
```

**缓存策略设计**

智能缓存可以显著降低成本和延迟：

```ocaml
module type LLM_CACHE = sig
  type cache_key = {
    prompt_hash: string;
    model_id: model_id;
    temperature: float;
  }
  
  type cache_value = {
    response: string;
    timestamp: float;
    hit_count: int;
  }
  
  val get : cache_key -> cache_value option
  val put : key:cache_key -> value:string -> ttl:int -> unit
  val invalidate_pattern : string -> int (* 返回清除的条目数 *)
  
  (* 语义相似度缓存 *)
  val get_similar : 
    key:cache_key -> 
    threshold:float -> 
    cache_value option
end
```

语义缓存的关键设计：
- **模糊匹配**：相似查询共享缓存结果
- **TTL 管理**：根据查询类型动态调整过期时间
- **预热机制**：高频查询主动刷新缓存

## 3. 意图识别的架构模式

### 3.1 多级意图分类

意图识别需要层次化的分类体系，从粗粒度到细粒度逐步细化。

```ocaml
module type INTENT_CLASSIFIER = sig
  type intent_hierarchy = {
    level1: string;  (* 如: "transactional", "informational", "navigational" *)
    level2: string option;  (* 如: "purchase", "booking", "comparison" *)
    level3: string option;  (* 如: "flight_booking", "hotel_booking" *)
    confidence: float;
  }
  
  type classification_result = {
    primary_intent: intent_hierarchy;
    secondary_intents: intent_hierarchy list;
    ambiguity_score: float;
    explanation: string option;
  }
  
  val classify : 
    query:string -> 
    context:query_context option -> 
    classification_result Lwt.t
end
```

**层次化意图体系设计原则**：
1. **互斥性**：同级意图应该互斥
2. **完备性**：覆盖所有可能的查询类型
3. **可解释性**：每个意图有清晰的定义
4. **可扩展性**：易于添加新的意图类别

**置信度阈值的动态调整**：

```ocaml
module type CONFIDENCE_MANAGER = sig
  type threshold_config = {
    base_threshold: float;
    domain_adjustments: (string * float) list;
    user_history_weight: float;
  }
  
  val compute_threshold : 
    intent:string ->
    user_id:string option ->
    config:threshold_config ->
    float
    
  val should_clarify : 
    result:classification_result ->
    threshold:float ->
    bool
end
```

关键考虑：
- **领域特异性**：医疗查询需要更高置信度
- **用户历史**：根据用户过往行为调整阈值
- **成本权衡**：错误分类的代价不同

### 3.2 Few-shot 学习集成

LLM 的 few-shot 能力让意图识别更加灵活。

```ocaml
module type FEW_SHOT_INTENT = sig
  type example_pool = {
    examples: (string * intent_hierarchy) list;
    embeddings: (string * float array) list Lazy.t;
    update_frequency: [ `Daily | `Weekly | `Realtime ];
  }
  
  type selection_config = {
    num_examples: int;
    diversity_weight: float;  (* 0.0 到 1.0 *)
    recency_weight: float;
  }
  
  val select_examples : 
    query:string ->
    pool:example_pool ->
    config:selection_config ->
    (string * intent_hierarchy) list
    
  val update_pool : 
    new_examples:(string * intent_hierarchy * float) list ->
    pool:example_pool ->
    example_pool
end
```

**示例选择的关键策略**：
1. **相关性**：选择语义相似的示例
2. **多样性**：覆盖不同的意图类别
3. **时效性**：优先最近的高质量示例
4. **难度匹配**：选择类似复杂度的示例

**动态示例更新机制**：

```ocaml
module type EXAMPLE_UPDATER = sig
  type feedback = 
    | Correct 
    | Incorrect of intent_hierarchy
    | Ambiguous
    
  type update_strategy =
    | Immediate     (* 实时更新 *)
    | Batched of { window_minutes: int }
    | Threshold of { min_feedback_count: int }
    
  val record_feedback : 
    query:string ->
    predicted:intent_hierarchy ->
    feedback:feedback ->
    unit
    
  val trigger_update : 
    strategy:update_strategy ->
    unit Lwt.t
end
```

### 3.3 混合架构设计

纯 LLM 方案成本高延迟大，混合架构是生产系统的最佳选择。

```ocaml
module type HYBRID_INTENT_SYSTEM = sig
  type routing_decision =
    | FastPath of rule_based_result
    | DeepAnalysis of llm_based_result
    | Combined of {
        rule_result: rule_based_result;
        llm_result: llm_based_result;
        final_weight: float;
      }
      
  and rule_based_result = {
    intent: string;
    confidence: float;
    matched_rules: string list;
  }
  
  and llm_based_result = {
    intent: intent_hierarchy;
    confidence: float;
    reasoning: string;
  }
  
  val route_query : 
    query:string ->
    latency_budget_ms:int option ->
    routing_decision Lwt.t
end
```

**快速路径的设计要点**：
- **高频模式**：缓存常见查询模式
- **规则引擎**：简单模式匹配
- **词典查找**：领域特定术语
- **正则匹配**：结构化查询

**深度分析的触发条件**：
- 规则引擎低置信度
- 查询包含复杂语义
- 用户明确要求
- 关键业务场景

## 4. 实体抽取的集成策略

### 4.1 实体类型系统

```ocaml
module type ENTITY_TYPE_SYSTEM = sig
  type entity_type =
    | Standard of standard_entity
    | Custom of { 
        name: string;
        validator: string -> bool;
        normalizer: string -> string;
      }
      
  and standard_entity =
    | Person
    | Location  
    | Organization
    | DateTime
    | Money
    | Product
    | Event
    
  type entity = {
    text: string;
    start_pos: int;
    end_pos: int;
    entity_type: entity_type;
    confidence: float;
    metadata: (string * string) list;
  }
  
  type entity_relation = {
    subject: entity;
    relation: string;
    object_: entity;
    confidence: float;
  }
  
  val extract_entities : 
    text:string ->
    types:entity_type list option ->
    entity list Lwt.t
    
  val extract_relations :
    entities:entity list ->
    entity_relation list Lwt.t
end
```

**领域特定实体的设计**：

医疗领域示例：
- 疾病名称
- 药物名称  
- 症状描述
- 医疗程序

电商领域示例：
- 产品型号
- 品牌名称
- 价格范围
- 规格参数

### 4.2 抽取管道设计

```ocaml
module type EXTRACTION_PIPELINE = sig
  type pipeline_stage =
    | Tokenization
    | POSTagging  
    | NERModel of { model_name: string }
    | LLMExtraction of { prompt_template: string }
    | RuleBasedPostProcess
    | EntityLinking
    
  type pipeline_config = {
    stages: pipeline_stage list;
    parallel_stages: pipeline_stage list list option;
    cache_intermediate: bool;
  }
  
  val build_pipeline : 
    config:pipeline_config ->
    (string -> entity list Lwt.t)
    
  val with_validation :
    extractor:(string -> entity list Lwt.t) ->
    validator:(entity -> bool) ->
    (string -> entity list Lwt.t)
end
```

**LLM 增强抽取的设计模式**：

```ocaml
module type LLM_ENTITY_ENHANCER = sig
  type enhancement_strategy =
    | Refinement     (* 优化已抽取实体 *)
    | Expansion      (* 发现遗漏实体 *)
    | Disambiguation (* 消歧实体指代 *)
    | Normalization  (* 标准化实体表示 *)
    
  val enhance_entities :
    text:string ->
    initial_entities:entity list ->
    strategy:enhancement_strategy ->
    entity list Lwt.t
end
```

**后处理与验证机制**：

1. **边界调整**：修正实体边界
2. **类型验证**：确保类型正确
3. **冲突解决**：处理重叠实体
4. **一致性检查**：跨句实体一致

### 4.3 知识库集成

```ocaml
module type KNOWLEDGE_BASE_INTEGRATION = sig
  type kb_entity = {
    id: string;
    canonical_name: string;
    aliases: string list;
    entity_type: string;
    properties: (string * string) list;
  }
  
  type linking_result = {
    surface_form: string;
    kb_entity: kb_entity option;
    candidates: (kb_entity * float) list;
    linking_confidence: float;
  }
  
  val link_entity :
    surface_form:string ->
    context:string ->
    entity_type:entity_type option ->
    linking_result Lwt.t
    
  val batch_link :
    entities:entity list ->
    text:string ->
    linking_result list Lwt.t
end
```

**实体链接的关键挑战**：

1. **歧义消解**
   - "苹果" → Apple Inc. 还是水果？
   - 利用上下文和实体类型

2. **规范化**
   - "谷歌"、"Google"、"谷歌公司" → Google Inc.
   - 维护别名词典

3. **新实体发现**
   - 知识库中不存在的实体
   - 动态扩展机制

## 5. 会话管理的状态设计

### 5.1 会话状态模型

```ocaml
module type SESSION_STATE = sig
  type memory_type =
    | ShortTerm of { capacity: int; ttl_seconds: int }
    | LongTerm of { 
        storage_backend: [ `Redis | `DynamoDB | `PostgreSQL ];
        compression: bool;
      }
      
  type session_state = {
    session_id: string;
    user_id: string option;
    created_at: float;
    last_active: float;
    
    (* 对话历史 *)
    messages: message list;
    
    (* 实体记忆 *)
    mentioned_entities: (entity * float) list;
    
    (* 意图链 *)
    intent_sequence: intent_hierarchy list;
    
    (* 话题追踪 *)
    current_topic: string option;
    topic_history: (string * float) list;
    
    (* 用户偏好 *)
    preferences: (string * string) list;
  }
  
  and message = {
    role: [ `User | `Assistant | `System ];
    content: string;
    timestamp: float;
    metadata: (string * string) list;
  }
  
  val create_session : 
    user_id:string option -> 
    session_state
    
  val update_state :
    state:session_state ->
    message:message ->
    session_state
    
  val compress_state :
    state:session_state ->
    max_messages:int ->
    session_state
end
```

**状态压缩策略**：

```ocaml
module type STATE_COMPRESSION = sig
  type compression_strategy =
    | SlidingWindow of { size: int }
    | ImportanceBased of { 
        scorer: message -> float;
        keep_ratio: float;
      }
    | Summarization of {
        summarizer: message list -> string Lwt.t;
        chunk_size: int;
      }
      
  val compress :
    messages:message list ->
    strategy:compression_strategy ->
    message list Lwt.t
end
```

压缩的关键考虑：
- **信息保留**：关键信息不能丢失
- **连贯性**：压缩后对话仍然连贯
- **效率**：压缩本身不能太耗时

### 5.2 上下文理解架构

```ocaml
module type CONTEXT_UNDERSTANDING = sig
  type coreference = {
    mention: string;
    referent: string;
    confidence: float;
  }
  
  type topic_shift = {
    from_topic: string option;
    to_topic: string;
    transition_type: [ `Gradual | `Abrupt | `Related | `Unrelated ];
    confidence: float;
  }
  
  val resolve_coreferences :
    current_message:string ->
    context:message list ->
    coreference list Lwt.t
    
  val detect_topic_shift :
    current_message:string ->
    context:session_state ->
    topic_shift option Lwt.t
    
  val infer_implicit_intent :
    explicit_query:string ->
    context:session_state ->
    intent_hierarchy option Lwt.t
end
```

**指代消解的实现策略**：

1. **代词消解**
   - "它"、"这个"、"那里"
   - 最近提及原则 + 语义相关性

2. **省略恢复**
   - "还有呢？"→ 继续上一个查询
   - "便宜一点的" → 上一个产品类别

3. **隐含关系**
   - "对比一下" → 最近两个实体
   - "类似的" → 上一个查询结果

### 5.3 多轮交互优化

```ocaml
module type INTERACTION_OPTIMIZER = sig
  type clarification_strategy =
    | Options of { 
        question: string;
        options: string list;
        allow_other: bool;
      }
    | OpenEnded of {
        question: string;
        examples: string list option;
      }
    | Confirmation of {
        statement: string;
        confidence_threshold: float;
      }
      
  type feedback_signal =
    | ExplicitFeedback of [ `Positive | `Negative | `Neutral ]
    | ImplicitSignal of {
        signal_type: [ `Reformulation | `Abandonment | `Selection ];
        strength: float;
      }
      
  val should_clarify :
    query_understanding:classification_result ->
    context:session_state ->
    bool * clarification_strategy option
    
  val generate_clarification :
    ambiguity:ambiguity_type ->
    context:session_state ->
    string Lwt.t
    
  val learn_from_feedback :
    query:string ->
    feedback:feedback_signal ->
    unit Lwt.t
end
```

**澄清问题的生成原则**：

1. **最小化轮次**：一次解决主要歧义
2. **自然表达**：符合对话习惯
3. **选项合理**：提供有意义的选择
4. **容错设计**：允许用户跳过

**个性化适配机制**：

```ocaml
module type PERSONALIZATION = sig
  type user_profile = {
    user_id: string;
    query_patterns: (string * float) list;
    domain_preferences: (string * float) list;
    interaction_style: [ `Concise | `Detailed | `Exploratory ];
    expertise_level: [ `Beginner | `Intermediate | `Expert ];
  }
  
  val adapt_understanding :
    base_result:classification_result ->
    profile:user_profile ->
    classification_result
    
  val personalize_response :
    response:string ->
    profile:user_profile ->
    string
end
```

## 6. 本章小结

本章深入探讨了基于 LLM 的查询理解系统架构设计。核心要点包括：

1. **LLM 服务抽象**：通过统一接口支持多模型，实现成本、质量和延迟的灵活权衡
2. **Prompt 工程系统化**：版本管理、动态示例选择和 A/B 测试机制
3. **混合架构优势**：结合规则引擎的低延迟和 LLM 的高质量
4. **实体抽取管道**：多阶段处理 + LLM 增强 + 知识库链接
5. **会话状态管理**：分层记忆 + 智能压缩 + 上下文理解

关键设计原则：
- **渐进式理解**：从快速路径到深度分析的多级架构
- **容错优先**：完善的降级和缓存策略
- **成本意识**：在质量可接受前提下优化 token 使用
- **可观测性**：详细的指标收集和分析

## 7. 练习题

### 基础题

**练习 7.1**：设计一个 LLM 调用的去重机制，当相同或相似的查询在短时间内重复出现时，如何避免重复调用 LLM？

*提示*：考虑模糊匹配和时间窗口。

<details>
<summary>参考答案</summary>

使用查询指纹 + 语义哈希的双重机制：
1. 计算查询的 SimHash 用于快速相似度判断
2. 对标准化后的查询计算 MD5 作为精确匹配
3. 使用滑动时间窗口（如 5 分钟）内的 LRU 缓存
4. 相似度阈值可配置（如 0.95）
5. 对不同意图类型使用不同的缓存策略

</details>

**练习 7.2**：如何设计一个 prompt 模板的自动优化系统？给出评估指标和优化流程。

*提示*：考虑在线学习和离线评估的结合。

<details>
<summary>参考答案</summary>

评估指标：
- 意图识别准确率
- 实体抽取 F1 分数
- 用户满意度（点击率、停留时间）
- Token 使用效率

优化流程：
1. 收集 prompt 变体的性能数据
2. 使用多臂老虎机算法进行在线选择
3. 定期离线分析，生成新的 prompt 变体
4. 通过小流量实验验证新变体
5. 基于统计显著性决定是否全量

</details>

**练习 7.3**：设计一个会话状态的分布式存储方案，要求支持快速读写和跨数据中心同步。

*提示*：考虑 CAP 定理的权衡。

<details>
<summary>参考答案</summary>

两层架构：
1. 本地层：Redis 集群，存储热数据，TTL 30 分钟
2. 持久层：DynamoDB/Cassandra，全量数据，支持跨区域复制

同步策略：
- 写入时双写，本地优先
- 读取时先查本地，miss 则查远程
- 使用 CRDT 数据结构处理并发更新
- 异步同步 + 版本向量解决冲突

</details>

### 挑战题

**练习 7.4**：设计一个自适应的意图识别系统，能够自动发现新的意图类别并调整分类体系。

*提示*：考虑聚类算法和增量学习。

<details>
<summary>参考答案</summary>

系统架构：
1. 在线聚类：对低置信度查询进行实时聚类
2. 离线分析：
   - 使用 HDBSCAN 发现稳定的查询簇
   - 计算簇内查询的语义一致性
   - 生成候选意图描述（使用 LLM）
3. 人工审核：新意图需要人工确认
4. 渐进部署：
   - 先作为子意图测试
   - 收集足够数据后提升为主意图
5. 模型更新：增量训练意图分类器

</details>

**练习 7.5**：如何设计一个多语言查询理解系统，支持跨语言的实体链接和意图识别？

*提示*：考虑零样本跨语言迁移。

<details>
<summary>参考答案</summary>

架构设计：
1. 统一表示层：
   - 多语言编码器（如 mBERT, XLM-R）
   - 语言无关的意图向量空间
2. 翻译桥接：
   - 关键查询翻译到枢轴语言
   - 保留原语言进行实体识别
3. 知识库对齐：
   - 维护多语言实体映射表
   - 使用 Wikidata 等跨语言知识库
4. Few-shot 适配：
   - 每种语言少量标注数据
   - 使用 prompt 工程实现跨语言迁移

</details>

**练习 7.6**：设计一个查询理解的可解释性框架，让用户理解系统是如何理解他们的查询的。

*提示*：考虑注意力可视化和决策路径追踪。

<details>
<summary>参考答案</summary>

可解释性组件：
1. 意图识别解释：
   - 显示匹配的关键词/短语
   - 给出置信度分数分解
   - 展示相似查询示例
2. 实体高亮：
   - 在原查询中标注识别的实体
   - 显示实体类型和链接结果
3. 决策路径：
   - 展示规则引擎 vs LLM 的选择原因
   - 显示每步的中间结果
4. 交互式澄清：
   - 允许用户纠正理解错误
   - 实时更新理解结果

</details>

**练习 7.7**：如何处理对抗性查询（如 prompt 注入攻击）？设计一个安全的查询理解流程。

*提示*：考虑输入验证和输出过滤。

<details>
<summary>参考答案</summary>

安全措施：
1. 输入过滤：
   - 检测异常长度和特殊字符
   - 识别已知的注入模式
   - 限制查询复杂度
2. Prompt 隔离：
   - 系统 prompt 与用户输入严格分离
   - 使用特殊标记界定用户内容
3. 输出验证：
   - 检查输出是否包含敏感信息
   - 验证意图是否在允许范围内
4. 监控告警：
   - 异常查询模式检测
   - 频率限制和黑名单机制

</details>

**练习 7.8**：设计一个基于强化学习的查询理解优化系统，通过用户反馈不断改进理解质量。

*提示*：定义好状态空间、动作空间和奖励函数。

<details>
<summary>参考答案</summary>

RL 框架设计：
1. 状态空间：
   - 查询特征向量
   - 历史交互序列
   - 当前会话上下文
2. 动作空间：
   - 选择意图分类策略
   - 决定是否需要澄清
   - 选择实体抽取模型
3. 奖励信号：
   - 即时：用户点击、停留时间
   - 延迟：会话完成度、用户满意度
4. 优化算法：
   - 使用 PPO 或 SAC 进行策略优化
   - 离线预训练 + 在线微调
5. 探索策略：
   - ε-贪婪用于新策略尝试
   - 上限置信界(UCB)平衡探索利用

</details>

## 8. 常见陷阱与错误 (Gotchas)

### LLM 集成陷阱

1. **Token 预算失控**
   - 错误：对所有查询使用相同的 max_tokens
   - 正确：根据查询类型动态调整 token 预算

2. **缺乏超时控制**
   - 错误：无限等待 LLM 响应
   - 正确：设置合理超时 + 降级方案

3. **Prompt 泄露**
   - 错误：在 prompt 中包含敏感信息
   - 正确：参数化 prompt，敏感信息后注入

### 意图识别陷阱

4. **过度依赖表面特征**
   - 错误：仅基于关键词判断意图
   - 正确：结合上下文和查询结构

5. **忽视意图演化**
   - 错误：固定的意图分类体系
   - 正确：定期分析新出现的查询模式

6. **阈值设置不当**
   - 错误：全局统一的置信度阈值
   - 正确：基于意图类型和场景调整

### 实体抽取陷阱

7. **实体边界错误**
   - 错误：简单的空格分词
   - 正确：考虑语言特性和领域知识

8. **忽视共指消解**
   - 错误：独立处理每个实体提及
   - 正确：全文级别的实体统一

9. **知识库不一致**
   - 错误：实体 ID 硬编码
   - 正确：定期同步和验证机制

### 会话管理陷阱

10. **状态膨胀**
    - 错误：无限保存所有历史
    - 正确：智能压缩和过期策略

11. **上下文丢失**
    - 错误：简单截断历史消息
    - 正确：基于重要性的选择性保留

12. **并发更新冲突**
    - 错误：后写覆盖
    - 正确：乐观锁或 CRDT

## 9. 最佳实践检查清单

### 架构设计审查

- [ ] 是否实现了 LLM 服务的抽象层？
- [ ] 是否有多模型路由和切换机制？
- [ ] 是否设计了完整的降级方案？
- [ ] 是否实现了智能缓存策略？

### 性能优化审查

- [ ] 是否使用了批处理减少 LLM 调用？
- [ ] 是否实现了查询去重机制？
- [ ] 是否有快速路径处理简单查询？
- [ ] 是否监控了 P95 延迟？

### 质量保证审查

- [ ] 是否有 prompt 版本管理？
- [ ] 是否实现了 A/B 测试框架？
- [ ] 是否收集了用户反馈信号？
- [ ] 是否有异常查询检测？

### 成本控制审查

- [ ] 是否跟踪了每个查询的 token 使用？
- [ ] 是否实现了基于预算的模型选择？
- [ ] 是否有 token 使用量告警？
- [ ] 是否定期分析成本效益？

### 安全性审查

- [ ] 是否防御了 prompt 注入？
- [ ] 是否过滤了敏感信息？
- [ ] 是否实现了访问控制？
- [ ] 是否有审计日志？

### 可扩展性审查

- [ ] 是否支持新意图类型的添加？
- [ ] 是否支持新语言的扩展？
- [ ] 是否预留了模型升级路径？
- [ ] 是否支持灰度发布？

### 监控告警审查

- [ ] 是否监控了意图识别准确率？
- [ ] 是否跟踪了实体抽取 F1？
- [ ] 是否监控了 LLM 服务可用性？
- [ ] 是否设置了异常流量告警？

### 用户体验审查

- [ ] 是否提供了查询理解的解释？
- [ ] 是否优雅处理理解失败？
- [ ] 是否支持查询建议？
- [ ] 是否有个性化适配？
