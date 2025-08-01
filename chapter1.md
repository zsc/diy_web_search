# Chapter 1: 搜索引擎架构概览

本章介绍现代搜索引擎的核心架构设计，深入探讨从用户查询到返回结果的完整数据流。我们将使用 OCaml 类型系统定义清晰的模块接口，理解各组件如何协同工作以实现毫秒级响应。通过本章学习，你将掌握搜索引擎的基础架构模式，为后续深入学习各个子系统奠定基础。

## 1.1 搜索引擎的核心组件与数据流

现代搜索引擎是一个复杂的分布式系统，由多个高度专业化的组件协同工作。理解这些组件的职责划分和数据流转是设计高性能搜索系统的基础。

### 1.1.1 离线处理管道

离线管道负责构建和维护搜索索引，包含以下核心组件：

**爬虫系统 (Crawler)**  
负责从互联网抓取网页内容。现代爬虫需要处理 JavaScript 渲染、动态内容加载、robots.txt 遵守等复杂场景。爬虫调度器通过 URL frontier 管理待抓取队列，实现礼貌爬取和优先级调度。

**文档处理器 (Document Processor)**  
将原始 HTML 转换为结构化数据。主要功能包括：
- 文本提取：去除 HTML 标签，保留语义信息
- 元数据解析：提取标题、描述、发布时间等
- 链接分析：构建网页间的链接关系图
- 内容去重：通过 SimHash 或 MinHash 检测重复内容

**索引构建器 (Index Builder)**  
将处理后的文档转换为倒排索引。核心步骤包括：
- 分词 (Tokenization)：将文本切分为索引单元
- 词项规范化：词干提取、大小写转换
- 位置索引：记录词项在文档中的位置
- 索引压缩：使用变长编码减少存储空间

**排序特征计算器 (Ranking Feature Extractor)**  
预计算影响排序的静态特征：
- PageRank：基于链接结构的网页重要性
- 文档质量信号：内容长度、更新频率、用户行为
- 语义嵌入：通过 BERT 等模型生成文档向量表示

### 1.1.2 在线查询管道

在线管道处理用户查询并返回结果，要求极低延迟：

**查询处理器 (Query Processor)**  
解析和理解用户查询意图：
- 查询解析：识别短语、布尔操作符
- 查询扩展：同义词、相关词扩展
- 意图识别：导航型、信息型、交易型查询分类
- 实体识别：识别人名、地名、品牌等

**检索器 (Retriever)**  
从索引中快速找到相关文档：
- 倒排列表遍历：使用跳表加速
- 布尔查询处理：AND/OR/NOT 操作
- 短语查询：利用位置索引验证词序
- 早期终止：动态剪枝减少计算量

**排序器 (Ranker)**  
对候选文档进行相关性排序：
- 第一阶段：使用 BM25 等简单模型快速筛选
- 第二阶段：应用复杂的机器学习模型精排
- 特征融合：结合文本相关性、点击率、新鲜度等
- 个性化：根据用户历史调整排序

**结果聚合器 (Result Aggregator)**  
整合多个数据源生成最终结果：
- 摘要生成：提取包含查询词的片段
- 富媒体集成：图片、视频、知识图谱
- 去重与多样化：避免相似结果
- 缓存管理：热门查询结果缓存

### 1.1.3 数据流架构

典型的查询处理流程：

1. **查询接收**：负载均衡器分发请求到查询服务器
2. **查询理解**：解析查询，识别意图和实体
3. **缓存检查**：查找结果缓存和查询建议
4. **分布式检索**：并行查询多个索引分片
5. **结果合并**：收集各分片结果，全局排序
6. **增强处理**：生成摘要，添加相关搜索
7. **结果返回**：通过 CDN 加速响应

### 1.1.4 实时更新流

现代搜索引擎需要处理实时内容：

**增量索引**  
- 内存缓冲区：新文档先写入内存索引
- 定期合并：将内存索引合并到主索引
- 版本控制：支持文档更新和删除

**流式处理**  
- 变更检测：监控网页更新
- 实时爬取：对热门内容增加爬取频率
- 事件驱动：基于用户行为触发更新

### 1.1.5 架构权衡

设计搜索引擎时需要在多个维度做出权衡：

**延迟 vs 吞吐量**  
- 缓存策略：空间换时间
- 索引分片：并行化提升吞吐
- 查询优化：早期终止降低延迟

**精确度 vs 召回率**  
- 相关性阈值：过滤低质量结果
- 查询扩展：提升召回但可能引入噪音
- 个性化程度：提升精确度但增加计算

**新鲜度 vs 稳定性**  
- 爬取频率：平衡资源消耗
- 索引更新：权衡一致性和可用性
- 排序模型更新：A/B 测试验证效果

通过理解这些核心组件和数据流，我们可以设计出满足不同场景需求的搜索系统。接下来我们将深入探讨倒排索引这一搜索引擎的核心数据结构。

## 1.2 倒排索引的数据结构设计

倒排索引是搜索引擎的核心数据结构，它将传统的"文档→词项"映射反转为"词项→文档"映射，从而实现快速的全文检索。理解倒排索引的设计原理对构建高效搜索系统至关重要。

### 1.2.1 基本结构

倒排索引由两个主要部分组成：

**词典 (Dictionary/Lexicon)**  
存储所有唯一词项的数据结构，每个词项关联一个指向倒排列表的指针。词典需要支持：
- 快速查找：O(1) 哈希表或 O(log n) B+树
- 前缀匹配：支持自动补全功能
- 模糊匹配：处理拼写错误
- 内存效率：使用 Trie 或 FST 压缩存储

**倒排列表 (Posting Lists)**  
记录包含特定词项的所有文档及相关信息：
```ocaml
type posting = {
  doc_id: int;          (* 文档标识符 *)
  frequency: int;       (* 词项在文档中出现次数 *)
  positions: int list;  (* 词项在文档中的位置列表 *)
  payload: bytes;       (* 自定义数据，如权重、标签等 *)
}

type posting_list = posting list
```

### 1.2.2 存储优化

倒排索引的存储优化直接影响查询性能：

**文档ID编码**  
- Delta编码：存储相邻ID的差值而非绝对值
- 变长编码：VByte、Simple9、PForDelta
- 位图索引：适用于高频词项
- 跳表结构：加速列表遍历

**位置信息压缩**  
位置索引支持短语查询但占用大量空间：
- 相对位置编码：存储与前一位置的偏移
- 区间编码：合并连续位置为区间
- 分层索引：仅为重要词项存储位置
- 按需加载：查询时动态解压位置数据

**分层存储策略**  
根据词项频率采用不同存储方案：
- 高频词：使用位图或特殊编码
- 中频词：标准压缩倒排列表
- 低频词：简单数组存储
- 停用词：可选择性不索引

### 1.2.3 索引组织方式

**段式架构 (Segment-based)**  
将索引分为多个不可变段，新文档写入新段：
- 段内有序：便于压缩和遍历
- 段间独立：支持并行查询
- 定期合并：减少段数量，提升查询效率
- 删除标记：使用位图标记已删除文档

**分片策略 (Sharding)**  
将大索引切分为多个分片：
- 文档分片：按文档ID范围或哈希分配
- 词项分片：按词典序或哈希分配词项
- 混合分片：结合文档和词项维度
- 动态分片：根据负载自动调整

**索引版本管理**  
支持索引的原子更新和回滚：
- 多版本并发控制 (MVCC)
- 写时复制 (Copy-on-Write)
- 增量快照
- 事务日志

### 1.2.4 高级索引结构

**位置敏感索引**  
支持复杂的位置相关查询：
```ocaml
type position_index = {
  term_positions: (term * position list) list;
  proximity_index: (term * term * int) list;  (* 词项对距离 *)
  phrase_index: (phrase * doc_id list) list;  (* 常见短语 *)
}
```

**字段索引**  
区分文档的不同字段（标题、正文、URL等）：
```ocaml
type field_posting = {
  doc_id: int;
  field_occurrences: (field_id * frequency * position list) list;
}
```

**向量索引集成**  
现代搜索引擎集成向量检索：
- 稠密向量：BERT、GPT等模型生成的嵌入
- 稀疏向量：基于词项的传统表示
- 混合索引：同时存储倒排列表和向量
- 近似索引：LSH、HNSW等近似最近邻结构

### 1.2.5 查询处理优化

**查询计划优化**  
- 词项顺序：先处理选择性高的词项
- 交集算法：根据列表长度选择算法
- 并集优化：使用堆合并多个列表
- 动态剪枝：基于当前最高分提前终止

**缓存层次**  
- 列表缓存：缓存常用词项的倒排列表
- 交集缓存：缓存常见词项组合的结果
- 跳表缓存：缓存跳表节点加速遍历
- 分数缓存：缓存文档的静态分数

**并行处理**  
- 词项并行：不同词项的列表并行处理
- 段并行：多个段同时查询
- 文档并行：批量计算文档分数
- SIMD优化：使用向量指令加速

### 1.2.6 索引构建流程

**批量构建**  
适用于初始索引构建或完全重建：
1. 文档解析和分词
2. 内存中构建临时索引
3. 外部排序生成有序词项流
4. 合并相同词项的倒排列表
5. 压缩编码并写入磁盘

**增量更新**  
处理文档的添加、更新和删除：
- 内存缓冲区：新文档先索引到内存
- 日志结构：追加写入避免随机IO
- 后台合并：异步合并小段为大段
- 原子切换：更新索引视图

### 1.2.7 设计权衡

**空间 vs 查询性能**  
- 更多索引信息（位置、字段）提升功能但增加存储
- 压缩算法的选择影响解压速度
- 缓存大小直接影响查询延迟

**索引延迟 vs 查询新鲜度**  
- 实时索引增加系统复杂度
- 批量索引提升吞吐但增加延迟
- 混合方案平衡两者需求

**精确性 vs 性能**  
- 近似算法（如跳过低分文档）提升性能
- 有损压缩减少存储但可能影响排序
- 早期终止可能错过相关结果

通过精心设计倒排索引结构，我们可以在存储效率和查询性能之间找到最佳平衡点。接下来我们将探讨如何使用类型系统定义清晰的模块接口。

## 1.3 模块系统与接口定义

使用类型系统定义清晰的模块接口是构建可维护搜索引擎的关键。OCaml 的模块系统提供了强大的抽象能力，让我们能够精确描述组件间的契约，实现高内聚低耦合的系统设计。

### 1.3.1 核心模块签名

**文档抽象**
```ocaml
module type DOCUMENT = sig
  type t
  type field_id = int
  type field_value = 
    | Text of string
    | Number of float
    | Date of float
    | Keywords of string list
    
  val create : id:string -> fields:(field_id * field_value) list -> t
  val id : t -> string
  val get_field : t -> field_id -> field_value option
  val fields : t -> (field_id * field_value) list
  val serialize : t -> bytes
  val deserialize : bytes -> t
end
```

**词项处理接口**
```ocaml
module type TOKENIZER = sig
  type t
  type token = {
    term: string;
    position: int;
    offset: int * int;  (* 起始和结束字符位置 *)
    typ: [`Word | `Number | `Email | `Url | `Emoji]
  }
  
  val create : ?stop_words:string list -> ?stemmer:bool -> unit -> t
  val tokenize : t -> string -> token list
  val analyze : t -> field_id -> string -> token list
  val configure : t -> [`Language of string | `Custom of (string -> token list)] -> t
end
```

**索引接口**
```ocaml
module type INDEX = sig
  type t
  type doc_id = int
  type term = string
  type posting = {
    doc_id: doc_id;
    frequency: int;
    positions: int array;
    score: float;
  }
  
  val create : unit -> t
  val add_document : t -> doc_id -> (term * int * int array) list -> unit
  val get_posting_list : t -> term -> posting array
  val delete_document : t -> doc_id -> unit
  val commit : t -> unit
  val close : t -> unit
  
  (* 高级查询接口 *)
  val term_query : t -> term -> doc_id array
  val phrase_query : t -> term array -> doc_id array
  val boolean_query : t -> [`And | `Or | `Not] -> term list -> doc_id array
end
```

### 1.3.2 查询处理模块

**查询解析器**
```ocaml
module type QUERY_PARSER = sig
  type query = 
    | Term of string
    | Phrase of string list
    | Boolean of bool_op * query * query
    | Field of field_id * query
    | Range of field_id * [`Gt | `Gte | `Lt | `Lte] * float
    | Fuzzy of string * int  (* 编辑距离 *)
    | Wildcard of string
  and bool_op = And | Or | Not
  
  val parse : string -> query
  val optimize : query -> query  (* 查询重写优化 *)
  val to_string : query -> string
  
  (* 查询分析 *)
  val extract_terms : query -> string list
  val estimate_cost : query -> int
  val requires_positions : query -> bool
end
```

**查询执行器**
```ocaml
module type QUERY_EXECUTOR = sig
  type t
  type result = {
    doc_id: doc_id;
    score: float;
    explanation: string option;
  }
  
  val create : index:INDEX.t -> scorer:SCORER.t -> t
  val execute : t -> query -> limit:int -> result array
  val explain : t -> query -> doc_id -> string
  
  (* 查询优化提示 *)
  type hint = 
    | UseCache of string
    | ParallelMerge of int
    | EarlyTermination of float
    | SkipOptimization
  
  val with_hints : t -> hint list -> t
end
```

### 1.3.3 存储层抽象

**持久化存储接口**
```ocaml
module type STORAGE = sig
  type t
  type key = bytes
  type value = bytes
  
  val open_db : string -> t
  val get : t -> key -> value option
  val put : t -> key -> value -> unit
  val delete : t -> key -> unit
  val batch : t -> (unit -> unit) -> unit  (* 批量操作 *)
  
  (* 迭代器接口 *)
  module Iterator : sig
    type iterator
    val create : t -> ?prefix:key -> unit -> iterator
    val next : iterator -> (key * value) option
    val seek : iterator -> key -> unit
    val close : iterator -> unit
  end
  
  (* 存储统计 *)
  val size : t -> int64
  val compact : t -> unit
  val checkpoint : t -> string -> unit
end
```

**段管理器**
```ocaml
module type SEGMENT_MANAGER = sig
  type t
  type segment_id = int
  
  module Segment : sig
    type t
    val id : t -> segment_id
    val doc_count : t -> int
    val size_bytes : t -> int64
    val created_at : t -> float
    val is_deleted : t -> doc_id -> bool
  end
  
  val create : storage:STORAGE.t -> t
  val new_segment : t -> Segment.t
  val get_segment : t -> segment_id -> Segment.t option
  val list_segments : t -> Segment.t list
  val merge_segments : t -> Segment.t list -> Segment.t
  val delete_segment : t -> segment_id -> unit
  
  (* 合并策略 *)
  type merge_policy = 
    | TieredMerge of {max_segments: int; segments_per_tier: int}
    | LogMerge of {merge_factor: int; max_merge_at_once: int}
    | NoMerge
  
  val set_merge_policy : t -> merge_policy -> unit
end
```

### 1.3.4 分布式接口

**RPC 通信接口**
```ocaml
module type RPC = sig
  type t
  type node_id = string
  type request = 
    | Search of {query: string; from: int; size: int}
    | Index of {doc_id: string; document: bytes}
    | Delete of {doc_id: string}
    | Status
  
  type response = 
    | SearchResult of {hits: doc_id array; total: int; took_ms: int}
    | IndexResult of {success: bool; version: int}
    | DeleteResult of {success: bool}
    | StatusResult of {docs: int; size_bytes: int64; uptime: float}
  
  val create : port:int -> t
  val call : t -> node_id -> request -> response Lwt.t
  val broadcast : t -> request -> (node_id * response) list Lwt.t
  
  (* 服务发现 *)
  val discover_nodes : t -> node_id list Lwt.t
  val register_node : t -> node_id -> string -> unit
  val health_check : t -> node_id -> bool Lwt.t
end
```

**分片路由器**
```ocaml
module type SHARD_ROUTER = sig
  type t
  type shard_id = int
  
  val create : total_shards:int -> replicas:int -> t
  val route_document : t -> doc_id -> shard_id
  val route_query : t -> query -> shard_id list
  
  (* 分片分配策略 *)
  type allocation = {
    shard: shard_id;
    primary: node_id;
    replicas: node_id list;
  }
  
  val get_allocation : t -> allocation list
  val reassign_shard : t -> shard_id -> primary:node_id -> unit
  val rebalance : t -> unit  (* 自动重平衡 *)
end
```

### 1.3.5 扩展点设计

**插件系统**
```ocaml
module type PLUGIN = sig
  type t
  val name : string
  val version : string
  val init : config:JSON.t -> t
  val destroy : t -> unit
end

module type ANALYZER_PLUGIN = sig
  include PLUGIN
  val analyze : t -> string -> TOKENIZER.token list
end

module type SCORER_PLUGIN = sig
  include PLUGIN
  val score : t -> doc_id -> query -> float
  val explain : t -> doc_id -> query -> string
end

module type FILTER_PLUGIN = sig
  include PLUGIN
  val filter : t -> doc_id array -> query -> doc_id array
end
```

**生命周期管理**
```ocaml
module type LIFECYCLE = sig
  type t
  type phase = 
    | Starting
    | Running
    | Stopping
    | Stopped
  
  val current_phase : t -> phase
  val on_start : t -> (unit -> unit) -> unit
  val on_stop : t -> (unit -> unit) -> unit
  val start : t -> unit Lwt.t
  val stop : t -> unit Lwt.t
  
  (* 健康检查 *)
  type health_status = Green | Yellow | Red
  val health_check : t -> health_status * string
  val add_health_check : t -> string -> (unit -> bool) -> unit
end
```

### 1.3.6 类型安全的配置

**配置模块**
```ocaml
module type CONFIG = sig
  type t
  
  val default : t
  val from_file : string -> t
  val validate : t -> (unit, string) result
  
  (* 类型安全的配置访问 *)
  val index_buffer_size : t -> int
  val merge_scheduler_threads : t -> int
  val query_timeout_ms : t -> int
  val cache_size_mb : t -> int
  
  (* 动态配置更新 *)
  val set : t -> string -> JSON.t -> (t, string) result
  val subscribe : t -> string -> (JSON.t -> unit) -> unit
end
```

通过这些精心设计的模块接口，我们实现了：
- **类型安全**：编译时捕获接口不匹配
- **抽象封装**：隐藏实现细节，只暴露必要接口
- **可组合性**：模块可以灵活组合构建复杂功能
- **可测试性**：易于创建模拟实现进行测试
- **可扩展性**：通过函子和插件机制支持扩展

接下来我们将深入探讨内存索引的具体类型设计。

## 1.4 内存索引的类型签名设计

## 本章小结

## 练习题

## 常见陷阱与错误

## 最佳实践检查清单