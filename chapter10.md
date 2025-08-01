# Chapter 10: 分布式查询处理 (Distributed Query Processing)

在构建能够处理数十亿文档、服务数百万并发用户的搜索引擎时，单机架构的局限性变得显而易见。本章深入探讨分布式查询处理的核心设计原则，从分片策略到一致性模型，从故障恢复到分布式协调。我们将分析如何在保证亚秒级响应时间的同时，实现系统的高可用性和可扩展性。通过 OCaml 类型系统，我们将定义清晰的分布式组件接口，帮助读者理解大规模搜索系统的架构权衡。

## 10.1 分片策略的设计原则 (Sharding Strategies)

分片是分布式搜索系统的基石。正确的分片策略直接影响查询性能、负载均衡和系统扩展性。

### 10.1.1 文档分片方法 (Document Sharding)

```ocaml
module type DOCUMENT_SHARDING = sig
  type doc_id = int64
  type shard_id = int
  type document
  
  (* 分片函数的类型签名 *)
  type sharding_function = doc_id -> shard_id
  
  (* 分片元数据 *)
  type shard_metadata = {
    shard_id: shard_id;
    doc_count: int;
    size_bytes: int64;
    replicas: node_id list;
  }
  
  val hash_based_sharding : num_shards:int -> sharding_function
  val range_based_sharding : boundaries:doc_id list -> sharding_function
  val custom_sharding : (document -> shard_id) -> sharding_function
end
```

**Hash-based Sharding**

最简单且最常用的分片策略。通过对文档ID进行哈希运算来决定分片：

- **优点**: 负载均衡良好，实现简单
- **缺点**: 增加分片需要大量数据迁移
- **适用场景**: 文档分布均匀，增长可预测的系统

**Range-based Sharding**

基于文档ID或其他属性的范围进行分片：

- **优点**: 范围查询效率高，易于增加新分片
- **缺点**: 可能导致热点问题
- **适用场景**: 有明确访问模式的系统

**Geo-sharding**

基于地理位置的分片策略：

```ocaml
type geo_shard_config = {
  regions: (region_id * coordinate_bounds) list;
  fallback_strategy: sharding_function;
}
```

- **优点**: 降低地理延迟，符合数据主权要求
- **缺点**: 跨地域查询复杂
- **适用场景**: 全球化部署的搜索服务

### 10.1.2 索引分片架构 (Index Partitioning)

**Term-partitioned Index**

按词项分片，每个分片包含特定词项范围的完整倒排列表：

```ocaml
module type TERM_PARTITIONED_INDEX = sig
  type term
  type posting_list
  
  (* 词项路由表 *)
  type routing_table = {
    term_ranges: (term * term * shard_id) list;
    bloom_filters: (shard_id * bloom_filter) list;
  }
  
  val route_term_query : term -> routing_table -> shard_id list
  val merge_posting_lists : posting_list list -> posting_list
end
```

**Document-partitioned Index**

按文档分片，每个分片包含一部分文档的完整索引：

- **优点**: 查询并行度高，易于扩展
- **缺点**: 需要查询所有分片
- **优化**: 使用分片级别的摘要信息进行剪枝

### 10.1.3 查询路由设计 (Query Routing)

```ocaml
module type QUERY_ROUTER = sig
  type query
  type shard_query
  type partial_result
  type final_result
  
  (* 查询计划 *)
  type query_plan = {
    shard_queries: (shard_id * shard_query) list;
    merge_strategy: partial_result list -> final_result;
    timeout_ms: int;
  }
  
  val plan_query : query -> shard_metadata list -> query_plan
  val adaptive_routing : query -> historical_stats -> query_plan
end
```

**Scatter-Gather 模式**

经典的分布式查询模式：
1. 查询分发到相关分片
2. 并行执行分片查询
3. 收集并合并结果

**智能路由优化**

- **Bloom Filter 预过滤**: 避免查询不包含目标词项的分片
- **统计信息路由**: 基于词频分布优化查询计划
- **自适应超时**: 根据历史延迟动态调整超时时间

## 10.2 一致性模型的选择 (Consistency Models)

在分布式搜索系统中，一致性模型的选择直接影响系统的可用性和性能。

### 10.2.1 CAP定理在搜索系统中的应用

搜索系统通常选择 AP (可用性 + 分区容错性)：

```ocaml
module type CONSISTENCY_MODEL = sig
  type version = int64
  type conflict_resolution = 
    | LastWriteWins
    | VectorClock of (node_id * int) list
    | CustomResolver of (document list -> document)
  
  type consistency_level =
    | Strong        (* 线性一致性 *)
    | Eventual      (* 最终一致性 *)
    | BoundedStaleness of int  (* 有界陈旧性 *)
    | SessionConsistent       (* 会话一致性 *)
end
```

### 10.2.2 最终一致性设计

**Version Vectors**

用于检测并发更新和解决冲突：

```ocaml
type version_vector = (node_id * version) list

module VersionVector = struct
  let compare vv1 vv2 =
    (* 返回: Concurrent | Before | After | Equal *)
    
  let merge vv1 vv2 =
    (* 合并两个版本向量 *)
    
  let increment node_id vv =
    (* 递增特定节点的版本号 *)
end
```

**冲突解决策略**

1. **Last-Write-Wins (LWW)**: 简单但可能丢失更新
2. **Multi-Value**: 保留所有版本，查询时解决
3. **CRDT**: 使用无冲突复制数据类型

### 10.2.3 强一致性需求场景

某些场景需要强一致性：

- 用户配置更新
- 访问控制列表
- 计费相关数据

```ocaml
module type CONSENSUS_PROTOCOL = sig
  type proposal
  type decision
  
  (* Raft/Paxos 接口 *)
  val propose : proposal -> (decision, error) result Lwt.t
  val read_committed : key -> value option Lwt.t
end
```

### 10.2.4 混合一致性模型

实际系统常采用混合模型，根据数据类型选择不同的一致性级别：

```ocaml
module type MIXED_CONSISTENCY = sig
  type data_class = 
    | Critical      (* 强一致性 *)
    | Important     (* 有界陈旧性 *)
    | Regular       (* 最终一致性 *)
    | Cached        (* 尽力而为 *)
    
  val classify_data : data_type -> data_class
  val get_consistency_level : data_class -> consistency_level
end
```

## 10.3 故障恢复的架构模式 (Fault Recovery Patterns)

高可用性是分布式搜索系统的核心需求。本节探讨各种故障恢复机制。

### 10.3.1 副本管理策略 (Replica Management)

**Primary-Backup Replication**

主从复制模式，适合读多写少的搜索场景：

```ocaml
module type PRIMARY_BACKUP = sig
  type replica_state = Primary | Backup | Syncing
  
  type replica_config = {
    replication_factor: int;
    sync_mode: Sync | Async | SemiSync;
    failover_timeout_ms: int;
  }
  
  (* 副本状态机 *)
  type state_transition =
    | Promote : backup_node -> state_transition
    | Demote : primary_node -> state_transition
    | AddReplica : node_id -> state_transition
    
  val elect_new_primary : replica_set -> node_id option
  val synchronize_replica : primary -> backup -> unit Lwt.t
end
```

**Multi-Master Replication**

多主复制，适合地理分布式部署：

- **优点**: 就近写入，高可用性
- **缺点**: 冲突解决复杂
- **应用**: 用户个性化数据、查询日志

**Quorum-based Replication**

基于多数派的复制策略：

```ocaml
type quorum_config = {
  n: int;  (* 总副本数 *)
  w: int;  (* 写入需要的副本数 *)
  r: int;  (* 读取需要的副本数 *)
}

(* 确保 w + r > n 以保证强一致性 *)
```

### 10.3.2 故障检测机制 (Failure Detection)

**Heartbeat Protocols**

```ocaml
module type HEARTBEAT_DETECTOR = sig
  type heartbeat = {
    node_id: node_id;
    timestamp: float;
    load_info: load_statistics;
  }
  
  type detector_config = {
    interval_ms: int;
    timeout_ms: int;
    failure_threshold: int;
  }
  
  val monitor_node : node_id -> health_status Lwt_stream.t
  val adaptive_timeout : historical_latency -> int
end
```

**Gossip-based Failure Detection**

使用 Gossip 协议进行分布式故障检测：

- **Phi Accrual Failure Detector**: 提供连续的怀疑级别而非二元判断
- **SWIM Protocol**: 可扩展的成员管理和故障检测

### 10.3.3 自动恢复流程 (Automatic Recovery)

**副本提升 (Replica Promotion)**

```ocaml
module type FAILOVER_COORDINATOR = sig
  type promotion_strategy =
    | FastestReplica       (* 选择延迟最低的副本 *)
    | MostUpToDate        (* 选择数据最新的副本 *)
    | LoadBased           (* 基于负载选择 *)
    | CustomScoring of (replica_info -> float)
    
  val initiate_failover : failed_primary -> replica_set -> unit Lwt.t
  val verify_promotion : new_primary -> bool Lwt.t
end
```

**数据重建 (Data Reconstruction)**

当副本数低于预期时触发：

1. **增量同步**: 只复制缺失的数据
2. **并行重建**: 从多个源并行恢复
3. **优先级重建**: 热数据优先恢复

**索引重建策略**

```ocaml
module type INDEX_RECOVERY = sig
  type recovery_mode =
    | FullRebuild         (* 完全重建 *)
    | IncrementalRepair   (* 增量修复 *)
    | LogReplay          (* 日志重放 *)
    
  type recovery_stats = {
    docs_processed: int;
    time_elapsed_ms: int;
    errors: error list;
  }
  
  val rebuild_index : shard_id -> recovery_mode -> recovery_stats Lwt.t
  val verify_index_integrity : shard_id -> validation_result
end
```

## 10.4 分布式协调的接口设计 (Distributed Coordination)

分布式协调是保证系统正确运行的关键。本节讨论协调服务的设计模式。

### 10.4.1 协调服务架构 (Coordination Service)

**ZooKeeper/etcd 集成模式**

```ocaml
module type COORDINATION_SERVICE = sig
  type path = string
  type data = bytes
  type version = int
  
  (* 基础操作 *)
  val create : path -> data -> persistent:bool -> sequential:bool -> path Lwt.t
  val get : path -> (data * version) option Lwt.t
  val set : path -> data -> version -> unit Lwt.t
  val delete : path -> version -> unit Lwt.t
  
  (* 监听机制 *)
  type watch_event = 
    | NodeCreated | NodeDeleted | NodeDataChanged | NodeChildrenChanged
    
  val watch : path -> watch_event Lwt_stream.t
end
```

**服务发现机制**

```ocaml
module type SERVICE_DISCOVERY = sig
  type service_name = string
  type service_instance = {
    id: string;
    address: string;
    port: int;
    metadata: (string * string) list;
    health_check_url: string option;
  }
  
  (* 服务注册与发现 *)
  val register : service_name -> service_instance -> lease Lwt.t
  val discover : service_name -> service_instance list Lwt.t
  val subscribe : service_name -> service_instance list Lwt_stream.t
end
```

**应用场景**：

1. **分片分配**: 动态管理分片到节点的映射
2. **配置管理**: 集中式配置更新与分发
3. **Leader 选举**: 协调组件的主节点选举

### 10.4.2 分布式锁与租约 (Distributed Locks and Leases)

**锁服务设计**

```ocaml
module type DISTRIBUTED_LOCK = sig
  type lock_path = string
  type lock_holder = {
    client_id: string;
    acquired_at: float;
    ttl_seconds: int;
  }
  
  (* 锁操作 *)
  val try_acquire : lock_path -> ttl:int -> lock_token option Lwt.t
  val acquire : lock_path -> ttl:int -> lock_token Lwt.t  (* 阻塞 *)
  val release : lock_token -> unit Lwt.t
  val extend : lock_token -> ttl:int -> bool Lwt.t
end
```

**基于租约的协调**

租约机制用于资源的临时独占访问：

```ocaml
module type LEASE_MANAGER = sig
  type lease = {
    resource_id: string;
    holder_id: string;
    expires_at: float;
    renewable: bool;
  }
  
  val acquire_lease : resource_id -> duration:int -> lease option Lwt.t
  val renew_lease : lease -> duration:int -> lease option Lwt.t
  val revoke_lease : lease -> unit Lwt.t
  
  (* 租约过期回调 *)
  val on_lease_expired : (lease -> unit Lwt.t) -> unit
end
```

**使用场景**：

- **索引更新锁**: 防止并发的段合并操作
- **爬虫 URL 分配**: 确保 URL 不被重复爬取
- **缓存失效**: 协调分布式缓存的更新

### 10.4.3 配置管理与动态更新 (Configuration Management)

**配置传播机制**

```ocaml
module type CONFIG_MANAGER = sig
  type config_key = string
  type config_value = string
  type config_version = int
  
  (* 配置变更通知 *)
  type config_change = {
    key: config_key;
    old_value: config_value option;
    new_value: config_value option;
    version: config_version;
  }
  
  val get_config : config_key -> (config_value * config_version) option
  val set_config : config_key -> config_value -> unit Lwt.t
  val watch_config : config_key -> config_change Lwt_stream.t
  
  (* 批量配置更新 *)
  val atomic_update : (config_key * config_value) list -> unit Lwt.t
end
```

**滚动更新策略**

```ocaml
module type ROLLING_UPDATE = sig
  type update_strategy =
    | Canary of float    (* 金丝雀发布百分比 *)
    | BlueGreen         (* 蓝绿部署 *)
    | Progressive of int (* 逐步更新批次大小 *)
    
  type update_state = 
    | Planning | InProgress of float | Completed | RolledBack
    
  val initiate_update : 
    update_strategy -> 
    target_version:string -> 
    update_handle Lwt.t
    
  val monitor_update : update_handle -> update_state Lwt_stream.t
  val rollback_update : update_handle -> unit Lwt.t
end
```

### 10.4.4 查询协调器设计 (Query Coordinator)

**查询计划与分发**

```ocaml
module type QUERY_COORDINATOR = sig
  type distributed_query_plan = {
    query_id: string;
    sub_queries: (shard_id * query_fragment) list;
    aggregation_plan: aggregation_strategy;
    timeout_budget: timeout_allocation;
  }
  
  (* 超时预算分配 *)
  type timeout_allocation = {
    total_timeout_ms: int;
    network_overhead_ms: int;
    processing_timeout_ms: int;
    aggregation_timeout_ms: int;
  }
  
  (* 查询执行 *)
  val execute_distributed_query : 
    query -> 
    shard_topology -> 
    distributed_result Lwt.t
    
  (* 自适应查询优化 *)
  val optimize_query_plan : 
    query -> 
    historical_performance -> 
    distributed_query_plan
end
```

**结果聚合模式**

```ocaml
module type RESULT_AGGREGATOR = sig
  type merge_strategy =
    | TopK of int           (* 合并 Top-K 结果 *)
    | SortedMerge          (* 有序合并 *)
    | SetUnion             (* 集合并集 *)
    | Custom of (partial_result list -> final_result)
    
  (* 流式聚合 *)
  val create_aggregator : 
    merge_strategy -> 
    expected_shards:int -> 
    aggregator
    
  val add_partial_result : 
    aggregator -> 
    shard_id -> 
    partial_result -> 
    unit
    
  val finalize : aggregator -> final_result
  
  (* 早期终止优化 *)
  val can_terminate_early : aggregator -> bool
end
```

**分布式追踪集成**

```ocaml
module type DISTRIBUTED_TRACING = sig
  type trace_id = string
  type span_id = string
  
  type span = {
    trace_id: trace_id;
    span_id: span_id;
    parent_span_id: span_id option;
    operation: string;
    start_time: float;
    duration_ms: float;
    tags: (string * string) list;
  }
  
  (* 追踪上下文传播 *)
  val inject_context : trace_context -> query -> query
  val extract_context : query -> trace_context option
  
  (* 性能分析 *)
  val analyze_trace : trace_id -> performance_report
end
```