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

这种模式的关键挑战在于：
- **尾延迟问题**: 最慢的分片决定整体延迟
- **部分失败处理**: 某些分片超时时的降级策略
- **资源效率**: 避免查询无关分片

**智能路由优化**

- **Bloom Filter 预过滤**: 避免查询不包含目标词项的分片
- **统计信息路由**: 基于词频分布优化查询计划
- **自适应超时**: 根据历史延迟动态调整超时时间

```ocaml
module SmartRouter = struct
  (* 分片摘要信息 *)
  type shard_summary = {
    shard_id: shard_id;
    bloom_filter: bloom_filter;
    term_statistics: term_stats;
    latency_p99: float;
  }
  
  (* 查询剪枝 *)
  let prune_shards query summaries =
    summaries
    |> List.filter (fun s ->
      query.terms 
      |> List.exists (fun term ->
        BloomFilter.might_contain s.bloom_filter term))
        
  (* 动态超时计算 *)
  let calculate_timeout query shard_latencies =
    let base_timeout = 100 in  (* 基础超时 100ms *)
    let p99_latency = Statistics.percentile 0.99 shard_latencies in
    min (base_timeout + int_of_float (p99_latency *. 1.5)) 1000
end
```

**查询复写与优化**

在路由之前，可以对查询进行优化：

1. **同义词扩展**: 在路由层处理可减少分片计算
2. **停用词过滤**: 避免高频词造成的负载倾斜
3. **查询重写**: 将复杂查询分解为更高效的子查询

### 10.1.4 负载均衡策略 (Load Balancing)

**分片负载评估**

```ocaml
module type LOAD_BALANCER = sig
  type load_metric = {
    cpu_usage: float;
    memory_usage: float;
    qps: int;
    avg_latency_ms: float;
    queue_length: int;
  }
  
  type routing_decision = 
    | Primary of shard_id
    | Replica of shard_id * replica_id
    | Reject  (* 过载保护 *)
    
  val select_replica : shard_id -> load_metric list -> routing_decision
  val rebalance_shards : load_metric list -> rebalancing_plan
end
```

**动态负载均衡算法**

1. **加权轮询 (Weighted Round Robin)**
   - 根据节点容量分配权重
   - 适合负载相对稳定的场景

2. **最小连接数 (Least Connections)**
   - 选择当前连接数最少的节点
   - 适合长连接场景

3. **响应时间感知 (Response Time Aware)**
   - 优先选择响应时间短的节点
   - 需要持续监控和统计

4. **一致性哈希 (Consistent Hashing)**
   - 保证相同查询路由到相同节点
   - 有利于缓存命中率

**热点缓解策略**

当某些分片成为热点时：

```ocaml
module HotspotMitigation = struct
  type mitigation_strategy =
    | AddReplicas of int          (* 增加副本数 *)
    | SplitShard of split_point   (* 分裂分片 *)
    | CachePopular of cache_config (* 缓存热数据 *)
    | RateLimiting of rate_config (* 限流保护 *)
    
  let detect_hotspot metrics threshold =
    metrics
    |> List.filter (fun m -> 
      m.cpu_usage > threshold.cpu_threshold ||
      m.qps > threshold.qps_threshold)
      
  let apply_mitigation shard_id strategy =
    match strategy with
    | AddReplicas n -> 
        (* 动态增加副本 *)
    | SplitShard point ->
        (* 分裂过热分片 *)
    | _ -> (* 其他策略 *)
end
```

### 10.1.5 分片再平衡 (Shard Rebalancing)

**触发条件**

分片再平衡通常由以下条件触发：

1. **容量触发**: 某分片接近存储上限
2. **性能触发**: 负载严重不均
3. **扩容触发**: 添加新节点
4. **故障触发**: 节点故障后的数据重分布

**再平衡算法**

```ocaml
module type REBALANCER = sig
  type rebalance_plan = {
    moves: (doc_range * source_shard * target_shard) list;
    estimated_duration: int;
    data_size_bytes: int64;
  }
  
  type constraints = {
    max_concurrent_moves: int;
    bandwidth_limit_mbps: int;
    business_hours_only: bool;
  }
  
  val plan_rebalance : 
    current_distribution -> 
    target_distribution -> 
    constraints -> 
    rebalance_plan
    
  val execute_rebalance : 
    rebalance_plan -> 
    progress Lwt_stream.t
end
```

**最小化数据移动**

再平衡的关键是最小化数据移动量：

1. **贪心算法**: 每次选择能最大改善平衡度的移动
2. **线性规划**: 将问题建模为优化问题
3. **启发式算法**: 基于经验规则的快速近似

**在线再平衡**

支持不停机的再平衡：

```ocaml
module OnlineRebalancing = struct
  type migration_state =
    | Preparing      (* 准备阶段 *)
    | Copying        (* 数据复制 *)
    | Catching_up    (* 追赶更新 *)
    | Switching      (* 切换流量 *)
    | Cleaning_up    (* 清理旧数据 *)
    
  let migrate_shard_range source target range =
    (* 1. 开始复制历史数据 *)
    let* snapshot = create_snapshot source range in
    let* () = copy_to_target target snapshot in
    
    (* 2. 追赶实时更新 *)
    let* update_stream = subscribe_updates source range in
    let* () = replay_updates target update_stream in
    
    (* 3. 原子切换 *)
    let* () = update_routing_table range target in
    
    (* 4. 清理源数据 *)
    cleanup_source source range
end
```

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

**数据分类策略**

不同类型的数据需要不同的一致性保证：

1. **索引数据**
   - 倒排列表: 最终一致性（可接受短暂不一致）
   - 文档元数据: 有界陈旧性（影响排序准确性）
   - 删除标记: 强一致性（避免显示已删除文档）

2. **用户数据**
   - 搜索历史: 会话一致性（用户看到自己的操作）
   - 个性化设置: 强一致性（立即生效）
   - 点击日志: 最终一致性（用于离线分析）

3. **系统元数据**
   - 分片映射: 强一致性（路由正确性）
   - 配置信息: 有界陈旧性（可容忍短暂延迟）
   - 监控数据: 最终一致性（趋势分析）

```ocaml
module AdaptiveConsistency = struct
  (* 动态调整一致性级别 *)
  type consistency_selector = {
    mutable default_level: consistency_level;
    overrides: (data_type, consistency_level) Hashtbl.t;
    load_threshold: float;
  }
  
  let select_consistency selector data_type current_load =
    match Hashtbl.find_opt selector.overrides data_type with
    | Some level -> level
    | None ->
        (* 高负载时降级一致性要求 *)
        if current_load > selector.load_threshold then
          downgrade_consistency selector.default_level
        else
          selector.default_level
          
  let downgrade_consistency = function
    | Strong -> BoundedStaleness 5
    | BoundedStaleness n -> BoundedStaleness (n * 2)
    | SessionConsistent -> Eventual
    | Eventual -> Eventual
end
```

### 10.2.5 读写分离与一致性 (Read-Write Separation)

**读写分离架构**

```ocaml
module type READ_WRITE_SEPARATION = sig
  type write_concern = {
    w: int;              (* 写入副本数 *)
    j: bool;             (* 是否等待持久化 *)
    timeout_ms: int;     (* 写入超时 *)
  }
  
  type read_preference = 
    | Primary            (* 只读主节点 *)
    | PrimaryPreferred   (* 优先主节点 *)
    | Secondary         (* 只读从节点 *)
    | SecondaryPreferred (* 优先从节点 *)
    | Nearest           (* 最近节点 *)
    
  val write : key -> value -> write_concern -> unit Lwt.t
  val read : key -> read_preference -> value option Lwt.t
end
```

**读一致性保证**

确保读取到最新写入的数据：

1. **Read-after-Write 一致性**
   ```ocaml
   module ReadAfterWrite = struct
     type session_token = {
       client_id: string;
       last_write_version: version;
       timestamp: float;
     }
     
     let ensure_read_after_write token read_preference =
       match read_preference with
       | Secondary | SecondaryPreferred ->
           (* 检查从节点是否已同步到所需版本 *)
           wait_for_replication token.last_write_version
       | _ -> Lwt.return_unit
   end
   ```

2. **单调读一致性**
   ```ocaml
   module MonotonicReads = struct
     (* 确保后续读取不会看到更旧的数据 *)
     type read_tracker = {
       mutable last_read_version: version;
       mutable last_read_node: node_id;
     }
     
     let select_read_node tracker available_nodes =
       (* 优先选择上次读取的节点或更新的节点 *)
       available_nodes
       |> List.filter (fun node ->
         get_node_version node >= tracker.last_read_version)
       |> select_best_node
   end
   ```

### 10.2.6 时钟同步与时序一致性 (Clock Synchronization)

分布式系统中的时间同步对一致性至关重要：

**混合逻辑时钟 (Hybrid Logical Clock)**

```ocaml
module HybridLogicalClock = struct
  type hlc = {
    physical_time: int64;
    logical_time: int;
    node_id: node_id;
  }
  
  let compare hlc1 hlc2 =
    match Int64.compare hlc1.physical_time hlc2.physical_time with
    | 0 -> 
        (match Int.compare hlc1.logical_time hlc2.logical_time with
         | 0 -> String.compare hlc1.node_id hlc2.node_id
         | n -> n)
    | n -> n
    
  let update local_hlc received_hlc =
    let physical_now = Unix.gettimeofday () |> Int64.of_float in
    let max_physical = Int64.max local_hlc.physical_time 
                                  received_hlc.physical_time in
    let new_physical = Int64.max physical_now max_physical in
    
    let new_logical = 
      if new_physical = local_hlc.physical_time &&
         new_physical = received_hlc.physical_time then
        (max local_hlc.logical_time received_hlc.logical_time) + 1
      else if new_physical = local_hlc.physical_time then
        local_hlc.logical_time + 1
      else if new_physical = received_hlc.physical_time then
        received_hlc.logical_time + 1
      else
        0
    in
    
    { physical_time = new_physical;
      logical_time = new_logical;
      node_id = local_hlc.node_id }
end
```

**TrueTime API 风格的设计**

受 Google Spanner 启发的时间不确定性处理：

```ocaml
module BoundedTime = struct
  type time_bound = {
    earliest: float;
    latest: float;
  }
  
  (* 获取当前时间的上下界 *)
  let now () =
    let current = Unix.gettimeofday () in
    let uncertainty = estimate_clock_uncertainty () in
    { earliest = current -. uncertainty;
      latest = current +. uncertainty }
    
  (* 等待直到确定时间已过 *)
  let wait_until timestamp =
    let rec wait () =
      let bound = now () in
      if bound.earliest > timestamp then
        Lwt.return_unit
      else
        let* () = Lwt_unix.sleep 0.001 in
        wait ()
    in
    wait ()
end
```

### 10.2.7 分布式事务与一致性 (Distributed Transactions)

虽然搜索系统通常避免分布式事务，但某些场景仍需要：

**两阶段提交 (2PC)**

```ocaml
module TwoPhaseCommit = struct
  type transaction_id = string
  type participant_vote = Commit | Abort
  
  type coordinator_state =
    | Preparing
    | Committing
    | Aborting
    | Completed
    
  module Coordinator = struct
    let execute_2pc participants transaction =
      (* Phase 1: Prepare *)
      let* votes = 
        participants
        |> Lwt_list.map_p (fun p ->
          timeout (prepare_transaction p transaction) 5.0)
      in
      
      (* Decision *)
      let decision = 
        if List.for_all ((=) Commit) votes then
          Commit
        else
          Abort
      in
      
      (* Phase 2: Commit/Abort *)
      let* () = 
        participants
        |> Lwt_list.iter_p (fun p ->
          match decision with
          | Commit -> commit_transaction p transaction
          | Abort -> abort_transaction p transaction)
      in
      
      Lwt.return decision
  end
end
```

**Saga 模式**

对于长时间运行的操作，使用补偿事务：

```ocaml
module Saga = struct
  type 'a step = {
    forward: unit -> 'a Lwt.t;
    compensate: 'a -> unit Lwt.t;
  }
  
  type 'a saga = 'a step list
  
  let execute_saga steps =
    let rec run completed remaining =
      match remaining with
      | [] -> Lwt.return (Ok (List.rev completed))
      | step :: rest ->
          Lwt.catch
            (fun () ->
              let* result = step.forward () in
              run ((step, result) :: completed) rest)
            (fun exn ->
              (* 补偿已完成的步骤 *)
              let* () = 
                completed
                |> Lwt_list.iter_s (fun (s, r) ->
                  Lwt.catch
                    (fun () -> s.compensate r)
                    (fun _ -> Lwt.return_unit))
              in
              Lwt.return (Error exn))
    in
    run [] steps
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