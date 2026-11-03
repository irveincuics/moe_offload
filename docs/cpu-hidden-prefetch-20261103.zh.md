# CPU 隐藏状态预取

分支：`prefetch_cpu`，基于 `prefetch_1028@5e8dc728`。本次保留 SSD 装载、零复制和缓存生命周期，替换 CPU 预测器。旧的在线转移矩阵策略改名为 `cpu_history`，新的 `cpu` 默认使用隐藏状态投影和独立预测线程。实验范围是 Qwen3-30B-A3B BF16、48 层、128 专家、Top-8，暂未泛化到其他 MoE 结构。

## 预测与执行

对第 L 层归一化并乘以 norm 权重后的隐藏状态 h，计算：

`score[L+d, t, e] = dot(router[L+d, e], h[L, t])`

默认 d=4，不跨 token 回绕。使用已有模型的 router 权重，无额外训练，不是在线转移概率模型。公式与无 adapter 的 GPU BGuF 相同；ARM NEON 与 GPU 的浮点累加顺序不同，不承诺预测分数逐位相同。预测只决定专家提前装载，不修改正式路由或模型计算。

1. 从计算图权重建立只读 FP32 router 副本，每个请求上下文初始化一次，44 层合计 44 MiB。
2. 只在精确命名为 `ffn_norm-L` 的张量就绪后获取快照。排除 `(reshaped)`、`(view)`，避免一次层计算重复预测。
3. 独立的单个 CPU 线程完成矩阵投影；aarch64 使用 NEON FMA。`cpu_async=false` 可切换同步消融。
4. 调度线程在后续隐藏状态或专家需求处轮询完成结果，进行候选筛选和 SSD 请求提交。预测线程不操作缓存。
5. 预测队列最多保留 d 个任务；跨步或已到目标层的结果不再提交。上下文销毁时先等待预测线程退出，再销毁其快照和权重。

隐藏状态获取仍需等待 GPU 写入完成。统一内存不等于没有数据依赖或同步成本；`score_sync_us`、`snapshot_us` 和 `compute_us` 分别记录等待、快照与投影开销。CPU 计算可与 GPU 后续工作重叠，但排序和提交仍在调度线程。

## Prefill 与 Decode

| 项目 | Prefill | Decode |
|---|---|---|
| 输入 | 当前 chunk 每个 token 的隐藏状态 | 当前 token 的隐藏状态 |
| 候选 | 每 token Top-8；相对第 8 名的 margin 达到 0.25 | Top-8；相同 margin 门限 |
| 排序 | 跨 token 支持数，其次累计 margin | router 投影分数 |
| 每次最多提交 | 16 个非驻留专家 | 2 个非驻留专家 |
| 未消费预取槽位总上限 | 64 | 4 |

以上均可通过配置调整。支持数聚合能体现当前 chunk 的专家需求，但当前并未做 token 采样、带宽反馈控制或在线学习门限。

## 缓存与统一内存

单个 packed BF16 专家记录约 9.004 MiB，包含其相关专家权重片段。Prefill/Decode 的未消费预取占用上限分别约 576/36 MiB，是专家缓存内部额度，不是额外分配一套缓存。

- 所有对照固定 `cache_mib=8192`、`history_reserve_mib=64`，因此专家池为 8128 MiB。CPU 的 44 MiB router 副本计入预测器预留；64 MiB 不是全进程内存硬上限，模型其余权重、KV 和 backend workspace 仍需另计。
- 预取跳过已驻留专家；未使用项以低 LRU 优先级进入缓存，并在目标层前获得现有的短期保护。
- 正式需求命中未完成的预取时提升其 SSD 队列优先级并等待；命中后转为普通缓存项、更新 LRU。
- 到期未使用项在 I/O 完成且没有 GPU 使用引用后回收。不得提前复用 GPU 仍在读取的内存。
- `zero_copy=true` 启用现有 P2 路径，GPU 直接使用专家缓存中的数据；运行结果必须同时检查 `zero_copy_binds>0`、`backend_copy=0`、`zero_copy_fallbacks=0`。

## 推荐配置

在现有 `llama-moe-m0` JSON 的 model、pack、input、output 等字段之外设置：

```json
{
  "baseline": "async",
  "prefetch": "cpu",
  "cpu_async": true,
  "zero_copy": true,
  "cache_mib": 8192,
  "history_reserve_mib": 64,
  "prefetch_distance": 4,
  "prefetch_count": 2,
  "cpu_decode_prefetch_slots": 4,
  "prefill_prefetch_count": 16,
  "prefill_prefetch_slots": 64,
  "b_margin": 0.25,
  "workers": 4,
  "batch": 32,
  "threads": 4
}
```

`cpu_decode_temporal_weight` 现在仅供 `cpu_history` 使用。已有 `prefetch=cpu` 的配置将采用新算法；需要复现旧实验时必须显式改为 `cpu_history`。新 CPU 模式不使用 `history_bank`，也没有实现 GPU 方案的 `b_adapter` 修正。

## 实验口径

复用 `moe-llama-dev`，aarch64 / NVIDIA GB10，驱动 580.173.02，容器 Ubuntu 24.04.4 LTS。构建命令为 `cmake --build /ssd/test/prefetch-cpu/build -j4 --target llama-moe-m0`。实验驱动保存于分支的 `scripts/cpu_hidden_prefetch_bench.py`，运行副本位于 `/workspace/results/prefetch-cpu-hidden-20261102/run.py`；最终数据使用 `--series final`，避免和早期重复触发版本混用。

在宿主机复现时使用新系列名，依次执行：

```bash
docker exec moe-llama-dev python3 /workspace/results/prefetch-cpu/worktree/scripts/cpu_hidden_prefetch_bench.py correctness --series reproduce --modes none cpu cpu_sync bguf
docker exec moe-llama-dev python3 /workspace/results/prefetch-cpu/worktree/scripts/cpu_hidden_prefetch_bench.py timing --series reproduce --modes none bguf cpu --rounds 3
```

脚本记录可执行文件和动态库 SHA256，并检查 logits、性能运行的 greedy token、零复制状态和每层预测次数。`report.json` 的 `counters` 对计数/时长求和，对驻留内存/router 副本取最大值，命中率由总 hits/misses 重新计算。

正确性使用 `input-4.json`，保存所有输出步的完整 logits 与无预取比较。性能使用 `input-32.json`，关闭 logits 文件写入，每轮随机排列方案顺序。四个请求分别是代码、数学、问答和摘要；每轮 109 个输出 token、105 个 Decode 间隔。TTFT 从 prompt 执行开始到首个输出就绪，未包含模型启动加载。吞吐按生成 token 计算，不把输入 token 混入端到端吞吐。

每个请求新建上下文与专家缓存，不清 OS page cache。日志检查直接读取、零复制、预测任务数、ready/late/unused、同步与计算成本。三轮重复不等价于十二个独立工作负载；TTFT P99 在该小样本上近似最大值，只能用于探索性比较，不能说明服务级尾延迟。

## 最终结果

三轮随机顺序，三个方案使用同一可执行文件和动态库。每个方案汇总 12 次请求执行、327 个输出 token、315 个 Decode 间隔。下表均由原始请求/间隔合并计算；P99 使用 nearest-rank，不是三次 P99 的平均值。

| 指标 | 无预取 | GPU BGuF | CPU 隐藏状态异步 |
|---|---:|---:|---:|
| 平均 TTFT（ms）↓ | 10199.29 | 9438.32 | 9509.68 |
| TTFT P99（ms）↓ | 16973.52 | 15440.13 | 15519.50 |
| Prefill 吞吐（输入 token/s）↑ | 12.869 | 13.906 | 13.802 |
| Decode TPOT（ms）↓ | 231.98 | 218.74 | 210.87 |
| Decode 间隔 P99（ms）↓ | 686.77 | 651.09 | 648.60 |
| Decode 吞吐（输出 token/s）↑ | 4.311 | 4.572 | 4.742 |
| 端到端生成吞吐（输出 token/s）↑ | 1.673 | 1.795 | 1.811 |

CPU 相对无预取：TTFT 降低 6.76%，Decode TPOT 降低 9.10%，Decode 吞吐提高 10.01%，端到端生成吞吐提高 8.27%。CPU 相对 GPU：TTFT 增加 0.76%，Decode TPOT 降低 3.60%，Decode 吞吐提高 3.73%，端到端生成吞吐提高 0.90%。结论是本工作负载上已达到 GPU 方案附近的效果，Decode 略好，Prefill 略差；不据此宣称跨工作负载或服务级统计显著优势。

逐轮 Decode TPOT：

| 轮次 | 无预取（ms） | GPU（ms） | CPU（ms） |
|---|---:|---:|---:|
| 0 | 231.10 | 217.47 | 212.77 |
| 1 | 233.99 | 220.46 | 210.42 |
| 2 | 230.87 | 218.28 | 209.44 |

正确性四组 `none/cpu/cpu_sync/bguf` 的每一步完整 logits 均逐字节一致；九组性能运行的 greedy token 也全部一致。`cpu_sync` 仅做正确性消融，没有将保存 logits 的短测试用作正式性能证据。所有运行均零复制生效，无 payload backend copy，无零复制回退。CPU 三轮总计 16368 次预测，精确符合每层每步一次；过期任务与退出时 pending 均为 0。

## 为什么已接近 GPU，以及剩余瓶颈

原 CPU 策略使用专家 ID 转移矩阵，丢掉了隐藏状态中可供未来 router 判别的信息；新版本与 GPU 使用相同投影和候选筛选。独立线程使投影不必串行阻塞推理线程，NEON 降低了点积成本。CPU 三轮投影累计约 2.05 秒、快照约 0.36 秒；投影时间可与 GPU 重叠，不能直接把这两个数字相加解释为推理额外延迟。

| 三轮合计 | GPU BGuF | CPU |
|---|---:|---:|
| Decode issued | 9357 | 10709 |
| Decode ready / late / unused | 8163 / 0 / 1194 | 9385 / 2 / 1322 |
| Decode demand 等待（s） | 46.66 | 42.89 |
| Prefill issued | 31494 | 37495 |
| Prefill ready / late / unused | 11290 / 19160 / 1044 | 10671 / 25768 / 1056 |

相同预测公式不意味着相同提交数量：CPU 在异步结果收取时提交，可能赶上需求消费释放的投机槽位，改变可提交的非驻留候选数。Decode 得到了更多及时命中，但 Prefill 的 late/issued 从约 60.8% 上升到 68.7%，ready 数反而下降。更多 issued 或更低 demand miss 并不保证更低 TTFT；late 项最终仍被使用，也不能全部计为无效流量，评估额外 SSD 流量应比较总物理读取量。

后续最值得验证的改动是：

1. 在当前层需求 I/O 完成后及时收取 CPU 结果，减少等到下一个隐藏状态回调的提交延迟；以 ready、TTFT 和 P99 验证，而非只看预测准确率。
2. Prefill 按预测完成时间、SSD 排队和目标层剩余时间分配槽位，减少不能及时完成的投机读取。当前仍是固定 16/64。
3. 保持同步开销可见。CPU 三轮隐藏状态相关同步累计约 8.50 秒；这是等待 GPU 当前计算完成的观测时间，不能全部视为新增开销，但仍说明统一内存并未消除依赖等待。需要设备时间线来区分本来就必须等待的部分与回调额外破坏的重叠。

这些后续项尚未实现或计入本轮收益。当前结果仅涵盖单请求 replay、四类短 prompt 和固定 8 GiB 预算，尚需长上下文、其他缓存容量及并发请求验证。

原始报告：`/workspace/results/prefetch-cpu-hidden-20261102/final/report.json`；汇总：`/workspace/results/prefetch-cpu-hidden-20261102/summary-final.json`；容器日志及输出：`/ssd/test/prefetch-cpu-hidden-20261102/final`。早期目录下不带 `final/` 的结果包含重复 reshape 回调，不能作为当前代码的性能证据。
