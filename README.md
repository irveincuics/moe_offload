# MoE SSD Offload on NVIDIA GB10

本项目基于 `llama.cpp`，研究 Qwen3 MoE 在 NVIDIA GB10 统一物理内存平台上的 SSD 专家权重卸载、缓存、零复制和预测预取。

本文档是顶层仓库的项目入口，用于维护：

- 每个分支的目标、代码状态和实验结论；
- 按日期记录已经完成的优化及效果；
- 区分已验证结果、探索性结果、负结果和尚未提交的实验；
- 给出复现实验时应使用的分支及证据位置。

最后更新：**2026-11-03**。

## 当前推荐版本

- 稳定开发入口：`dev@5e8dc72`。
- 基础 SSD 卸载与零复制基线：`ssd_off@e8f3c34`。
- 当前已提交的预取组合：`dev` / `prefetch_1028`，包含有界 BGu 预取和 BGuF Prefill 预取。
- CPU 隐藏状态预取已提交到 `prefetch_cpu@7bc267b`，并已快进合入和推送到远程 `dev@7bc267b`；`prefetch_cpu` 分支本身尚未推送。
- 顶层 `ssd_off` 工作树正在移除 M0 专用测试工具；该清理尚未提交，远程 `ssd_off` 仍包含 M0。

状态含义：

| 状态 | 含义 |
| --- | --- |
| 已验证 | 真实 30B BF16 模型完成正常路径数值检查和对应实验 |
| 功能通过 | 构建、单元测试或路径检查通过，但没有完整性能验收 |
| 探索性 | 有初步真实性能数据，但样本、机器控制或基线不足以形成普遍结论 |
| 否决 | 当前数据不支持作为默认方案继续采用 |
| 设计中 | 只有设计、工作树改动或尚未提交的实验 |

## 分支维护表

下表覆盖当前所有本地分支。分支新增、合并、废弃或获得新实验结果时，必须同步更新本表和后面的时间线。

| 分支 | 代码基线 | 用途与主要改动 | 当前效果与结论 | 状态 |
| --- | --- | --- | --- | --- |
| `main` | `cb5e030` | 初始工程与上游 `llama.cpp` 快照 | 未包含 SSD MoE 优化，作为仓库起点 | 基线 |
| `ssd_off` | `e8f3c34` | mmap 按需专家、有界 SSD cache、异步需求读取、整专家打包、GB10 地址表零复制 | P1 平均 TPOT 降低 29.39%；P2 相对 P1 平均 TPOT 再降低 17.36%，专家 payload backend copy 降为 0。当前本地另有未提交的 M0 清理 | 已验证 |
| `ssd_off_m0` | `e8f3c34` | 保留 M0 路径、数值和生命周期观测工具的冻结分支 | 用于诊断和验收，不把同步 trace 结果作为性能结论 | 功能通过 |
| `test/ssd-prefetch-a-20261012` | `e8f3c34` | 历史层间 A 预取实验工作树 | A 平均 TPOT 232.49 ms/token，对照 none 为 231.42，未获得净收益 | 探索性 |
| `test/ssd-prefetch-c-20261012` | `e8f3c34` | 层内 C 预取实验工作树 | C 平均 TPOT 242.98 ms/token，较 none 退化约 5% | 否决 |
| `test/ssd-prefetch-ac-20261012` | `e8f3c34` | A+C 联合预取和统一对照驱动 | AC 平均 TPOT 232.52 ms/token，未优于 none；保持无预取为默认 | 探索性 |
| `test/ssd-prefetch-b-20261013` | `e8f3c34` | B1/B2/BG 路由预测与投机预取 | 部分方案降低延迟，但额外 SSD 读量大且存在慢轮，未形成默认方案 | 探索性 |
| `test/ssd-prefetch-b-uma-20261013` | `913982b` | BGu：目标层前保护有界投机槽，并关闭低价值跨 token 预测 | TPOT 229.91 -> 222.43 ms/token，配对改善 3.31%；Decode 读量约增加 2.69%，TTFT 无稳定改善 | 探索性推荐 |
| `test/ssd-prefetch-cache-20261014` | `e8f3c34` | BGu 上的 SLRU 缓存替换实验 | 相对 LRU 的吞吐、TPOT、P99 增量区间均跨 0；保留预取感知 LRU | 否决 |
| `prefetch_prifill1029` | `0301290` | BGuF：Decode BGu 加批量聚合的 Prefill 路由预测 | 相对 none，TTFT 均值降低 3.62%、P99 降低 6.83%，但 Decode 吞吐降低 8.92%，端到端生成吞吐降低 3.12%；仅完成单轮探索 | 探索性 |
| `prefetch_1028` | `5e8dc72` | 合并有界 BGu 与 BGuF Prefill 预取 | 本地比远程 `origin/prefetch_1028` 领先 2 个提交；作为已提交预取组合来源 | 功能通过 |
| `prefetch_cpu` | `7bc267b` | 用独立 CPU 线程执行与 GPU 方案同公式的隐藏状态 router 投影 | 三轮探索中，相对 none：TTFT 降低 6.76%、Decode TPOT 降低 9.10%、端到端生成吞吐提高 8.27%；正确性一致。分支本身尚未推送，但提交已进入远程 `dev` | 探索性 |
| `dev` | `7bc267b` | 当前最佳远程开发分支，包含有界预取、BGuF Prefill 预取和 CPU 隐藏状态预取 | 已与 `origin/dev` 同步；CPU 预取正确性和三轮探索结果已记录 | 推荐开发分支 |

注意：多个 `test/*` 分支的实验代码保存在独立工作树中，分支指针仍可能停在共同基线。表中的实验结论来自冻结结果，不代表对应提交已经完整包含工作树代码。

## 按日期的优化记录

### 2026-09-20：建立工程基线

- 分支：`main`。
- 建立顶层工程、Docker 配置和 `llama.cpp` 源码快照。
- 该版本不包含 SSD MoE 专家卸载。

### 2026-09-22：V1/V2 SSD 专家按需卸载

- 提交：`e81ef7b`，后续进入 `ssd_off`。
- V1 将 Qwen3 MoE routed experts 保持为文件映射，router 产生 Top-K 后只读取所选专家。
- V2 引入应用级有界 cache、LRU、direct I/O、多读取 worker、同步需求和异步需求流水。
- 增加 `--ssd-moe-offload`、cache 容量/worker 参数和 `llama-bench` 输出。
- 本阶段建立正确数据路径和强基线，没有单独记录可作为最终结论的冻结性能数字。

### 2026-10-08 至 2026-10-09：P1 整专家打包

- 分支：`ssd_off`。
- 将 gate/up/down 重组为每个 `(layer, expert)` 一条 4096 字节对齐记录。
- 三个投影共享一次读取、一个 cache entry、一个 future 和一个生命周期。
- 对齐 direct I/O 直接写入 cache，消除 bounce buffer 到 cache entry 的 payload copy。

真实 Qwen3-30B-A3B BF16 结果：

| 指标 | 原 SSD cache | P1 打包 | 变化 |
| --- | ---: | ---: | ---: |
| Direct read 请求 | 465,270 | 155,055 | -66.67% |
| `host_bounce_bytes`，三轮累计 | 1.331 TiB | 0 | -100% |
| 平均 TTFT | 12.60 s | 11.36 s | -9.82% |
| 平均 TPOT | 408.74 ms/token | 288.62 ms/token | -29.39% |

6,144 条记录逐字节校验通过；resident、普通 SSD 和打包 SSD 的保存 logits 最大绝对误差为 0，greedy token 一致。此阶段仍保留 1,746,749.49 MiB 的专家 payload backend copy。

### 2026-10-09：P2 GB10 地址表零复制

- 分支：`ssd_off`。
- 在支持 pageable host page table 的 CUDA 设备上，MMF kernel 直接读取应用 cache 中的专家权重。
- 每个投影只更新约 1 KiB 专家地址表，不再复制所选专家 payload 到 CUDA 临时权重张量。
- cache entry 保持占用直至真实 CUDA 消费结束，防止异步执行期间被淘汰复用。

三轮四请求结果：

| 指标 | P1 打包 | P2 零复制 | 变化 |
| --- | ---: | ---: | ---: |
| 平均 TTFT | 11,451.76 ms | 10,159.04 ms | -11.29% |
| 平均 TPOT | 295.94 ms/token | 244.57 ms/token | -17.36% |
| Decode 速度 | 1.000x | 1.210x | +21.01% |
| 四请求端到端时间 | 62,628.46 ms | 54,519.43 ms | -12.95% |
| 专家 payload `backend_copy` | 1,746,749.49 MiB | 0 MiB | -100% |
| 地址表控制复制 | 0 MiB | 32.0625 MiB | 新增控制开销 |

8 组正式比较共 18,536,192 个 logits 与同版本 resident 基线逐元素一致，最大绝对误差为 0；`zero_copy_fallbacks=0`。结论仅适用于本次 GB10、BF16 MMF 和短负载，不能直接外推到离散 GPU。

### 2026-10-12：A、C、AC 历史预取

- 分支：`test/ssd-prefetch-a-20261012`、`test/ssd-prefetch-c-20261012`、`test/ssd-prefetch-ac-20261012`。
- 使用历史路由相似性，提前 4 层、至多 4 个非驻留候选。
- 平均 TPOT：none 231.42、A 232.49、C 242.98、AC 232.52 ms/token。
- A/AC 虽减少部分等待，但增加读取量且无净加速；C 明显退化。
- 决策：不作为默认策略，继续寻找有截止期和容量约束的方案。

### 2026-10-14：BGu 有界投机预取

- 分支：`test/ssd-prefetch-b-20261013`、`test/ssd-prefetch-b-uma-20261013`。
- 修复投机记录被需求 LRU 提前淘汰的问题；保护有界槽位至目标层，并关闭最后四层低准确率跨 token 预测。
- BGu 平均 TPOT 从 229.91 降至 222.43 ms/token，配对改善 3.31%，95% CI 为 2.17% 至 4.85%。
- Decode 吞吐提高约 3.45%，端到端吞吐提高约 1.09%；Decode 读量约增加 2.69%，全流程读量增加约 0.74%。
- TTFT 无稳定改善；实验只有四条短请求，仍需独立请求、长生成和受控机器负载确认。

### 2026-10-14：SLRU 缓存策略

- 分支：`test/ssd-prefetch-cache-20261014`。
- 采用 probation/protected 两段缓存，protected 上限为 50%。
- 离线 hit rate 从 58.83% 提高到 59.96%，但在线 none 模式仅减少 0.32% Decode demand miss。
- 吞吐、TPOT 和 P99 的 95% CI 均跨 0。
- 决策：否决 SLRU 作为默认策略，保留 BGu + 预取感知 LRU。

### 2026-10-29：BGuF Prefill 预取

- 分支：`prefetch_prifill1029`，随后合入 `prefetch_1028` 和 `dev`。
- 对 Prefill chunk 的路由预测按专家支持数聚合；最佳测试配置为 distance=4、count=16、slots=64。
- 10,505 次预测中 10,130 次 ready、29 次 late、346 次 unused。
- 相对 none，TTFT 均值降低 3.62%、TTFT P99 降低 6.83%。
- Decode 吞吐降低 8.92%，端到端生成吞吐降低 3.12%，磁盘读取增加 1.35%。
- 决策：只在 TTFT 优先时作为候选；Decode score 同步和稳态吞吐仍需优化。该性能对照只有单轮四请求，不是正式服务级结论。

### 2026-10-29：合并预取开发线

- `prefetch_1028@5e8dc72` 合并有界 BGu 和 BGuF Prefill 预取。
- `dev@5e8dc72` 从 `ssd_off` 快进到该已提交组合，作为当前开发入口。
- `origin/prefetch_1028` 仍停在较早的 `34eb9d2`；远程 `dev` 已是 `5e8dc72`。

### 2026-11-02 至 2026-11-03：CPU 隐藏状态预取

- 分支：`prefetch_cpu@7bc267b`；代码、报告和驱动已于 2026-11-03 提交，并通过快进合并进入远程 `dev@7bc267b`。
- 固定与 GPU 方案相同的预测公式，使用已有 router 权重；独立 CPU 线程执行隐藏状态投影，预测只影响提前加载，不改变真实 router、Top-K 或模型计算。
- 三轮随机顺序、每方案 12 次请求执行；正确性组完整 logits 逐字节一致，所有正式运行均保持 P2 零复制生效。

| 指标 | 无预取 | GPU BGuF | CPU 隐藏状态异步 |
| --- | ---: | ---: | ---: |
| 平均 TTFT | 10,199.29 ms | 9,438.32 ms | 9,509.68 ms |
| Decode TPOT | 231.98 ms | 218.74 ms | 210.87 ms |
| Decode 吞吐 | 4.311 tok/s | 4.572 tok/s | 4.742 tok/s |
| 端到端生成吞吐 | 1.673 tok/s | 1.795 tok/s | 1.811 tok/s |

CPU 相对无预取：TTFT 降低 6.76%、Decode TPOT 降低 9.10%、Decode 吞吐提高 10.01%、端到端生成吞吐提高 8.27%。CPU 相对 GPU 的 TTFT 增加 0.76%，Decode TPOT 降低 3.60%。该结果只覆盖四类短请求和固定 8 GiB 预算，尚未验证长上下文、多缓存容量或并发服务。

## 正确性和性能口径

所有后续优化至少应记录：

1. 日期、分支、提交或工作树状态；
2. 模型、设备、Docker、缓存额度、worker、输入和运行顺序；
3. 与同版本 resident/none 基线的 logits、greedy token 和 NaN/Inf 检查；
4. TTFT、Prefill 吞吐、Decode TPOT/吞吐、端到端吞吐及尾延迟；
5. SSD 物理读取、需求等待、ready/late/unused、payload/control copy 和 fallback；
6. 样本数、重复轮次、置信区间、机器状态和适用范围；
7. 负结果、失败轮次和被否决方案，不得只保留有利数字。

性能运行应关闭同步 trace、权重 readback 和 logits 写盘；正确性运行不得使用旧版本 logits 代替同版本基线。统一物理内存不等于没有复制、同步或资源竞争，必须用实际计数和时间线验证。

## 分支更新规则

每次修改任一分支时：

1. 更新“分支维护表”中的代码基线、用途、状态和结论；
2. 在“按日期的优化记录”追加日期、实现、正确性、性能、限制和下一步；
3. 如果结果尚未提交，明确标记“工作树实验”，不能写成远程分支能力；
4. 如果远程落后本地，明确写出远程停留的提交；
5. `dev` 只合入能够复现、正确性通过且收益/代价已有记录的提交；
6. 实验分支保留负结果，确认不再使用时再标记废弃，不直接删除证据。

新增分支时复制下面的记录模板：

```text
分支：
日期：
基线提交：
目标与假设：
主要改动：
正确性：
性能与资源：
失败与限制：
状态：设计中 / 功能通过 / 探索性 / 已验证 / 否决
证据位置：
下一步：
```

## 详细文档

- [SSD MoE 持续优化总记录](docs/ssd-moe-optimization-log.zh.md)
- [整专家打包验证报告](docs/ssd-pack-validation-20261008.zh.md)
- [预取策略选择与实验协议](docs/gb10-moe-prefetch-options-20261012.zh.md)
- [后续研究设计](docs/gb10-moe-research-design-20261102.zh.md)
- [具体实现方案](docs/gb10-moe-concrete-design.zh.md)
- [实验计划](docs/gb10-moe-thesis-experiment-program.zh.md)

冻结模型、二进制、日志和机器可读结果位于 `/ssd/test/` 与本地 `results/`。这些目录中的大文件不进入 Git；README 只记录可复核的摘要和对应路径。
