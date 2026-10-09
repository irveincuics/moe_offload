本文把前两版候选收敛成可在当前 llama.cpp 分支实现和验收的具体设计。代码核验日期为 2026-09-23。本文中的新接口、配置项和统计字段均是拟实现内容，不是已有功能；本次仅完成设计，未运行新 GPU 性能实验。开题报告与学位层次尚未提供。

主方案：以 Qwen3-30B-A3B BF16、单请求、单 token decode 为初版范围，实现“当前层确定性异步读取 + 按层分配的专家缓存 + 只面向缓存缺失的预算预取”。沿用原 router、完整 Top-8 和现有 CUDA MUL_MAT_ID 内核。Prefill 使用统一的基线策略并计入 TTFT，初版的预测调度仅在明确标记的 decode 阶段启用。

**1. 当前代码提供了哪些入口。**

| 位置 | 已有行为 | 拟修改 |
|---|---|---|
| [llama-model.cpp:1882](/home/cuichengshang/moe_offload/llama.cpp/src/llama-model.cpp:1882) | 遍历各层，注册 gate/up/down 的文件位置 | source 描述增加 layer_id 和 projection，组成每层三矩阵目录 |
| [llama-model.cpp:1206](/home/cuichengshang/moe_offload/llama.cpp/src/llama-model.cpp:1206) | 为 context 的 scheduler 注册 SSD 张量 | 传递明确的层号、矩阵类型和形状，避免运行中反复解析名称 |
| [ggml-backend.cpp:794](/home/cuichengshang/moe_offload/llama.cpp/ggml/src/ggml-backend.cpp:794) | 单 FIFO 工作队列 | 增加需求与预测队列、请求升级、预测并发额度 |
| [ggml-backend.cpp:873](/home/cuichengshang/moe_offload/llama.cpp/ggml/src/ggml-backend.cpp:873) | 对齐临时分配、pread、memcpy 到 vector | 对齐槽位复用；条件满足时直接读入槽位 |
| [ggml-backend.cpp:978](/home/cuichengshang/moe_offload/llama.cpp/ggml/src/ggml-backend.cpp:978) | acquire 提交所有本次缺失，再等待全部 future | 拆成非阻塞提交、按消费者等待、复制生命周期管理 |
| [ggml-backend.cpp:2100](/home/cuichengshang/moe_offload/llama.cpp/ggml/src/ggml-backend.cpp:2100) | 取得真实 Top-K 后选择对应权重 | 在每层首次观察真实路由时提交该层三矩阵，并触发未来层候选生成 |
| [llama-context.cpp:1402](/home/cuichengshang/moe_offload/llama.cpp/src/llama-context.cpp:1402) | 使用 ubatch 生成与计算图 | 增加 eval 上下文：请求、位置、阶段；单 token 不自动等同 decode |

当前 `ggml_backend_cuda_set_tensor_async` 使用普通 CUDA stream 上的 cudaMemcpyAsync；该调用仍会复制权重。当前 graph split 机制会为跨后端权重建立边界，具体 gate/up/down 的 split 必须在 GPU 实际运行中确认，不能仅凭代码假定三者一定能重叠。

**2. 第一项工作是取得可信基线。**

在 GPU 可用的宿主或容器执行当前已有接口的路径检查，例如从 llama.cpp 目录运行：

```bash
mkdir -p /home/cuichengshang/moe_ssd/test/design-v3
GGML_OP_OFFLOAD_MIN_BATCH=1 ./build/bin/llama-bench \
  -m /home/cuichengshang/moe_ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
  -ngl 999 --ssd-moe-offload --ssd-moe-cache-mib 8192 \
  --ssd-moe-cache-workers 4 --ssd-moe-stats 1 \
  -p 0 -n 1 -r 1 --no-warmup -v -o jsonl \
  > /home/cuichengshang/moe_ssd/test/design-v3/path-check.jsonl \
  2> /home/cuichengshang/moe_ssd/test/design-v3/path-check.log
```

容器中相应路径使用挂载的 `/ssd`。环境变量仅用于诊断现有机会性卸载阈值，是否生效以实际算子落点为准。成功条件是 GPU 专家算子路径、非零 demand miss/显式读取/backend_copy 与有效输出；不能仅根据 `gpu_info` 判断。`-v` 与完整 trace 用于诊断，正式性能测试关闭逐节点打印。

这条命令只有一 token，且 llama-bench 的 test_gen 使用随机 token，因此它不是最终工作负载。后续基准驱动必须走正常 llama_decode：先对真实 prompt 预填充，再对固定的真实生成 token 序列逐步重放。每个策略采用同一 token 序列；另做端到端真实采样确认行为。请求开始、KV 重置和实验重复都要显式通知预测状态。

基线分别为全驻留 GPU、mmap 按需、当前 V2、后续异步但没有预测的版本。对比应用缓存策略时，所有实验都使用相同读取方式、计算路径、精度和总预算。真实 SSD 实验必须记录 direct I/O 的实际使用与块设备读量，发生 buffered/mmap 回退时单独标记，不能默默算作同一组。

**3. 定义可检查的数据结构与线程职责。**

以下是设计草图，字段类型以实际工程约束调整：

```text
LayerSource[48]
  layer_id
  projection[gate, up, down]: tensor, file, offset, stride, dtype

ExpertRecord[layer, expert]
  slice[3]: slot_id, state, io_future, copy_epoch
  requested_by: demand / prediction
  last_real_use_step
  real_use_frequency
  prediction_origin_step, target_layer, generation

Slot
  aligned_base, payload_offset, allocated_bytes, valid_bytes
  state: FREE / QUEUED / READING / READY
  pin_count
  generation

IoRequest
  layer, expert, projection, generation
  priority, deadline_estimate, issued_bytes

EvalInfo
  request_id, eval_id, token_position, phase
```

缓存生命周期由推理调度线程管理，I/O worker 只执行预先确定的读任务、写入其专属槽位，并通过 future/完成队列交付结果。Worker 不直接调用 CUDA 或修改主缓存 map。slot 写入与 future ready 之间要建立线程同步关系。

一个物理读取对应一个 `(layer, expert, projection, generation)`。预测任务被真实需求命中时升级同一任务，不能再次提交。已经进入 READING 的槽位不能取消后立即复用，必须等读取完成。过期的 QUEUED 任务可以撤销；generation 防止旧完成通知污染重用槽位。

停止、错误返回和 context 销毁时，先停止提交并收尾 I/O，再释放文件与槽位；还需等待保留源指针的 CUDA 复制结束。future 成功不代表复制已结束。

**4. 版本 V3.1：拆开提交和等待，先优化已经知道的当前层需求。**

拟增加三个概念接口：

```text
submit_expert_slices(layer, ids, projections, priority) -> handles
wait_for_projection(handles, projection) -> ready source spans
record_copy_completion(handles, backend) -> copy epoch/event
```

在本层首次取得真实 Top-K 时，立即登记 gate/up/down 三组请求。在计算图需要 gate 权重时只等待 gate；需要 up 时只等待 up；需要 down 时再等待 down。实际每一组需要的专家必须全部准备好，初版不拆 Top-K 的 GPU kernel。

预期时序为：真实路由已知后并行读入 gate/up/down；gate 就绪后提交其复制与计算，剩余读任务继续运行；up 和 down 同理。是否形成有效重叠，取决于 graph split、读取顺序与 GPU 提交时间，应以 timeline 为准。如果实际 split 在启动计算前要求三组全就绪，需调整该 SSD 路径的 split 边界并更新 allocator 生命周期；不能只改函数名称就宣称实现流水。

初版读取优先级：当前消费者所需矩阵最高，当前层已确定但稍后使用的矩阵次之，未来层预测最低。4 个 worker 起步，首先完全关闭预测，扫描 1/2/4/8 个 worker。预测加入后限制同时在设备读取阶段的预测请求不超过 1 或 2 个；“低优先级队列”不能抢占已经进入 NVMe 的读取。

最保守版本在当前层的全部确定性读取结束后，才释放预测读取额度；随后根据测量再判断是否允许部分并发。这样先保证机制正确，再评价更积极的重叠。

源缓存槽位在最后一次 source-to-CUDA 复制结束后即可解除复制 pin；目的 CUDA workspace 的计算生命周期仍由原调度器管理。当前公共 backend API 有 event_record/event_synchronize/event_wait，没有非阻塞 event_query。最小版本可在已有的同一 backend 同步点释放先前 copy epoch；内存紧张时等待候选槽位所属复制 event。不要凭空调用未实现的 event_query，也不要提前复用 event。

源码中 tensor_set_async 的源目前是普通 vector；其实际异步程度需要测量。初版保留相同分配属性做公平比较，另用小规模缓冲区比较对齐 pageable 与可用的固定页分配，不能未经测量锁定全部 8 GiB。

**5. I/O 槽位布局：先用原 GGUF，避免先制造第二个 60 GB 文件。**

对原始文件 offset o、读取长度 n、实际 direct-I/O 对齐 a，计算：

```text
o_aligned = floor(o / a) * a
prefix    = o - o_aligned
n_aligned = ceil((prefix + n) / a) * a
```

直接将 `n_aligned` 字节读入复用的对齐槽位，payload 指针为 `base + prefix`。这样可以去掉每次 miss 的 bounce-buffer 到 vector 拷贝。槽位实际占用按对齐后的分配计数。

当前 BF16 每个矩阵切片为 3 MiB，原路径还有最多 512 字节的 kernel 保护读取。若已经验证 a=4096，可用 `3 MiB + 8 KiB` 作为单切片的保守槽位大小；其他对齐要求用公式计算。保护字节的来源和语义保持与原实现一致，不能随意改成零。若文件提前 EOF，真实权重范围内短读必须报错，只有合法范围外的保护尾部才按明确规则处理。

只有原 GGUF 直接入槽方案的真实随机读吞吐仍明显受布局影响时，再评估 sidecar。sidecar 至少比较 `[gate][up][down]` 完整专家与 `[gate+up]、[down]` 两段，并包含模型身份、offset、dtype、形状和校验。第一版缓存与预取不依赖 sidecar。

**6. 内存预算给出具体初值，但不把缓存额度误写成整机额度。**

| 项目 | 初始候选 | 含义 |
|---|---|---|
| 专家槽位总预算 | 8192 MiB | 包含已驻留、读取中、预测和被 pin 的源槽位，计实际对齐字节 |
| 需求保留额度 | 至少两个 Top-8 完整专家组的实际槽位字节 | payload 约 144 MiB，加对齐开销；按需可回收 READY 且未 pin 的缓存，防止预测堵死真实需求 |
| 尚未被真实使用的预测占用上限 | 64/128/256 MiB | QUEUED、READING、READY 的预测槽位统一计入；初始 128 MiB |
| 总读取 worker | 4 | 扫描 1/2/4/8 |
| 预测读取并发上限 | 1 | 后续对比 0/1/2，低优先级任务最多占用部分 worker |
| 前瞻距离 | 1/2/4 层 | 窗口不够时跳过，不强行预测；8 层是后续扩展 |
| 每个目标层的候选上限 | 4 个未驻留完整专家 | 最多约 36 MiB 有效权重，仍受总预测空间和字节预算限制 |

这些额度是同一池内的逻辑限制，不能相加为额外分配。需求保留空间可在空闲时被可立即回收的热缓存借用，但不能被长时间 pin 或在途预测占满。每次请求先确保完整请求可容纳，再提交 I/O，避免部分分配后的死锁。

整机预算还必须包含非专家权重、KV、CUDA workspace、分配器开销和其它进程。现有 scheduler 复制张量仍保留完整专家维度，GPU workspace 不等于 Top-8 的 72 MiB。预算依据实际 reserve/分配日志和物理驻留核算，不假设使用 compact GPU buffer。

本地配置下 4096 上下文的 FP16 K/V 纯元素大小约为 384 MiB，但实际分配含对齐和运行时开销。非专家张量、workspace 以运行测量为准。Qwen3-30B BF16 可以放入整机物理内存，因此应明确这是受限预算实验。

**7. 版本 V3.2：先做可复现的分层缓存，再考虑在线调整。**

把热度与保留决策的逻辑单元设为 `(layer, expert)`，保留三矩阵各自的 ready/copy 状态。三个切片全部被正式需求使用后才形成完整命中记录；不能把 gate/up/down 的三次访问误当成三次专家复用。

每层维护 protected/probation 两个队列。第一次被真实使用进入 probation；在后续 token 再次使用则进入 protected；仅预测加载不进入 protected。容量不足时优先淘汰已过期且未使用的预测项，再淘汰 probation，最后淘汰 protected 中按策略选出的未 pin 项。任何正在读取或被复制持有的项不能直接回收。

第一组实验只比较：当前全局 LRU、等额每层缓存、带老化频率的全局策略、按层代价分配。先统一关闭预测，避免多因素混淆。

按层代价分配的具体实现：收集独立校准请求的真实路由轨迹，得到每层在不同容量 c 下的 miss 数 m_l(c)，测量该层读取导致的暴露等待代价 tau_l。构建替代目标：

`J_l(c) = m_l(c) * tau_l / 校准 token 数`。

对每层 c=0..128 个逻辑专家，在总槽位限制下求使 `Σ_l J_l(c_l)` 最小的配额。48 层、约数百到九百个专家容量的规模，可以先离线动态规划；它只对这个可加替代目标最优，不是有共享 SSD 排队的真实系统全局最优。

当预测加入后，以未被及时预取覆盖的 miss 和真实剩余等待重新构建代价；否则容易给已经很好预测的层过多空间。分别对比纯频率配额、浅层偏置配额和该代价配额，确认贡献来自可隐藏性和加载代价，而不是单纯增加热点容量。

运行初版加载离线配额即可。在线版本可以每 256 decode token 基于过去窗口重新提出配额，小步调整且每次仅释放未 pin 的专家；后台计算不阻塞推理线程，新配额在安全边界发布。每次最多迁移少量专家容量，不能瞬间清空大批层。若离线配额相对等额/LRU 没有稳定收益，不升级在线控制器。

**8. 版本 V3.3：高层预取先使用现有专家 ID，不额外回传隐藏状态。**

当前 scheduler 已经为了选择权重读回 Top-K ID。首版预测器直接使用这些 ID，因此不依赖训练网络或新增每层 GPU 隐藏状态同步。它属于条件共激活统计，是否有效必须先做留出轨迹评估。

维护以下统计：

```text
last_ids[layer]                   上一 decode token 的同层 Top-K
recent_count[layer, expert]       最近 32 个 decode token 的出现次数
transition[layer, distance, i, e] 当前层专家 i 出现时，目标层专家 e 的计数
source_count[layer, distance, i]  条件事件的样本数
```

distance 取 1/2/4 时，transition 的上界为 `48*3*128*128` 个 float32，约 9 MiB；仅维护合法目标层会略小。每个目标层真实路由到达时，根据同 token 的较早层路由更新最多 3 组 8×8 共现计数。每隔固定 token 数统一老化，分母与分子同时老化。

对目标层 m=l+d 和专家 e，先使用固定、可消融的评分：

`q(e) = 0.5 * mean_{i in R_l} P(e in R_m | i in R_l) + 0.5 * recent_count[m,e]/W`。

W 是实际有效窗口长度，最多 32。条件概率用目标层基础频率做平滑；样本不足时退回近期频率。0.5 是起始候选权重，不是最优结论；需比较时间历史-only、条件统计-only、混合三者。q 是排序估计，不自动等于校准概率，需通过独立校准数据或仅使用过去信息的在线分桶估计转为可靠性值 p。

训练/校准请求与测试请求分开。测试期间允许在线更新，但只使用已经发生的真实路由，并明确区分冻结预测器与在线适应两种实验。请求切换时清理近期历史；条件统计是否跨请求保留要作为固定配置披露，不能跨测试重复偷偷变热。

一个目标层若被多个前瞻距离预测，只更新同一候选记录的分数和截止时间。实际已驻留或已经在途的专家先去重，只对其余候选考虑新增读取。已缓存专家不能计入预取命中来抬高预测性能。

预测器第一轮必须报告缓存 miss 集合上的 Precision/Recall 和实际可隐藏等待，不能只报告全部 Top-K 上的命中率。如果真实需要读 SSD 的候选基本无法预测，就保持该模块关闭；已有跨层 gate 或轻量训练预测器只作为后续可替换的候选来源。

**9. 每个预测请求必须经过具体的收益和时机判断。**

给候选 e 估计：p 为校准后的需要概率；R 为完整专家剩余读取的边际耗时；H 为本次提前读取预计能隐藏的需求等待；E 为挤出其他缓存的预计损失；P 为预测与提交开销。一个可实现的保守筛选式是：

`gain(e) = p*H - (1-p)*R - E - P`。

只接受 gain>0 且符合空间、队列与字节预算的候选。R 不是每线程耗时的简单求和，应从当前队列深度下的实际服务曲线估计。这是启发式，不宣称理论最优。

例如，假设某候选 p=0.8，提前读可隐藏 2 ms，错读边际成本 2 ms，淘汰损失 0.2 ms，预测提交成本 0.02 ms，则 gain 约为 0.98 ms；p=0.3 时约为 -1.02 ms，应跳过。上述数字仅说明判定方法，不是设备测量值。

时机判断使用 `预计排队 + 该队列深度下的读取 p90 ≤ 目标层剩余时间窗`。时间窗从近期该执行路径的层间时间估计，持续更新，不能固定拿独立 benchmark 的 GPU 时间相加。当预测错过时机但仍可能部分隐藏读取时，第二版再评估这种收益；第一版采用保守规则。

示例假设：9 MiB 完整专家在有效 4 GiB/s 下约需 2.20 ms，计排队后 2.5 ms；若每层可用窗口约 0.8 ms，提前一层、两层都不足，四层的约 3.2 ms 才可能及时。真实窗口和带宽均需测量，所有候选来不及就不提交。

每个 decode token 的预测额外读取预算初始可用“最近 32 token 按需缺失字节均值的 10%”作为保守配置，另扫 0/5%/10%/20%；零历史时关闭预测。接受候选时按其完整剩余字节预扣预算，即使本次提交后来才实际读取也不重复获批。请求长度短、预算小于一个专家时允许完全不预取，不能为保证实验看见预取而偷偷提升预算。

过去窗口变化、预算百分比和分层配额相互影响，所以先固定预测预算对比分层策略，再固定分层策略对比预算控制，最后才组合；每种组合都记录实际新增字节和原需求读取变化。

**10. 执行伪代码明确到触发位置。**

```text
on_begin_eval(info):
    校验请求/位置/阶段；只为适用的 decode 激活预测
    收割完成通知；处理过期但尚未提交的预测任务

on_real_router(layer, ids):           每 eval、每层只执行一次
    用真实 ids 评价先前预测，并更新过去信息统计
    命中在途预测 -> 升级原任务；禁止重复读
    为当前层三个矩阵确定性提交缺失读取
    按距离生成未来层候选，过滤驻留项与重复在途项
    通过收益、时间窗和预算约束 -> 放入低优先级队列

before_weight_consumer(layer, projection):
    提升当前 projection 的读取优先级
    等待真实选中专家该 projection 的全部切片 ready
    pin 源切片；沿用现有 tensor_set_async 提交复制
    记录本次复制的 epoch/event
    沿用现有 graph split 提交计算

at_known_backend_completion():
    释放该 backend 上已确认复制完成的 source pin
    应用安全的配额更新；保留尚未完成读取和复制的槽位
```

trace 记录消费者顺序，以实际 gate/up 的图顺序提升优先级，不硬编码模型一定先 gate 后 up。保留原有 allocator 依赖、destination buffer event 与 backend 同步；优化的是 source cache 生命周期，不能顺手删除保护目的张量复用的同步。

**11. CPU 辅助计算采用独立的小型可行性试验。**

首先让 CPU 承担预测表更新、队列、I/O 与缓存，不立即改写模型计算图。专家计算是否值得实现，先在单层固定 Top-8 微基准测量 GPU 8 个、GPU 7+CPU 1、GPU 6+CPU 2 的完整结束时间，CPU 扫 1/2/4/8 个核并核验大小核绑定。

所有方案使用同一 BF16 权重与输入，计入 CPU/GPU 输出汇合、缓存数据复制、同步和必要格式转换。必须并发实测，不能用两个单独耗时相加或取最大值预测收益。记录 CPU 并发时 GPU 8-expert/部分-expert kernel 是否因 DRAM 竞争变慢。

若单层有稳定收益，再考虑缓存 0/1/2 个 CPU 专家数量对应的图变体，GPU 与 CPU 各执行不重叠的真实专家子集，按原 gate 权重汇合。当前实现没有这个选择性混合路径，不能把设置线程数当作实现。图形状、host 缓存非连续布局和 CPU kernel 输入适配都是额外工作，应单独估算。

若单层含全部开销后无收益，论文仍可报告这个设备条件下的测量结论，主方案继续使用 GPU 专家计算。不能为了满足“CPU 辅助”标题保留负收益模块。

**12. 用哪些指标和实验决定版本是否值得继续。**

新增 trace 最少包含 `request/eval/token/layer/expert/projection`、需求与预测来源、排队/读取开始/读取结束、需求首次到达、需求等待开始与结束、复制 event、分配字节和回退原因。I/O worker 耗时累计与主线程真实等待分开；多个并发 worker 的耗时不能相加为 TPOT 分解。

| 版本 | 改动 | 首要验收 | 不成立时处理 |
|---|---|---|---|
| V3.0 | 路径核验、真实 token 重放、分解统计 | 确认目标 GPU 路径；冷热与真实 SSD 字节可解释 | 先修执行路径，不开始策略比较 |
| V3.1 | 对齐槽位、submit/wait 拆分、三矩阵确定性读取 | 相同正式路由/权重；无源缓冲提前复用；时间线确有重叠或拷贝减少 | 调整实际 split/生命周期；无收益则保留简单实现 |
| V3.2 | 分层缓存与代价配额 | 同预算优于调优的 LRU/频率/等额配额，且代价模型能解释收益 | 不引入更复杂在线配额控制 |
| V3.3 | 条件路由统计、只预取缺失候选、预算门控 | 及时预取减少的阻塞超过额外读取和调度开销 | 关闭预测；定位候选准确率、窗口或带宽限制 |
| V3.4 | 缓存与预取联合校准 | 相比各模块单独最佳参数仍有增益；没有靠扩大容量获益 | 收窄结论为独立模块有效 |
| 独立 CPU 试验 | 固定 0/1/2 个专家分担 | 含同步/合并后优于 GPU-only，且并发测量可重复 | 不整合全模型 |

首轮小规模矩阵：4/8/16 GiB 专家槽位预算；四类真实请求（代码、问答、数学、混合/工具文本）；每类先选 4 个独立请求，每个 256 prompt token、128 decode token，确认方法后增加样本和长上下文。校准数据另外准备，不把这些测试请求用于调参。小规模阶段用于筛选，正式论文样本规模按方差和结论范围扩大。

128 decode token 的初筛只评价固定离线配额与预取，不足以评价每 256 token 才触发的在线配额更新。在线配额必须另用至少 512/1024 decode token 的长请求，并记录实际更新次数、迁移字节和切换后的延迟。

固定 workload 的方法比较使用逐 token 重放，并增加真实生成验证。至少多次交错重复，比较整段总 tokens/总时间、TTFT、TPOT p50/p95、实际 SSD 字节、内存峰值与控制开销。是否超过测量噪声用多次运行的分布判断，不预先保证加速倍数。

策略消融建议顺序：异步基线；加分层缓存；加固定预算预测；加时机/收益门控；组合配额与预测。预测器消融为历史-only、条件统计-only、混合。加入最相关已有策略的可用实现或忠实适配，不能只击败朴素 LRU 就声称领先。

日志和 JSON 继续写 `/home/cuichengshang/moe_ssd/test`。详细 trace 写入批量缓冲，正式计时关闭逐请求文件同步，另测 trace-on/off 开销。所有新增参数在实现前都只属于设计，现有命令行不能直接使用。

**13. 论文贡献如何表述才与实际工作一致。**

V3.1 属于必要实现基础。拟研究的两个方法贡献是：利用层间不同的缺失代价与预取可隐藏性分配缓存；利用真实可用路由统计，对缓存缺失候选进行带字节、时机和队列约束的预取。每一项都要通过强基线、消融和适用条件证明，不能把接口拆分或新评分名称当作创新。

ProMoE 已研究预测提前量和调度协作，Fate 已研究跨层预测与浅层缓存，SeqMoE 已研究截止时间调度和预测驱动淘汰。因此本设计的优势目前只能称为实现可行性；新颖性要在单 NVMe、统一物理预算和实际 miss 集合下的具体决策差异上继续核验。[ProMoE](https://arxiv.org/html/2410.22134v1)、[Fate](https://arxiv.org/html/2502.12224v2)、[SeqMoE](https://arxiv.org/abs/2609.12978)

建议先完成 V3.0 与 V3.1 的数据，再决定把 V3.2 或 V3.3 哪一项发展为主贡献。可行的研究路线允许某个候选在实验中失败，不应把未验证的多个机制全部写进必须完成的论文目标。
