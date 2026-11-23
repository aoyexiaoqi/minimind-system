# 自有模型部署与持续迭代方案（定版 V1.0，2026-11-23）

三方产出：GPT 出草稿（gpt-plan-draft.md）→ Gemini 审查（gemini-plan-review.md）→ 豆包核断整合。审查意见采纳情况逐条标注。

## 一、目标与基底
- 模型：minimind 64M（hidden 768、8 层、中文），定位为**文本续写模型**，不是指令对话模型；验收分两层（能稳定生成／指令能力），不混淆。
- 基底权重 v1.0：InstaBox 上一轮复跑的 Phase B 检查点（600 步、loss 终点 5.1380，三方复跑中最低），执行时先核验文件完整性，不合格则改用 Colab Phase B 检查点。
- 代码基线：本机 ~/workspace/minimind 克隆，执行时记录其 git HEAD 作为代码版本锚点；权重与代码版本解耦管理。

## 二、三节点架构（采纳 GPT 草稿）
- **Colab（T4）**：训练/续训、固定集评估、导出发布包；成果存 Google Drive（原生挂载）。
- **InstaBox（4 vCPU / 7.8 GB / 无 GPU）**：CPU 推理服务，公开发布端口 8080（3000 已被 AIHOT 占用）；闲置 30 分钟暂停为平台特性，不绕过；只提供按需推理，不承诺 24 小时在线。
- **本机 VM**：控制中心——代码仓库、模型注册表、发布调度、历史归档；是 Drive 与 InstaBox 之间的中转站。
- 不引入 K8s / MLflow 等重型平台；文件制注册表＋清单（manifest）管理版本。

## 三、权重流转（采纳 Gemini 修正 2、4）
- Colab 训练完成 → 发布包存 Drive。
- VM 从 Drive 拉取发布包（走已连接的 Google Drive 通道），校验 SHA256 后经已验证的终端通道推送到 InstaBox，落地后再验一次 SHA256。
- **禁止**让 InstaBox 无头直拉 Drive（OAuth/限流/大文件确认弹窗会使闭环断裂）。

## 四、权重格式与检查点分治（采纳 Gemini 修正 3、改法 3）
- **发布包只用 safetensors**：model.safetensors＋config.json＋tokenizer.json＋tokenizer_config.json＋manifest.json（数据 SHA、代码 HEAD、超参、步数、评估报告路径）。禁止以 .pt 作发布格式（pickle 反序列化风险＋跨版本兼容问题）。
- **训练恢复包与发布包严格分开**：恢复包（含模型＋优化器＋调度器＋随机状态＋trainer_state）仅供 Colab 断点续训，每 500 步或 15 分钟保存一次，先写本地临时文件、校验后改正式名；发布包由导出脚本剥离优化器状态生成。

## 五、推理服务（GPT 草稿＋Gemini 修正 1、5 与改法 1）
- FastAPI＋原生 PyTorch CPU、FP32、单 worker；优先复用 minimind 上游自带服务脚本（执行时先核查仓库 scripts/）。
- **进程托管不用 systemd**（InstaBox 沙箱 PID 1 大概率非 systemd，执行时先实测确认）：用 supervisord 或 PID 文件包装脚本托管，要求休眠唤醒后自动拉起。
- **线程限制**：torch.set_num_threads(2)、set_num_interop_threads(1)（4 vCPU 下为生成与 HTTP I/O 留头寸，实测后可调）。
- 接口：GET /health/live、GET /health/ready、GET /v1/models、POST /v1/generate。生成接口 Bearer Token 认证（令牌由豆包生成、只存 InstaBox 环境文件与 VM 注册表，不入聊天与记忆）。
- 并发：asyncio.Semaphore(1) 串行生成＋有界排队（队列 ≤5、等待 ≤15 秒，超时才返回 429）。
- 参数基线（采纳 Gemini 改法 5）：输入 ≤384 tokens、max_new_tokens 默认 128、总窗口 ≤512；评估用贪心解码；业务默认 temperature 0.7、top_p 0.85、repetition_penalty 1.1。
- V1 不做量化（FP32 权重仅约 256 MB，内存不是瓶颈）；若实测吞吐 <20 tok/s，下一阶段转 llama.cpp/GGUF，不用 PyTorch 动态 INT8。
- 版本切换：候选权重先用**独立校验脚本**离线加载冒烟（采纳 Gemini 改进 1，不起第二个 HTTP 进程），通过后原子替换 current 符号链接（os.replace）、重启服务、/health/ready 确认新版本；失败自动指回 previous 并验证旧版，InstaBox 本地只保留 current＋previous 两版。

## 六、版本规范（采纳 Gemini 改进 2）
- 模型产物版本：`日期-阶段-run序号`（如 20261123-pretrain-run03），注册表 registry/versions.json 关联 git HEAD、数据 manifest SHA、评估结论、发布时间与回滚指向。
- 代码（服务与脚本）单独用 SemVer。二者不混用。

## 七、迭代闭环与评估门禁（采纳 GPT 草稿＋Gemini 改进 5）
1. 数据准备：JSONL 校验（可解析、字段、数量、SHA256、去重、空记录）；训练/验证按稳定规则切分；**固定提示集去污染**：与训练语料做 n-gram 重合检查、高重合样本剔除。
2. 训练：固定配置与种子；六步固定流程（环境恢复→数据校验→加载/续训→训练→保存评估→归档打 COMPLETE 标记）。
3. 三层评估：①固定验证集 loss/PPL，相对线上版恶化 ≤2%；②固定提示集 100 条（续写 40／常识 20／格式 20／领域 20）确定性解码，关键类别不低于基线 5 个百分点；③工程稳定性（空输出、NaN/Inf、特殊 token 泄漏、异常重复、超时、内存）全过。另留 20 条人工盲测题只用于重要版本复核、不用于调参。
4. 发布：门禁全过→VM 中转分发→离线校验→原子切换→线上冒烟；任一步失败即拒发、保留线上版、问题归档。
5. 回滚：previous 指针一键回滚并验证，回滚失败必须如实标故障、不宣称成功。

## 八、实施步骤与验收
- **阶段 A（服务上线）**：建 minimind-system 仓库骨架（configs/、eval/prompts.jsonl、serving/、deploy/、registry/）；核验 InstaBox 现有检查点并导出为 safetensors 发布包 v1.0；InstaBox 部署服务并完成托管与冷恢复验证。验收：外部经 8080 发布地址带令牌调用 /v1/generate 成功；连续 20 次短请求全返回合法结果；/v1/models 返回 v1.0；暂停唤醒后服务自恢复且版本不变。
- **阶段 B（迭代闭环跑通）**：Colab 笔记本标准化（分治检查点＋导出＋归档）；跑一次短迭代（约 100 步）产出候选版→过评估门禁→VM 中转发布→InstaBox 切换新版本→再回滚演练一次。验收：注册表有完整两版记录；线上版本按预期切换且冒烟通过；回滚后恢复 v1.0 并验证。
- **阶段 C（沉淀）**：交付运维说明（迭代操作清单、调用示例、冷启动重试示例：指数退避探活、业务与探活接口解耦——采纳 Gemini 改进 4）。

## 九、风险与取舍
- InstaBox 暂停机制决定其只能按需服务；冷启动 30–60 秒由调用端退避吸收。
- Colab 会话断裂靠恢复包＋先本地后同步的保存纪律对冲；恢复连续性（步数与优化器状态）须在阶段 B 实测验证，不以「能加载权重」冒充。
- 单机单并发是免费资源下的合理取舍；吞吐不达标时走 GGUF 路线而非堆 PyTorch 参数。
- 发布包全程 SHA256 双验（VM 拉取后、InstaBox 落地后各一次）。
