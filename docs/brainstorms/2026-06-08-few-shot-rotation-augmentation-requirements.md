---
date: 2026-06-08
topic: few-shot-rotation-augmentation
---

# Few-Shot Rotation Augmentation Requirements

## Summary

增加一个 MVTec、MPDD、ViSA 共用的 few-shot 数据集模式：`FEW_SHOT_ROOT` 设置后成为完整有效数据集根目录，并完全替代 `DATA_ROOT`。每个选中类别的 `train/good` 中有几张正常图就使用几张，在 preprocessing 前扩展成 `0/90/180/270` 四种图像视图，再进入 normal scoring、hard-sample search、enhancer 训练和同 root 下完整测试集评估。

---

## Problem Frame

当前 server pipeline 主要面向 full-data 跑法，并且 MVTec/MPDD 路径仍默认依赖已有或搜索到的 Dinomaly checkpoint。few-shot 实验会把完整数据集重组到一个新 root 下：训练集只保留少量 `train/good`，但 `test` 和 `ground_truth` 仍保持完整。这个实验不应该再用 `K` 这样的额外参数描述 shot 数，也不应该把 few-shot 训练 root 和另一个 full-data 评估 root 混用。

---

## Key Decisions

- **Few-shot root 完全替代 dataset root。** `FEW_SHOT_ROOT` 设置后，训练、base checkpoint 训练、hard samples、enhancer 和 eval 都从这个 root 读取。
- **目录内容定义 shot 数。** pipeline 不配置 K，只使用每个选中类别 `train/good` 中实际存在的图片。
- **ViSA 只支持预处理后的 1cls 布局。** 输入默认是 Dinomaly 官方 `prepare_visa.py --split-type 1cls` 生成的 MVTec-like 结构。
- **Few-shot 模式强制训练新的 unified base checkpoint。** 不复用 `CHECKPOINT_PATH`，也不搜索已有 unified checkpoint；仍沿用官方 Dinomaly recipe 中的 DINOv2 pretrained encoder 初始化。
- **旋转是正常图像视图。** `0/90/180/270` 扩展正常覆盖，标签仍为 normal，不作为伪异常。
- **旋转进入 hard-sample search。** rotated normal views 参与 normal score statistics 和候选搜索，以更多生成成本换 few-shot 覆盖。

---

## Requirements

**Dataset root behavior**

- R1. 当 `FEW_SHOT_ROOT` 设置时，它必须完全替代 `DATA_ROOT`，覆盖 dataset indexing、base training、normal training image loading、hard-sample generation 和 evaluation。
- R2. 当 `FEW_SHOT_ROOT` 未设置时，现有 `DATA_ROOT` 行为必须保持兼容。
- R3. 有效 root 必须使用 MVTec-like 类别布局：`<category>/train/good`、`<category>/test/*`、`<category>/ground_truth/*`。
- R4. 行为必须覆盖 MVTec、MPDD、ViSA 三个 dataset dispatch。

**Dataset coverage**

- R5. MVTec 必须继续支持当前 15 类类别集合和现有 full/smoke 展开逻辑。
- R6. MPDD 必须继续支持当前 6 类 unified 类别集合和现有 full/smoke 展开逻辑。
- R7. ViSA 必须支持官方预处理后 1cls 的 12 类集合：`candle`、`capsules`、`cashew`、`chewinggum`、`fryum`、`macaroni1`、`macaroni2`、`pcb1`、`pcb2`、`pcb3`、`pcb4`、`pipe_fryum`。
- R8. 本版不从原始 ViSA 数据和 split CSV 生成 `VisA_pytorch/1cls`。

**Few-shot normal source**

- R9. pipeline 必须使用每个选中类别 `train/good` 下所有支持的图片文件，不再需要 shot-count 设置。
- R10. 选中类别的 `train/good` 为空时，必须在 hard-sample generation 前给出清晰错误。

**Base checkpoint training**

- R11. few-shot 模式必须训练新的 unified base checkpoint，不复用 `CHECKPOINT_PATH` 或搜索到的已有 checkpoint。
- R12. 新 checkpoint 必须基于有效 root 的 selected categories 训练，并保存到当前 run 的输出约定路径。
- R13. few-shot 模式下的 stage 行为必须能表达 base training、hard samples、enhancer 和 eval 的完整顺序。

**Rotation augmentation**

- R14. pipeline 必须为正常训练图生成 `0`、`90`、`180`、`270` 度旋转视图。
- R15. rotated views 必须保留 normal label、source category、source path 和 rotation metadata。
- R16. rotated views 必须参与 normal score statistics、hard-sample search 和 enhancer 训练。
- R17. evaluation images 和 masks 在本版不能做旋转增强。

**Caching and reproducibility**

- R18. hard-sample 和 enhancer cache validation 必须包含有效 dataset root、dataset name、categories、base checkpoint、few-shot active state 和 rotation angles。
- R19. full-data run、不同 `FEW_SHOT_ROOT`、不同 rotation angles、不同 base checkpoint 之间的 cache 不能静默复用。
- R20. run summary 必须记录 `FEW_SHOT_ROOT` 是否启用、effective data root、dataset、categories、rotation angles、base normal image 数、rotated normal view 数和 base checkpoint training source。

---

## Key Flows

- F1. Few-shot full run
  - **Trigger:** 用户设置 `FEW_SHOT_ROOT` 并运行 server pipeline。
  - **Steps:** pipeline 解析 effective root，按 dataset 展开类别，从 effective root 训练新的 unified base checkpoint，读取每类 `train/good`，生成四向旋转 normal views，基于这些 views 生成 hard samples，训练 enhancer，并在同 root 的 test split 上评估。
  - **Outcome:** 输出和指标描述同一个 few-shot root 下的完整实验，不混用 root 或旧 checkpoint。
  - **Covers:** R1, R4, R9, R11, R14, R16, R20

- F2. Existing full-data run
  - **Trigger:** 用户不设置 `FEW_SHOT_ROOT`。
  - **Steps:** pipeline 使用现有 `DATA_ROOT`、checkpoint resolution 和 dataset 行为。
  - **Outcome:** 现有 MVTec 和 MPDD 命令继续工作，ViSA 在新增 dataset 配置下也能走同一类 full-data 入口。
  - **Covers:** R2, R4

- F3. ViSA few-shot run
  - **Trigger:** 用户设置 `DATASET=visa` 和指向 `VisA_pytorch/1cls` 风格目录的 `FEW_SHOT_ROOT`。
  - **Steps:** pipeline 使用 ViSA 类别集合和 MVTec-like loader，读取 `test/good|bad` 与 `ground_truth/bad`。
  - **Outcome:** ViSA few-shot 训练和评估不需要原始 CSV prepare 步骤。
  - **Covers:** R7, R8

---

## Acceptance Examples

- AE1. Given `FEW_SHOT_ROOT` 中 `bottle/train/good` 有两张图片, when run selects `bottle`, then hard-sample generation 在 candidate search 前看到八个 normal views。
- AE2. Given `FEW_SHOT_ROOT` 设置且包含 `bottle/test` 和 `bottle/ground_truth`, when evaluation runs, then metrics 来自 `FEW_SHOT_ROOT` 而不是 `DATA_ROOT`。
- AE3. Given 旧 cache 来自 `DATA_ROOT` 且没有 rotations, when few-shot rotation run starts, then cache 被拒绝或重新生成，而不是静默复用。
- AE4. Given `FEW_SHOT_ROOT` 未设置, when existing MVTec or MPDD runner is used, then training、checkpoint resolution 和 evaluation root 行为保持当前兼容。
- AE5. Given `DATASET=visa` 且 root 是预处理后的 `VisA_pytorch/1cls`, when selected category is `candle`, then pipeline 能读取 `candle/train/good`、`candle/test/good|bad` 和 `candle/ground_truth/bad`。
- AE6. Given few-shot mode is active and `CHECKPOINT_PATH` points to an old checkpoint, when stage is `all`, then run summary shows a newly trained base checkpoint was used.

---

## Success Criteria

- few-shot runs 能仅凭数据目录内容和 run summary 复现 shot 数、rotated view 数、dataset root 和 checkpoint 来源。
- 现有非 few-shot MVTec 和 MPDD server 命令保持兼容。
- ViSA 能通过预处理后的 1cls root 进入同一套 server pipeline。
- 输出指标清楚代表 effective root 的完整 test set。
- cache reuse 不会跨 full-data、few-shot root、rotation 配置或 base checkpoint。

---

## Scope Boundaries

- 测试时旋转平均 deferred。
- 小角度或任意角度插值旋转 deferred。
- rotation-generated pseudo-anomaly labels out of scope。
- 原始 ViSA + CSV prepare out of scope。
- pixel metrics 继续来自 base Dinomaly anomaly map；本改动不把 enhancer 变成 pixel-level segmenter。
- 本版不随机初始化 DINOv2 encoder；“从头训练”指不复用已有 Dinomaly checkpoint，仍沿用官方 DINOv2 pretrained encoder 初始化。

---

## Dependencies And Assumptions

- `FEW_SHOT_ROOT` 指向完整 MVTec-like dataset root，不只是训练子集。
- ViSA 输入已经由官方 prepare 脚本整理为 `VisA_pytorch/1cls` 风格。
- 每个选中类别都保留 `train/good`、`test` 和 `ground_truth` 目录。
- 四向离散旋转是目标 few-shot 实验可接受的 normal views。

---

## Sources

- `llm_das_dinomaly/pipelines/server_mvtec.py` 是当前 server stages、base checkpoint training、hard-sample generation、enhancer training、cache context 和 run summary 的主线。
- `llm_das_dinomaly/data/mvtec.py` 和 `llm_das_dinomaly/data/mpdd.py` 定义了当前 MVTec-like dataset indexing contract。
- `third_party/Dinomaly/dinomaly_visa_uni.py` 和 `third_party/Dinomaly/prepare_data/prepare_visa.py` 表明 Dinomaly 的 ViSA 路径使用预处理后的 `VisA_pytorch/1cls` MVTec-like 数据。
- `configs/server_mvtec.yaml`、`configs/server_mpdd.yaml`、`scripts/run_server_mvtec.sh` 和 `scripts/run_server_mpdd.sh` 提供了现有 env-driven server 配置模式。
