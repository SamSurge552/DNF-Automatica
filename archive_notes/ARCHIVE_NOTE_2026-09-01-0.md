# 会话归档：模型视角回放、抽象 FSM 图标、分层控制

**日期：** 2026-09-01  

**工程：** `d:\Desktop\T\test`  

**配套文档：** `blueprint.txt`（意图）· `CONTEXT_HANDOFF.md`（当天状态）· `TRAIN_ALIGN.md`（对齐）· `DECISIONS.md`（架构理由）· `FSM_DESIGN.txt`（串行草案）

---

## 1. 核心想法

过图闭环要拆成「眼睛看见什么」和「这一步该干什么 / 手怎么动」，回放工具必须站在**模型输入**上看，而不是站在玩家屏幕上看。

- **模型只拥有结构化标签。** 一帧就是 `player_xy`（可空）+ `mon/loot/gate/boss` 计数。没有 PNG、没有贴图语义。用游戏实拍当 FSM 图标，等于偷看像素；那就不如录制时直接存图。FSMICON 必须是**手绘抽象符号**（人、红脸怪、金币、右箭头门、星形 Boss），和 jsonl 字段一一对应。
- **分层，不是串行开关。** YOLO 给特征；FSM 给画面外的慢状态（goal：开打 / 捡 / 前进）；按键模型学「这一眼里手按住了多久」。错误结构是 `if state==FIGHT: net(features)`；正确结构是 `goal = fsm(features); keys = net(features, goal)`。goal 是压缩历史，不是剥夺网络决策权。
- **YOLO+FSM = 该干嘛；按键 = 干嘛了。** 训执行层仍用导出后的 `held_frac`（0/1 会糊点按与长按）。连发 press 只在导出丢掉，原始 `keys.jsonl` 原样留给回放和卡住统计。
- **卡住只能在原始时间线上数。** 同一 `(player_xy, mon, loot, gate, boss)` 持续了多久 = 卡住。因此录制禁止按特征去重；`dataset_export/` 是过图训练集，FSM / `fsm_replay.py` 不准读它。
- **感知统一、策略分图。** 一个五类检测器（`solarwarden_b`）；OCR 读图名选路线/FSM，不选 detector。`varien_t` 只做自动标注单类 player，禁止当过图 YOLO。
- **能快就别开 HDR。** HDR 下 FP16+色调映射 ~98ms，撑不住 0.05s。用户已关 Windows HDR，走 dxcam 8bit。颜色对但慢的补丁留着，默认不要再开。

---

## 2. 演进逻辑 / 技术路线

### 本会话落地

1. **OCR：** 框要含右上角地名+小地图（约 225×210）；间隔与视觉 0.05s 分离；conf 默认 0.95，可运行中改。热路径放大/防抖已回撤。
2. **录制：** 一图一段；`solarwarden_b`；不存 PNG。2026-09-01 深渊段里 **仅 `20260901_183607_SolarWarden` 有键**（535 帧 / 1242 键），其余三段 `keys=0` 不能进过图集。
3. **FSM 回放：** `python fsm_replay.py` 读原始 `frames.jsonl`，抽象图标+计数+仅 player 点的场地+草案状态+stuck+keys。`make_fsm_icons.py` 生成几何图；回放启动会覆盖 `fsm_icons/`。
4. **草案状态** 目前仍是 `FSM_DESIGN.txt` 串行五步（怪/Boss→开打→loot→gate→回城）。`DECISIONS.md` 第 4 节才是下一版该写的：卡住提到并行、状态超时、进入/退出迟滞、只选 `gate.x > player.x`。尚未写成代码。

### 已定、尚未写进 FSM 代码的路线

```
并行监视器（可打断任何状态）: player 不动 N 帧 / 状态超时 → RECOVERY
主流程带迟滞: 进图 → FIGHT → LOOT → ADVANCE（右门）→ 等待
RECOVERY / 回城: 硬编码，不经过按键网络
FIGHT/LOOT/ADVANCE: 只作为 goal 喂网络
```

阶段一通关标准（可测）：连续 10 次深渊，≥8 次完整回城、无人干预。先薄垂直切片，再按死因加 goal、滑窗、双击跑。

过图小模型仍：单帧特征 → MLP → `held_frac`（BCE）→ 阈值按键。导出才 fill player / 去 dup / 去连发。

---

## 3. 待解决的问题

1. **把回放对照刷图后的直觉写成可执行 FSM**，对齐 DECISIONS 第 4 节，不要停在串行五步。
2. **空键段**（183520 / 184549 / 184640）不要进过图集；admin 硬拦截（开录数秒无键则中止）仍未做。
3. **`export_dataset.py` + `shift_align_check.py` 尚未在 183607 段上跑**；`count` 分布、对齐峰值、人类延迟 Δ 都还没测。
4. **`mon` 真实召回无法事后审计**（不存 PNG）。这是 FSM 开打条件的核心输入。
5. DECISIONS 里仍待做：player fill 上限 5 帧、`meta.completed`、抽样调试图、回归集、成功率脚本、RECOVERY 序列。
6. 过图小模型 / 自动化未开始。`config.json` 的 `ocr_interval` 现为 0.5，不要误改视觉间隔。
7. `Aset/varien` 空自动标不能当负样本；HDR 过曝图不要再训 `varien_t`。

---

## 4. 新对话怎么接

把 `d:\Desktop\T\test\CONTEXT_HANDOFF.md` 交给助手。架构以 `DECISIONS.md` 为准，对齐以 `TRAIN_ALIGN.md` 为准。图标继续抽象，不要换回实拍。
