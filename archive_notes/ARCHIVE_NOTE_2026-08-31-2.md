# 会话归档：标签对齐、负样本与五类检测默认

**日期：** 2026-08-31  

**工程：** `d:\Desktop\T\test`  

**配套文档：** `blueprint.txt`（意图）· `CONTEXT_HANDOFF.md`（交接）· `TRAIN_ALIGN.md`（录制→过图对齐）

---

## 1. 核心想法

过图策略学的是「这一眼结构化状态下，人在这段真实 dt 里按住了多久」，检测学的是「整窗里有什么」。两件事不要混：键频次是**该图的打法**，读图/黑屏是**检测的负样本**。

- **过图标签：** 磁盘只留事件流；导出只留每键 `held_frac`。50ms 一帧后 `count` 与上升沿重合，已冗余，去掉。
- **采集 vs 导出：** jsonl 保留重复特征帧（FSM 用「卡住」= 同一状态持续多久）。过图训练集才丢 dup。不要在录制端按特征去重。
- **分布：** 深渊 `right`≈24%、其它键 <4% 是正确行为，不是采样没采够。同样通关再录十段比例不该变；给罕见键加权等于逼模型多按左。
- **检测负样本：** solarw 190 张全是正样本、player 每张都有。blank（读图/黑屏、空 txt）才是背景类。加进 **train**、不要进 val，才能和「只训游戏图」公平对比。
- **骨架冻结：** YOLO 支持 `freeze=10`，适合从 COCO 预训练小数据。但已有很强的五类权重时，**不冻 + 暖启动 + blank** 优于从 yolo26n 冻骨架重训。
- **对齐验对：** 不要看训练 loss。把标签平移 K 帧，看 val R² 是否单峰（`shift_align_check.py`）。

---

## 2. 演进逻辑 / 技术路线

### 过图采集与标签（本会话落地）

1. 默认间隔 **0.05s**（dxcam 墙钟中位 ~8ms 撑得住）。
2. 导出脚本 `export_dataset.py`：player fill、去 auto-repeat、真实 dt 上的 `held_frac`；第一帧丢掉；过短段跳过。
3. 特征重复帧：**仅导出**丢弃并计 `dup_skipped`。
4. 开录 `GetAsyncKeyState` **一次** → `keys_held_at_start`，不是每帧轮询。
5. `count` 删除。正向验证：K=-10…+10 小模型曲线。
6. 否决「补录左/上/下/技能来均衡」和「罕见键加 loss 权重」。

### 检测数据与训练（本会话落地）

1. `Atrain` 改为 `Aset/<集>` + `Xout/<集>`；`dnf_detect_v7` 改名为 `solarwarden`。
2. `aset_balance.py` 随时出正负/各类表。blank 已有空 txt。
3. 阶梯按**类出现率从高到低加图**被否：YOLO 吃整图，solarw 每张都有 player，拆不开「先人后怪」。
4. 合理阶梯只剩：冻不冻、加不加 blank。四模同 val（38 张 solarw）：

| 模型 | 做法 | val mAP50 |
|------|------|-----------|
| solarwarden | 旧五类，不冻，无 blank | 0.841 |
| **solarwarden_b** | 上者续训，不冻，train +blank | **0.882** |
| SOLAR_F | yolo26n + freeze=10，仅 solarw | 0.744 |
| SOLAR_FB | freeze=10，solarw+blank | 0.762 |

5. 默认过图权重已切 **`solarwarden_b`**（`gui_module` / `yolo_engine`）。`train_yolo.py` 的 `FREEZE=10` 仍给自动标注小数据；五类续 blank 用 `--freeze 0 --weights solarwarden/best.pt`。

### 实机录制

`20260831_180010`：0.05s 视觉正常（~100s / 1849 帧），**未管理员，keys=0**，不能进过图集。

---

## 3. 待解决的问题

1. **重启 GUI** 才会默认 `solarwarden_b`；用 YOLO测试核对游戏内框 + 读图/黑屏误检。
2. **`config.json` 间隔曾被改成 0.5s**，正式录制宜 0.05s 并点应用。
3. **管理员重录** 0.05s 通关；旧 0.3s 有键的三段可试导出，正式集应重录。
4. `shift_align_check` 要在新数据上再跑；旧偏 `right`、0.3s 段 R² 为负，不能当真。
5. **过图小模型 / 自动化 / FSM** 均未实现。FSM 读原始 jsonl。
6. **`mon` 仍弱**；blank 仍少（19 张）。继续加负样本时 val 仍应只留游戏帧。
7. 自动标注 `varien` 在另一对话；勿当过图 YOLO。

---

## 4. 新对话怎么接

把 `d:\Desktop\T\test\CONTEXT_HANDOFF.md` 交给助手。对齐以 `TRAIN_ALIGN.md` 为准。
