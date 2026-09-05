# 项目归档笔记：意图已出、发键已接、焦点未切

**日期：** 2026-09-03  

**工程：** `d:\Desktop\T\test`  

**配套文档：** `CONTEXT_HANDOFF.md`（下一对话先看第 1 节）· `FSM_DESIGN.txt`（现行草案）· `DECISIONS.md`（§1.0 架构）· `TRAIN_ALIGN.md`（组集）

---

## 1. 核心想法

过图不是回放死脚本，也不是行为克隆。现行是**模块化 agent**：YOLO+OCR 感知 → FSM 决策出意图 → 快捷栏执行层发键。

三层必须分开，不能糊成一块：

| 层 | 做什么 | 不做什么 |
|----|--------|----------|
| 核心 `fsm_core.py` | 纯函数 `step` → `TAP` / `HOLD` / `CAST` / `PICK` | 不读文件、不发键、不 sleep |
| 回放 `fsm_replay.py` | 绿=状态、蓝=意图、黄=人按了什么 | **永远不发键** |
| 实机执行 `fsm_execute.py` | 只在 FSM测试勾「发键」时把意图变成 `SendInput` | 不改决策 |

开打数据从录像里抽成**按房间**的过图技能特征，不是全图共用一套。CD 跨房保持，只在该图【重置】点清空。假释放用杀 MON 效率 < F% 标注，不进序列。

写盘也拆开：PNG 在 `images/<时间戳>/`；`keys.jsonl` + `frames.jsonl` 在 `recordings/<地下城>/<时间戳>_<角色>/`。采集必须当场跑 YOLO，否则没法提技能特征。

---

## 2. 演进逻辑 / 技术路线

### 2.1 本会话覆盖的旧约定

1. **开打/捡物**：txt 六步 + 捡物方法先进核心出蓝字，发键另等点头 → **发键已接线**（仅 FSM测试勾选）。  
2. **采集路径**：曾只写 `images/` → PNG 仍在 images，jsonl 按原格式回 `recordings/`。  
3. **采集 YOLO**：曾「采集不跑 YOLO」→ **必须当场推理**，`frames.jsonl` 恢复 `player_xy` 等旧字段；`t_ns` 仍在截图完成后、YOLO 前打。  
4. **参数同步**：不要单独同步按钮。回放改旋钮写 `_fsm_replay_ui.json`；主面板点开始**先读文件**，避免过期旋钮盖掉。FSM测试跑着 json mtime 变了会重载。  
5. **捡物参数**：`LM/LC` → `PM/PC`，另加 `PT`（连续 N 帧位移不超过 PM 才算停下，机制同 M/L/G）。  
6. **下一对话任务**：原先「等用户方案再改回放」→ **先修日志刷屏 + 发键焦点**。

### 2.2 当前主链路

```
感知：截图完成 → 打 t_ns → YOLO（solarwarden_b）
    → fsm_core.step(snapshot, ctx, params)
         开打：该房序列 − CD → 距≤范围 CAST（再 hold_frames 等待），否则接近
         捡物：LOOT 在动等 PT 帧停下；数量>PC → PICK(左Alt)；否则依次走近
         前进：左右 TAP 再 HOLD（疾跑）；上下第一帧 HOLD
    → [仅 FSM测试且勾「发键」] fsm_execute.FsmExecutor.apply
         TAP/HOLD 方向；CAST 用键位表 command（+ 先后、空格同时）；PICK=alt_l
回放：同一 step，只画蓝字，不发键
```

注入走 `key_inject.py`（扫码 SendInput，含扩展键方向键）。须管理员。点停止先 `executor.stop()` 松开我们按下的键。

### 2.3 本会话还落地的盘面

- 回放 GUI：「当前数量」改成预览左上半透明字；参数行拆开以免右栏裁切「相似 S%」。  
- 键位：`skill_binds/SolarWarden.json`。CAST 复合指令步间 `CAST_STEP_S=0.02`。  
- 缺检测的旧 png 段可用 `backfill_session_yolo.py` 补框。  
- 特征表可能仍无 `map_tags`：有重置证据也不会在相似房清 CD，须再提取。

---

## 3. 待解决的问题

**下一对话先做（用户已定）：**

1. **状态日志刷屏**：FSM测试每帧 `gui.log("FSM#258 14ms | 前进 | …")`，0.05s 一帧会淹。去掉，或 5～10 秒一条（改状态/意图再打也可以）。顶栏已有实时字。YOLO# 每帧日志一并节流。出处 `status_analysis_module.py` `_run_yolo_frame`。  
2. **按键没进游戏**：用户怀疑点「开始」后焦点没切回 DNF。开始后没有 `SetForegroundWindow`；每帧 `cv2.imshow` + `waitKey(1)` 更可能抢焦点。可复用 `skill_bind_tool.focus_point`；对齐 `region` 常带 `hwnd`。不要把发键写进核心/回放。

**仍挂着、本轮不做：**

3. 【重置】未进 `map_tags`，开打不会在相似房清 CD。  
4. 组集清洗未确认（player fill、相对坐标）；脚本 vs `TRAIN_ALIGN` 打架先问。  
5. DECISIONS 第 4 节（并行卡住 / 不对称迟滞 / `gate.x > player.x`）未落地，不准覆盖 `FSM_DESIGN.txt`。  
6. OCR 回城 `town_return` 未进快照；CAST 时序手感不对再调 `CAST_STEP_S`。  
7. 修 YOLO 写盘时 `images/` 与部分 recordings 子目录曾变空，原因未证实；勿当已恢复。

---

## 4. 怎么接着干

新对话读 `CONTEXT_HANDOFF.md` 第 1 节。修好后重启主面板，FSM测试 + 勾「发键」再测。没方案不要自己加回放功能。
