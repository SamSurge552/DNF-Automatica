# 项目归档笔记：A 类裁定 / 提取 lag / t_ns / 点按 80–120

**日期：** 2026-09-06 16:00

**工程：** `d:\Desktop\T\test`

**配套：** `FSM_DESIGN.txt`（现行）· `CONTEXT_HANDOFF.md` 第 1 节 · 此前 `ARCHIVE_NOTE_2026-09-06-0.md`

---

## 1. 本轮做了什么

上午 11:30 那份归档之后，同一天后半段。核心仍以 txt 为准，没有另开架构。

1. **A1 / A2 / A3 / A5 落地**（A4 卡住恢复方向未改）。回城只 `tn_s` 秒；补正平移逻辑点（jsonl/叠图仍原始）；OCR 无关键词沿用上一帧 raw；捡物等停下上限 **PW**（默认 3000ms）。
2. **提取 `i_feat`：** 技能起点仍黄字 onset `i0`（hold / CD / `t0` 不变）。特征快照 `i_feat = min(i0+press_lag_frames, i_end, last)`，默认 lag=1。`skill_hold` 轨不预平移（禁止双计）。
3. **`press_lag_frames` 可调：** `_fsm_replay_ui.json`；回放「过图技能特征」控件。同一值：提取 `i_feat` + 黄字整列延后。绿/蓝/红/叠图/YOLO 框不延后。lag=0 与旧黄字同帧。
4. **回放参数区：** 拆成多个 LabelFrame（判定 / 卡住 / 过门 / 开打 / 连按·移动 / 捡物·回城 / 其它 / 补正）。每行 ≤2（短名最多 3）。只纵向滚。
5. **`t_ns`：** `capture` 在 grab 得到像素后立刻打戳，再 save、再 YOLO。jsonl 用该戳。PNG 后台写，停录 `flush_saves`。键钩、OCR 线程、回放对齐公式未改。
6. **点按：** 每次 TAP/CAST/连按每次/PICK = `uniform[tap_ms_min, tap_ms_max]`，默认 **80–120**。`_tap_down_up` 一条路。HOLD / ATTACK 仍按住。PICK 节流 = `tap_ms_max × 5`。json 字段 `tap_ms_min`/`tap_ms_max`；旧 `tap_ms` 忽略。

改补正或 lag 后须 **重提 `skill_features`**。本项点按 / t_ns 不改旧 jsonl。

---

## 2. 已验证 / 试验中 / 已撤销

**已验证（代码已落地，按裁定写完）：**  
A1/A2/A3/A5；`i_feat` + 黄字共用 UI lag；参数分类布局；`t_ns` 在 grab 后（探针：dxcam 连抓含异步 save、无 YOLO，dt 中位约 8.5ms，不再叠整段 PNG 编码）；点按抽时长在 80–120 且有波动，HOLD 无 sleep。

**试验中（实机手感未当事实）：**  
- lag=1 在 ~165–177ms 采集段等于跳一整帧墙钟，可能过冲；未擅自 lag=2。  
- 点按 80–120 在游戏里的手感、改 min/max 后跟变：代码路径通，未在本会话做键钩实测。  
- 现场采集 dt 仍含 YOLO。  
- A4 卡住恢复方向仍挂。  
- 叠图是否像素级对齐、`th_ms`/过门手感仍是边打边调。

**已撤销：**  
特征快照用黄字 `i0` 同帧画面；`capture()` 整段返回（含 PNG save）后才打 `t_ns`；点按写死 50ms / 移动 TAP 瞬时 down+up；参数塞满单行靠加宽窗口。

---

## 3. 下一对话先做

继续 **边打边调**，不要另开架构。

- 改过补正 / lag 的地下城：**重新提取本图**。  
- FSM测试看点按是否在 80–120 有波动；HOLD 方向和普攻 X 应仍按住。改 min/max 后点开始会读共用 json。  
- 回放叠图核对 YOLO。调 GX/GY/`th_ms`。范围异常仍释放。  
- 不要把连按塞回 `FsmParams`。不要为省事把 save 挪到 `t_ns` 前。

---

## 4. 不要当现行的旧条

| 旧 | 新 |
|----|----|
| 特征用 `views[i0]` | `i_feat = i0+press_lag_frames`；hold/CD 仍 `i0` |
| 黄字跟真实 press 同帧 | 黄字整列延后同一 lag；轨本身不改 |
| `t_ns` 在 capture 返回后 | grab 后立刻打；save/YOLO 在后 |
| `tap_ms` 默认 50；移动瞬时抬起 | min/max 默认 80–120 随机；HOLD/ATTACK 按住 |
| 同图提取无条件扫 FSM_TEST | 默认只采集；勾「含FSM测试」才并入 |
| 回城 TN 帧换算 | 只 `tn_s` 秒 |
| 补正只给 FSM | 逻辑点同一补正；jsonl 原始 |

---

## 5. 怎么接着干

新对话读 `CONTEXT_HANDOFF.md` 第 1 节。打开 `FSM_DESIGN.txt`。磁盘代码是真相。边打边调；组集 fill/相对未确认，不要顺手改导出。
