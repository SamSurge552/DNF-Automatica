# 会话归档：模块化 agent、纯函数核心、前进/卡住意图

**日期：** 2026-09-02  

**工程：** `d:\Desktop\T\test`  

**配套文档：** `blueprint.txt`（长期意图）· `CONTEXT_HANDOFF.md`（当天状态）· `TRAIN_ALIGN.md` · `DECISIONS.md` §1.0 · `FSM_DESIGN.txt` · `.cursor/rules/new-idea-wins.mdc`

**下一对话：** 等用户把 `FSM_DESIGN.txt` 的开打方法写完再接核心。不要猜技能循环、不要发键。新对话入口仍是 `CONTEXT_HANDOFF.md`。

---

## 1. 核心想法

当前要做的是 **模块化 agent**，不是从录像克隆按键时序。

- **感知 → 决策 → 执行。** YOLO+OCR 看画面，FSM 判状态并给出意图，快捷栏/方向键由执行层落地。网络（过图小模型）是远期，不是现在。
- **核心一份纯函数 + 两个宿主。** `fsm_core.step(snapshot, ctx, params)` 不截屏、不发键、不读文件、不 `sleep`、不调墙钟。回放宿主喂 jsonl 的 `t_ns`；实机宿主在截图完成后、YOLO 前打 `t_ns`。不要拆三个独立进程，也不要把发键写进回放。
- **决策是枚举，不是拼出来的字符串。** `{state, flags, action, move_dir}`。卡住/预热进 **flags**，只用 `in` 判断。`t_ns` 必须单调不减。
- **防抖必须因果。** 只看当前帧和 ctx，不向后看。开头 `max(M,L,G)` 帧打 `WARMUP`。400 帧 deque 是把回放批处理误搬到实机，已删。
- **held_frac 改判。** 技能键废弃（瞬发，只要按下时间戳）。移动键是执行参数「按住毫秒」，不是动作编码。组集分母已是真实 dt，`held_ms = held_frac × dt_s × 1000`；jsonl 不用重录。
- **前进不是双按。** 朝最近门（截屏坐标 y 向下，主轴四向）；第一帧点按，之后按住；穿门后锁原方向再走 **A** 帧，不因人到门另一侧折返。卡住：上/下/左/右各 **Y** 帧，同一套点按再按住。
- **正向事件优于负向；能进表的别进代码；按抽象层切不按角色切。** 新旧想法冲突用新的，事后告知。组集脚本 vs TRAIN_ALIGN 已确认步骤仍先问。
- **开环回放只适用于无战斗交互的速刷图。** 有战斗必须闭环。开打方法还在 txt 里空着——实现前不要发明。

---

## 2. 演进逻辑 / 技术路线

### 本会话怎么走到这里

1. 交接第 1 节落地：YOLO 权重已加载且路径不变则 `YoloEngine.apply` 复用，不再每次读 `best.pt`。
2. 主 GUI 可改 M/L/G（后扩 A/X/Y），与回放共用 `_fsm_replay_ui.json`（merge 写入）。停仍是整段停；再开始清空 FSM context。未把键钩塞进 FSM测试。
3. 用户定性架构：模块化 agent，非行为克隆。写入 DECISIONS §1.0。覆盖第 2 节「MLP → held_frac」当通关策略。
4. held_frac 用途改判：技能废弃、移动改毫秒。组集脚本未改（等确认）。
5. FSM 正确切法拍板：核心纯函数 + replay/实机两个宿主。ctx 值语义。回放缺 `t_ns` 禁止墙钟填充。
6. `FSM_DESIGN.txt` 执行段进核心：前进 TAP→HOLD + 穿门后再走 A 帧；卡住四向各 Y 帧。开打仍是「实现中 稍后补充」。
7. 旧 `ARCHIVE_NOTE_*.md` 挪到 `archive_notes/`。
8. 用户要自己写完开打方法再继续。交接已重写：下一对话第一缺口不再是「停改参再测」。

### 路线

```
写盘（原样 jsonl）
  → 回放 / FSM测试 看视觉状态 + 蓝色意图 vs 黄色真实操作
  → fsm_core 只出意图（TAP / HOLD / 以后的技能）
  → 【现在停在这里】用户写完 FSM_DESIGN.txt 开打方法
  → 开打意图进核心（仍不发键）
  → 执行层：查键位表 → 按键（核心仍不碰键盘）
  → Live 接键、OCR 回城进快照
  → 有战斗交互才闭环；过图小模型仍是远期
```

可调参数 A/M/L/G/X/Y 在主面板。判定按 **帧** 不是秒；FSM测试间隔与录制同档 0.05s。  
主集：`20260902_074831_SolarWarden`（471 帧 / 791 键 / 有 `*_xy`）。YOLO：`solarwarden_b`。  
键位：`skill_binds/SolarWarden.json`（技能2 `f` 快捷栏，技能6 `space`）。

覆盖的旧约定：

| 旧 | 新 |
|----|----|
| MLP → held_frac 当通关策略 | 模块化 agent（感知→FSM→快捷栏） |
| 训练 y = 每键 held_frac | 技能只要按下时刻；移动发 hold_ms |
| 三个独立进程 / 发键塞进回放 | 一份核心 + 两个宿主 |
| 前进「双按」 | 点一下再按住 |
| 实机攒 400 帧 deque | 只留 context 计数 |

---

## 3. 待解决的问题

1. **开打方法未写。** `FSM_DESIGN.txt` 仍是「实现中 稍后补充」。用户写完之前不要猜技能顺序、朝怪走、乱放快捷栏。
2. **执行层未接。** `TAP`/`HOLD` 只是意图。捡物（alt）、返回城镇执行未写。回城 OCR 不在 jsonl，草案到空闲。
3. **组集未确认：** player fill、相对坐标、原点翻转、改输出 `held_ms`。先回放，点头再改文档和 `export_dataset.py`。
4. **待验证（未测之前不要当事实）：** 未知操作占比、技能事件帧长、状态 `frame_count` 分布。
5. **并行未做：** OCR 回城误切、admin 硬拦截、组集平移实验、Live 接键、自动化闭环。
6. **DECISIONS 第 4 节**（并行卡住 / 不对称迟滞 / `gate.x > player.x`）未落地，也不准覆盖 `FSM_DESIGN.txt`。

---

## 4. 今天的日期

**2026-09-02。**

新对话把 `d:\Desktop\T\test\CONTEXT_HANDOFF.md` 交给助手。先看 `FSM_DESIGN.txt` 开打是否已写完；没写就停。磁盘代码才是真相。
