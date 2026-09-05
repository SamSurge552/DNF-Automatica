# CONTEXT 交接报告

> 新开对话时把本文件路径交给助手。**磁盘代码才是真相**（实现）；**通关草案以 `FSM_DESIGN.txt` 为准**。  
> 对齐以 `TRAIN_ALIGN.md` 为准。架构以 `DECISIONS.md` 为准（**当前阶段定性见 §1.0**）。  
> DECISIONS 第 4 节（并行卡住 / 不对称迟滞 / `gate.x > player.x`）**尚未落地**，也没并进 `FSM_DESIGN.txt`。不要用第 4 节覆盖 txt。

生成时间：2026-09-05 14:50 (UTC+8)  
项目根目录：`d:\Desktop\T\test`  
本会话已做：对照清单裁定落地——E/F 进 FSM 设置；全模式 OCR+进图；回城=连续无地下城关键词（回城秒默认 30）；卡住升为状态并可打断方法；进图前加载技能表；无技能表中止 FSM测试/自动化。

> **组集脚本与 TRAIN_ALIGN 已确认步骤打架时，问用户改文档还是改代码。**  
> **新旧想法冲突：用新的，事后告知即可。**  
> 用词：**写盘** = 采集 PNG 在 `images/`，按键+YOLO 在 `recordings/<地下城>/<时间戳>_<角色>/`；**FSM测试写盘** = `FSM_TEST/`（与采集分开）；**回放** = 试清洗 / 试 FSM / 过图技能特征；**组集** = `dataset_export/`。不要把写盘叫导出。  
> `.gitignore`：根目录 `images/` 整夹忽略；`FSM_TEST` 只忽略图片，jsonl/参数可上。

---

## 1. 下一对话先做

**边打边调。** FSM测试默认发键并写 `FSM_TEST/`。对照绿/蓝字改 **FSM设置**（含连按 COUNT/间隔、MON/BOSS 补正）。回放也能改同一份 json。txt 没改就不要推翻核心。

范围异常仍释放，不要发明走近怪。旧 `skill_features` 若还是改规则前提的，回放「重新提取本图」。组集 fill/相对未确认。mash COUNT/间隔归属未决。

### 1.1 已落地（对照 txt）

总流程短路未改。方法层不可覆盖，**卡住除外**（全局最高优先级，状态改为卡住）。FSM测试默认发键并写 `FSM_TEST/`。回放不发键。各模式开 OCR；回城=连续无地下城关键词（回城秒→TN 帧）。技能表/过图文件进图前加载。

**前进：** 进前进一次性记下过门方向。之后每帧用**当前门**算 GX/GY。两轴都停后才走 AX/AY。卡住：状态=卡住，上下左右各 HOLD Y 帧（不是前进）。

**开打：** 进图前加载。排除 CD 后取该分布文件**第一个不在 CD** 的技能。无则提示无技能可放，只从已勾选快捷栏取持续帧最短。快捷栏也没有 → 普攻 X（无 CD）XXX 帧。范围异常仍然释放。

**提取：** BOSS 房不算效率/群单/假释放，仍留 killed_mon。E/F 在 FSM 设置。

**CD：** 提取短间隔 → 地图【CD重置】（连续释放 / 多次释放 / 假释放不算）。运行时该图有标记且 **击败 BOSS 数 +1** 才清 CD。MULTI 拆成多份独立 CD。

**提取：** 按下快捷栏即收，不要求开打。分布按按下帧现算。范围=最远敌对距离中位数。杀 MON>E → 群。F 默认 20。面板持续帧是提取真相，点「重新读取」才跟磁盘。

**发键：** `fsm_execute` + `key_inject`。点按 ms 默认 50。键位表【连按】→ COUNT 次点按 + 间隔 ms（默认 3 次 / 50ms）。一键拾取=左 Alt（VK）。参数在 `_fsm_replay_ui.json`。

GUI：主面板 **FSM设置**；FSM测试无发键勾选（默认发键）。写盘 `FSM_TEST/<图>/<时间戳>_<角色>/`（png + frames.jsonl + keys.jsonl + fsm_params.json）。补正 `mon_corr_u/d/l/r`；连按 `mash_count` / `mash_gap_ms`。

---

## 2. 当前目标与卡点

长期：地下城 × 角色 → 通关操作 + YOLO 特征 → 自动化。  
**当前对照（DECISIONS §1.0）：** 模块化 agent（YOLO+OCR 感知 → FSM 决策 → 快捷栏执行），**非行为克隆**。

**现在卡在哪：** 核心已跟 `FSM_DESIGN.txt`，进入实机边打边调。旧特征表可能要重提。回放不发键。组集清洗未确认。

**现成盘面（旧表结构，提取改完要重提）：** `skill_features/深渊：最终调律者/SolarWarden.json`；`skill_binds/SolarWarden.json`；YOLO `solarwarden_b`。推荐回放段 `recordings/深渊：最终调律者/20260902_074831_SolarWarden`。

---

## 3. 架构（必须守）

### 核心一份 + 两个宿主

| 层 | 文件 | 做什么 |
|----|------|--------|
| 核心 | `fsm_core.py` | 纯函数 `step(snapshot, ctx, params) → (decision, new_ctx)` |
| 回放宿主 | `fsm_replay.py` | jsonl 的 `t_ns` **原样** → `run_track` → 可视化。绿=FSM 状态；蓝=意图（该干什么）；黄=录像真实操作。过图技能特征写 `skill_features/`，不改 jsonl |
| 实机宿主 | `status_analysis_module.py` | 截图完成后、YOLO **之前**打 `t_ns` → `step`。不攒整帧队列。YOLO测试只检测不跑 FSM |

核心禁止：`time` / `sleep` / 读文件 / 发键 / 截屏 / `random` / 模块级可变状态。ctx 值语义，不就地改。`t_ns` 必须单调不减，否则抛错。回放缺 `t_ns` 禁止用墙钟填充。

决策形状：`FsmDecision`（state/flags/action/move_dir + 技能、CD、分布计数等）。卡住是状态 `卡住`；预热在 flags。`FsmAction.CAST` = 开打「立即释放」；`FsmAction.ATTACK` = 按住普攻 X（无 CD）；`FsmAction.PICK` = 捡物一键拾取。

`FsmParams`：GX/GY/AX/AY、XXX、`mon_off_x/y`（MON/BOSS 补正）、`fight_plan`（按分布 key）、`hotbar`、`dist_table`、`map_reset`。

不要拆三个独立进程。不要把发键写进回放。

### 防抖

因果（现行规则）：只看当前帧和 ctx，不向后看。开头 `max(M,L,G)` 帧打 `WARMUP`（初始 judged=False，绿字「预热」）。「旧批量从左扫到右 ≡ 逐步 step」是断言不是实测（DECISIONS §8）；若原实现向后看，回放对比基准全部作废。

### held_frac

- 技能键：**废弃**（瞬发，只要按下时间戳）。
- 移动键：执行参数，目标是**按住毫秒**，不是动作编码。组集仍输出 `held_frac`+`dt_s`（分母已是真实 dt），`held_ms = held_frac × dt_s × 1000`。改脚本直接写 ms 等确认。jsonl 不用重录。

---

## 4. 已完成（稳定约定）

### 用词与数据三层

| 层 | 路径 | 允许 |
|----|------|------|
| 写盘 | PNG=`images/<时间戳>/`；jsonl=`recordings/<地下城>/<时间戳>_<角色>/` | PNG + `keys.jsonl` + 带检测框的 `frames.jsonl`。采集当场 YOLO。`t_ns` 在截图完成后、推理前打 |
| 回放 | `python fsm_replay.py` | 试 fill / 相对 / FSM；可写 `skill_features/`，不改 jsonl |
| 组集 | `python export_dataset.py` | 仅 TRAIN_ALIGN **已确认**：held_frac、去 dup、丢第一帧、坐标原样 |

`relative_xy.py` 只给回放（提取技能特征时强制 fill+相对，不看勾选）。

### 过图 YOLO

默认 **`solarwarden_b`**：`D:/Atrain/runs/solarwarden_b/weights/best.pt`  
五类 `boss / gate / loot / mon / player`。conf 0.1、iou 0.7。`varien_t` 禁止当过图 YOLO。

人工校对标注：与 png **同目录** 的 X-AnyLabeling **json**（格式同 `Aset/solarwarden/solarwarden_b`）。`Xout/` 只给 X-AnyLabeling **导出** YOLO txt，自动标不要往 Xout 写。`auto_label.py` 只写 sidecar json，且逐张推理（禁止把路径列表一次丢给 predict）。

写盘：`t_ns, png`（新采集）。旧段另有 `player_xy, mon_xy, loot_xy, gate_xy, boss_xy, mon, loot, gate, boss, infer_ms`。  
坐标=截屏图中心（左上、y 向下）。player 只留最高 conf 一个。离线 YOLO 未做。

YOLO 权重已在内存且路径未变 → `YoloEngine.apply` **复用**。用于采集 / YOLO测试 / FSM测试。缺检测的旧 png 段可用 `backfill_session_yolo.py` 补。

### 写盘 / OCR / 键钩

- **采集模式**是唯一写盘路径：点开始即 PNG + 键盘 + YOLO，无 OCR 开录。
- 非管理员点开始 → 弹窗拒绝。开录 3 秒无新按键 → 弹窗中止并删段。
- 视觉 0.05s。OCR 只给自动化进图/回城（采集不用）。
- 键钩进程内只挂一次。须整进程重启 GUI 才生效。
- 开录 `GetAsyncKeyState` 一次 → `keys_held_at_start`（不算 3 秒门槛里的「新按键」）。
- 采集预览**只开按键窗**，不再带空 YOLO 窗。

### 组集已确认（TRAIN_ALIGN 4.3）

held_frac（真实 dt）、去 auto-repeat、相邻特征全同丢后一帧、丢无左端点第一帧。坐标原样。  
**未确认：** player fill、相对坐标、原点翻转、填充上限。

### recordings（深渊：最终调律者 / SolarWarden / solarwarden_b / 0.05s）

| 段 | frames | keys | `*_xy` | 备注 |
|----|--------|------|--------|------|
| `20260901_183607` | 535 | 1242 | 无 | 旧格式 |
| `20260901_184549` | 363 | **0** | 无 | 作废 |
| `20260901_184640` | 100 | **0** | 无 | 作废 |
| **`20260902_074831`** | **471** | **791** | **有** | **回放主集** |
| 当晚 `191013`～`191216` | ~200–480 | 有 | 有 | dt 中位约 53ms；`074831` dt 中位约 120ms |

黑字 `t=` 是从第一帧起的累计秒（两位小数）。本帧间隔看 `dt=xxms`。`推理=` 是旧段 YOLO 耗时，不进 `t_ns`。新采集无此项。

### 键位表

`python skill_bind_tool.py`。不写 `recordings/`。行号=技能1…9。快捷栏单键 + `space`。橙色提示不要乱改。  
非管理员点「读取技能」→ 弹窗拒绝。保存/载入 json 不拦。  
每个 skill：`command` / `hotbar` / `hold_frames` / `cooldown_s` / **`combo`（连续释放）**。回放改帧数与连续释放、键位工具 OCR「操作指令」「冷却时间」冒号右，**按 slot 合并**，不丢未知字段、不丢空槽旧行。技能表「连续释放」勾选后点保存。  
权威文件：`skill_binds/<角色>.json`。回放 `_fsm_replay_ui.json` 的 `skill_hold` 当备份。回放顶栏可直接打开本工具。

### 回放黄字（录像真实操作）

优先级：快捷栏技能段（独占）→ 否则可并列 捡物 / 普攻；移动与跑互斥。

- **普攻**：`x` 按住，或本帧沿有 `x` press。
- **移动**：方向键按住（或本帧沿有方向 press）；**不显示方向**。
- **跑**：本帧任一方向 `press` 次数 >「跑 press>」才叫跑（连发）。不显示方向。
- **捡物**：alt press 起持续 N 帧。
- **技能**：快捷栏单键或 `space`，持续该技能 `hold_frames`。

绿/蓝/红/黄四列固定宽**且固定高**，空字占位。蓝=方法意图；红=这一帧会发的键（回放不注入）；黄=录像真实按键。绿字第二行「怪物分布 / 已过房间 / 击败BOSS」。回城/等待清空指纹与计数，**不清 CD**（该图【CD重置】且击败 BOSS +1 才清）。  
**预热** = 开头 `max(M,L,G)` 帧防抖未满。黑字：`特征未变` ≠ FSM 卡住；`状态连续` = 当前 FSM 状态已连续帧数；`推理` = 该帧 YOLO ms。  
回放橙字显示各技能 CD 剩余。窗口贴工作区；左、下边缘尺子。过图技能特征在**右侧**。

### 过图技能特征

磁盘：`skill_feature_extract.py` 已按 txt。落盘仍是 `skill_features/<地下城>/<角色>.json`。旧 json 对不上，需重提。

### 主 GUI 模式

从左到右：**采集 | YOLO测试 | 自动化**，旁边勾 **FSM测试**。

- 采集：截图+键盘，管理员硬拦。
- **YOLO测试**：只检测，弹 `YOLO Test` 窗，不跑 FSM、不存图、不采键盘、不要求管理员。新图识别用这个。
- 自动化：OCR 进图占位。
- FSM测试：YOLO+FSM，弹 `FSM Test` 窗（不抢激活、置顶）。旁路勾 **发键** 才注入（须管理员）；点开始会把游戏拉到前台。间隔与采集同档 0.05s。日志不每帧刷，改状态/意图或约 8 秒一条。

主 GUI 停/开循环不要做成「暂停但 YOLO 还在跑」。

### 前进 / 卡住 / 开打 / 捡物

前进：GX/GY 每帧相对当前门接近；一次性方向只给过门；AX/AY 过门。  
开打：排除 CD 后取文件序列第一个就绪（仅快捷栏单键/space）；最远敌对；范围异常仍然释放；没有就绪 → 最短持续帧；全 CD 普攻 X。等待显示技能和剩余帧。CAST 后 `hold_frames`。  
捡物：PM/PT/PC；数量>PC → 左 Alt。发键仅 FSM测试勾选。点按 ms 默认 50。

### 本会话覆盖的旧约定（不要当现行）

| 旧（磁盘 / 上午 txt） | 新（`FSM_DESIGN.txt` 2026-09-04） |
|----|----|
| 房间号 = MON+BOSS 绝对中心+数量 | **怪物分布**：BOSS 表 / 小怪表，相对坐标，S% |
| 地图【重置】+ 相似房清 CD；进带【重置点】的分布才清 | 地图【CD重置】；**击败 BOSS 数 +1** 才清 |
| 范围 = 群中心 dist 中位；黄字消失后算效率 | 按下时最远敌对点距离中位；持续帧结束后算效率 |
| 提取须开打；群 = 按下时 MON>E | 按下快捷栏即提取；群 = **杀 MON 数 > E** |
| 假释放只不进序列 | 假释放不进序列 / 范围 / **CD 重置** |
| 开打只用文件第 1 个槽（该槽多次释放各份） | 排除 CD 后取**文件序列里第一个就绪**（后面的技能可以顶上） |
| F 默认 50；序列一律次数 | F 默认 **20**；BOSS 房次数、MON 房效率 |
| 开打距离不够 → 接近；范围异常不放 | **范围异常仍然释放（暂定）** |
| 提取分组沿用 FSM 当时的怪物分布 | 按下技能那一帧按分布判定现算 |
| 全 CD 则空等 | 按住普攻 X，XXX 帧（默认 20） |
| 前进过坐标后计单个 A；TAP 空帧 | GX/GY 阈值接近；一次性方向；AX/AY 过门 |
| 前进用【门坐标快照】，不再跟实时门 | 前进每帧相对**当前门** GX/GY；过门方向仍一次性 |
| 发键用键位表完整指令（含 +） | 只用快捷栏单键 / space（`bind_hotkey`） |
| 点按立刻抬起；CAST_STEP_S 写死 | **点按 ms** 默认 50，主面板/回放共用 |
| 左 Alt 扫码瞬点 | VK 左 Alt + 短按 |

### 归档

`archive_notes/`。上午大改：`ARCHIVE_NOTE_2026-09-04-0.md`。本晚：`archive_notes/ARCHIVE_NOTE_2026-09-04-1.md`。

---

## 5. 文件职责

| 文件 | 职责 |
|------|------|
| `fsm_core.py` | **唯一** FSM 判定 + 前进/卡住/开打/捡物意图 |
| `fsm_replay.py` | 回放宿主；黄蓝绿；技能特征入口；F / 删【重置】 |
| `skill_feature_extract.py` | 快捷栏技能段 → `skill_features/`；`apply_f` / `sequences` / `map_tags` |
| `gui_module.py` | 面板；采集 / YOLO测试 / 自动化 / FSM测试 |
| `fsm_execute.py` | 意图 → SendInput；`tap_ms` 点按间隔 |
| `key_inject.py` | SendInput；修饰键走 VK |
| `_fsm_replay_ui.json` | 回放/测试共用参数（含 E、F、PM、PC、PT、**tap_ms**） |
| `status_analysis_module.py` | OCR 进图；YOLO测试；FSM测试宿主（可发键） |
| `record_preview.py` | 采集只开按键窗 |
| `yolo_engine.py` | 检测；`apply` 复用权重；采集阻塞 `infer_features` |
| `backfill_session_yolo.py` | 给只有 png 的 frames 补检测框 |
| `operation_analysis_module.py` | 写盘（截图+键盘+YOLO）；键钩只挂一次 |
| `export_dataset.py` | 组集 |
| `skill_bind_tool.py` | 战斗统计 → 键位表（指令+冷却+连续释放） |
| `window_geom.py` | 各 GUI 上次关闭时的窗口大小 |
| `_window_geom.json` | 窗口大小记录 |
| `window_align.py` | 对齐窗口；`focus_region` / `focus_point` 切前台 |
| `FSM_DESIGN.txt` | **现行通关草案（2026-09-04 大改）** |
| `TRAIN_ALIGN.md` | 组集已确认 / 未确认 |
| `DECISIONS.md` | 为什么；§1.0 现行 |
| `CONTEXT_HANDOFF.md` | 本文件 |
| `.cursor/rules/new-idea-wins.mdc` | 新旧想法：新的为准 |

---

## 6. 关键路径

- 工程：`d:\Desktop\T\test`
- 推荐段：`recordings/深渊：最终调律者/20260902_074831_SolarWarden`
- 特征：`skill_features/深渊：最终调律者/SolarWarden.json`
- 键位：`skill_binds/SolarWarden.json`
- YOLO：`D:\Atrain\runs\solarwarden_b/weights/best.pt`
- GUI：`python gui_module.py`（采集须管理员）
- 回放：`python fsm_replay.py`
- 键位工具：`python skill_bind_tool.py`（读取技能须管理员）
- 组集：`python export_dataset.py`

---

## 7. 再往后

1. 边打边调开打/捡物/前进手感（`tap_ms`、持续帧、GX/GY…）。  
2. OCR 回城进快照 `town_return`。

并行未做：组集 fill/相对、自动化闭环、采集后离线 YOLO。

---

## 8. 怎么交接

1. 新对话首条：`d:\Desktop\T\test\CONTEXT_HANDOFF.md`，并打开 `FSM_DESIGN.txt`。  
2. **边打边调**；txt 没改就不要推翻核心。范围异常仍然释放，不要发明接近。  
3. 发键只在 FSM测试「发键」勾选；不要写进回放 / `fsm_core`。开打技能范围 = 快捷栏单键 / space。  
4. `blueprint.txt` = 长期意图；本文件 = 当天状态。磁盘代码才是真相。回复简体中文。
