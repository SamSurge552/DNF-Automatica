# 项目归档笔记：回放无法启动，交接给新对话

**日期：** 2026-09-05 15:40

**工程：** `d:\Desktop\T\test`

**配套：** `FSM_DESIGN.txt`（现行）· `CONTEXT_HANDOFF.md` 第 1 节 · `archive_notes/ARCHIVE_NOTE_2026-09-05-0.md`

---

## 1. 本轮做了什么

对照清单裁定已落地（卡住状态、回城 debounce、进图前加载技能表、BOSS 房提取不算群单等）。mash 从核心挪到执行层。准备边打边调时发现 **回放起不来**，本会话**不修代码**，交给新对话。

---

## 2. 下一对话第一件事

`python fsm_replay.py` → `__init__` → `_open_selected` → `_rebuild_draft` → `casts_from_tracks`：

```
skill_feature_extract.py:276
kind = None if is_boss else ("群" if killed_mon > e else "单")
UnboundLocalError: is_boss
```

`is_boss` 在 287 行才赋值。意图：BOSS 房 `kind=None`，仍写 `killed_mon`。先 `classify_dist` 再算 kind。修完确认窗口能开即可。

---

## 3. 不要当现行的旧条

| 旧 | 新 |
|----|----|
| mash COUNT 进 `FsmParams` / 核心 `send_label` | 核心只出槽；执行层连按；字幕宿主拼 `连按N×` |
| FSM测试要勾「发键」 | 默认发键，写 `FSM_TEST/` |
| DECISIONS §4 卡住草案 | 以 `FSM_DESIGN.txt` + 磁盘为准（卡住是状态） |
| 回城靠关键词命中 | 连续 N 帧无地下城关键词（回城秒默认 30） |

---

## 4. 怎么接着干

新对话读 `CONTEXT_HANDOFF.md` 第 1 节。先修回放，再边打边调。发键不要写进回放 / `fsm_core`。范围异常仍释放。
