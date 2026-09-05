# 项目归档笔记：FSM测试写盘 + 连按 + 补正收进设置

**日期：** 2026-09-05 午

**工程：** `d:\Desktop\T\test`

**配套：** `FSM_DESIGN.txt`（现行）· `CONTEXT_HANDOFF.md` · `.gitignore` · `archive_notes/ARCHIVE_NOTE_2026-09-04-1.md`

---

## 1. 本轮做了什么

边打边调期间加了 MON/BOSS 位置补正，主面板 FSM 旋钮收进「FSM设置」。随后 FSM测试改为**默认发键并写盘**；技能表加执行层**连按**。最后加 `.gitignore`，交接。

下一对话继续实机边打边调。txt 没改就不要推翻核心。发键不要写进回放 / `fsm_core`。

---

## 2. 已落地

**补正：** 上下左右滑块。往上站 = YOLO 的 MON/BOSS **Y 全部减去**。只改 FSM 用的坐标，不改 jsonl、预览框。json：`mon_corr_u/d/l/r`。

**GUI：** 主面板「FSM设置」收纳 M/L/G、GX/GY/AX/AY、XXX、捡物、点按 ms、连按 COUNT/间隔、补正。跑着测试也能开。

**FSM测试：** 去掉发键勾选，默认发键（须管理员）。只把 **FSM测试** 写到 `FSM_TEST/<图>/<时间戳>_<角色>/`（`png/`、`frames.jsonl`、`keys.jsonl`、`fsm_params.json`，并复制 `skill_binds.json`）。采集仍走 `recordings/` + `images/`。YOLO测试不落盘。不做 OCR；图名用 `guess_dungeon`（该角色若只有一份 `skill_features/<图>/<角色>.json` 就用该图名，否则目录叫 `FSM测试`）。

**连按：** 键位表勾选 `mash`。与「连续释放」(CD 270)、「多次释放」(MULTI 份 CD) 不是一回事。核心仍 CAST 一次；`fsm_execute` 展开成 COUNT 次点按 + 间隔。默认 COUNT=3、间隔=50ms（松开后再等间隔）。参数：`mash_count` / `mash_gap_ms`。

**Git：** `.gitignore` 忽略根目录 `images/`，以及 `FSM_TEST` 里的图片（jsonl/参数仍可上传）。

---

## 3. 不要当现行的旧条

| 旧 | 新 |
|----|----|
| FSM测试要勾「发键」；不落盘 | 默认发键；写 `FSM_TEST/` |
| FSM 旋钮摊在主面板模式栏 | 「FSM设置」窗口 |
| 只有连续释放 / 多次释放 | 另有执行层【连按】 |
| 放技能站位完全信 YOLO 框中心 | MON/BOSS 可补正 |

---

## 4. 怎么接着干

新对话读 `CONTEXT_HANDOFF.md` 第 1 节。边打边调：GX/GY/AX/AY、XXX、补正、连按 COUNT/间隔、`tap_ms`、PC。范围异常仍释放，不要发明走近怪。
