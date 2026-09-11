# 项目归档笔记：mix_a 默认 YOLO；开打走近 / gaps_ms / 提取范围落地

**日期：** 2026-09-07 22:35

**工程：** `d:\Desktop\T\test`

**配套：** `FSM_DESIGN.txt`（现行）· `CONTEXT_HANDOFF.md` · `DECISIONS.md` §1.0 · 此前 `ARCHIVE_NOTE_2026-09-07-0.md`（21：40 只交接、代码未跟）

---

## 1. 本轮做了什么

同一对话后半段。用户要求归档交接后，又要求：解释「txt 没推翻」、默认 YOLO 改 mix、把 DECISIONS 里已写进 txt 的开打/提取落到代码。

1. **用词：** 「txt 没推翻」= 草案没改的条款代码接着用（前进/捡物/卡住/OCR 建段/S_pos 等）。不是「DECISIONS 整份都不落地」。
2. **DECISIONS §4 仍不落地**（每状态超时 / 不对称迟滞 / `gate.x > player.x`），不得覆盖 txt。
3. **默认过图 YOLO → `mix_a`**（`D:/Desktop/T/test/images/Atrain/runs/mix_a/weights/best.pt`）。覆盖 GUI 默认 `solarwarden_b`。`varien_*` 仍禁止当过图。`auto_label.py` 默认权重未改。
4. **开打：** 范围内才 CAST；范围外蓝字「范围异常」，捡物同款 TAP→TH→HOLD 朝分布中心走，直到本帧最远敌对进范围。无 `range_px`（效率不足未记范围）视为在范围内。
5. **技能组：** 提取写 `gaps_ms`（相邻按下间隔，跨次中位数）。FSM 按间隔 CAST 组内后续技能；结束后等到组结束时刻。
6. **提取范围：** 非 BOSS 且效率 =100% 才用步骤 3 范围框；不足不计范围；该分布该组对已记范围取 max。

改完须 **重新提取本图**。

---

## 2. 已验证 / 试验中 / 已撤销

**已验证（代码路径写完，脚本自测：范围外 TAP→HOLD 走近、组 CAST a 再 CAST b、不足 100% 不进 range_max）：**  
mix_a 默认选中逻辑；开打走近；`gaps_ms` 复现；提取不计不足 100% 范围。未当实机通关事实。

**试验中：**  
走近手感、组间隔在游戏里是否像录像；`th_ms` / GX / 点按 80–120；叠图对齐；`press_lag=1`；重提后的 skill_features 是否够用。

**已撤销：**

| 旧 | 新 |
|----|----|
| GUI 默认 `solarwarden_b`；助手不要擅自切 mix | 默认 **`mix_a`** |
| 范围异常仍然释放 | 走近再放 |
| 效率不足用最远距离记范围 | 不足 100% 不计范围 |
| 组只 CAST 第一技能 | 按 `gaps_ms` 复现后续技能 |
| 用 DECISIONS §4 当通关 TODO | §4 仍是未落地旧稿 |

---

## 3. 下一对话先做

边打边调。改规则的图先「重新提取本图」。不要用 §4 盖 txt。不要动组集 fill/相对。不要把默认 YOLO 切回 `solarwarden_b`，除非用户要对照。

---

## 4. 怎么接着干

新对话读 `CONTEXT_HANDOFF.md` 第 1 节，打开 `FSM_DESIGN.txt`。磁盘代码是真相。回复简体中文。
