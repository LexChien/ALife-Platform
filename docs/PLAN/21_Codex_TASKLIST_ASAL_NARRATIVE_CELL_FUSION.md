# 21_Codex_TASKLIST_ASAL_NARRATIVE_CELL_FUSION

## 目的

本文件將 [20_ASAL_NARRATIVE_CELL_FUSION_SPEC.md](/home/lexchien/Documents/ALife-Platform/docs/PLAN/20_ASAL_NARRATIVE_CELL_FUSION_SPEC.md)
拆成可直接執行的 task list，目標是把 `ASAL` 從單段 target search
擴充為可評估：

- 單細胞誕生
- 兩細胞分裂
- 再融合為單一主體

的敘事型演化過程。

---

## 工作原則

1. 第一版先求「事件可檢測」，不先求最漂亮圖像
2. 第一版先以 `boids` 為主體，因為目前歷史可比基準在這條路上
3. 每一階段都要輸出真實 artifact，不接受只看 console score
4. 不得只用最後一幀宣稱敘事成功
5. 每完成一組 task，都必須能留下可驗證的中間結果

---

## TASK GROUP N1 — Morphology Analyzer

### 目標
建立最小可用的形態分析器，能從 rendered frame 推出結構事件。

### 新增檔案
- `research/asal_engine/morphology.py`

### 實作內容
1. 支援從 RGB frame 轉灰階
2. 進行閾值化得到前景 mask
3. 計算 connected components
4. 輸出至少：
   - `num_components`
   - `largest_area`
   - `second_area`
   - `dominant_mass_ratio`
   - `centroid_distance`
   - `largest_centroid`
5. 需支援直接分析 numpy array / PIL image

### 驗收條件
- 單張圖可得到穩定 morphology stats
- 可明確區分：
  - 單主體
  - 雙主體
  - 碎裂多主體

### 建議 commit
`asal: add morphology analyzer`

---

## TASK GROUP N2 — Narrative Scoring

### 目標
建立 phase-aware scoring，不再只看最後一幀。

### 新增檔案
- `research/asal_engine/narrative_scores.py`

### 實作內容
1. 定義 phase score 函式：
   - `score_birth_phase`
   - `score_split_phase`
   - `score_fusion_phase`
2. 定義 `score_narrative_trajectory(frames, narrative_cfg)`
3. 支援：
   - phase frame range
   - target component count
   - per-phase weight
4. 回傳：
   - `birth_score`
   - `split_score`
   - `fusion_score`
   - `total_score`
   - `phase_order_valid`
   - `selected_keyframes`

### 驗收條件
- 同一段軌跡可被分段評分
- 能明確拒絕不符合 `1 -> 2 -> 1` 的序列

### 建議 commit
`asal: add narrative trajectory scoring`

---

## TASK GROUP N3 — Narrative Config

### 目標
把 narrative objective 變成正式 config，而不是硬寫在 code。

### 新增檔案
- `configs/asal/target_cell_fusion_narrative.yaml`

### 實作內容
1. 定義 `narrative.enabled`
2. 定義三段 phase：
   - birth
   - split
   - fusion
3. 每段至少包含：
   - `name`
   - `frame_range`
   - `target_components`
   - `weight`
4. 加入 narrative score 權重，例如：
   - `narrative.weight`
   - `semantic.weight`
   - `morphology.weight`

### 驗收條件
- engine 可從 config 讀出 narrative phases
- 不需在 engine 裡硬編碼三段時間窗

### 建議 commit
`asal: add narrative fusion config`

---

## TASK GROUP N4 — Engine Integration

### 目標
讓 `ASALEngine` 支援用整段軌跡評分，並導出敘事 artifact。

### 修改檔案
- `research/asal_engine/engine.py`

### 實作內容
1. 若 `narrative.enabled = true`：
   - 在 `eval_theta()` 中收集整段 frame trajectory
   - 執行 narrative scoring
2. 分數聚合應至少支援：
   - semantic score
   - morphology score
   - narrative score
3. 在最佳候選輸出時儲存：
   - `phase_keyframes/phase1_birth.png`
   - `phase_keyframes/phase2_split.png`
   - `phase_keyframes/phase3_fusion.png`
   - `narrative_summary.json`
4. `summary.json` 需保留 narrative metadata

### 驗收條件
- run 結果不只有 `best.gif`
- 還能看到三段關鍵幀與 narrative summary

### 建議 commit
`asal: integrate narrative scoring and keyframe export`

---

## TASK GROUP N5 — Validation And Tests

### 目標
驗證 narrative path 真的能工作，而不是只有文件和分數。

### 新增檔案
- `tests/test_asal_morphology.py`
- `tests/test_asal_narrative_scores.py`

### 實作內容
1. 為 morphology analyzer 建人工測資：
   - 單圓
   - 雙圓
   - 多碎片
2. 為 narrative score 建人工軌跡：
   - 符合 `1 -> 2 -> 1`
   - 不符合 `1 -> 2 -> 1`
3. 新增 narrative smoke config 驗證命令

### 驗收條件
- tests 能明確證明 analyzer 和 score 有效
- narrative run 產出可被人工檢查

### 建議 commit
`tests: add asal narrative coverage`

---

## 推薦執行順序

1. `N1`
2. `N2`
3. `N3`
4. `N4`
5. `N5`

---

## 第一版完成條件

第一版要算完成，至少要達成：

- 有 morphology analyzer
- 有 narrative score
- 有 narrative config
- engine 能輸出三段 keyframe
- engine 能輸出 `narrative_summary.json`
- 至少有一個 smoke run 可人工檢查是否接近：
  - birth
  - split
  - fusion

---

## 一句話結論

**`21` 的目標不是再做一條單張圖最佳化，而是把 `ASAL` 正式推進到「能判定與輸出細胞誕生、分裂、融合敘事」的第一版工程實作。**
