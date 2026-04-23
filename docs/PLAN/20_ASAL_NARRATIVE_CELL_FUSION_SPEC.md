# 20_ASAL_NARRATIVE_CELL_FUSION_SPEC

## 目的

本文件定義 `ASAL` 如何從目前的：

- 單一 substrate
- 單一目標分數
- 單段時間序列

升級成能表達以下敘事的模擬系統：

1. 單細胞誕生
2. 分裂為兩個細胞
3. 兩個細胞重新融合成唯一主體

一句話說明：

> 目標不是只產生「像細胞的單張圖」，而是要產生一段具有結構事件與敘事順序的演化過程。

---

## 一、目前架構缺口

目前 `ASAL` 實際具備的是：

- 單一 substrate 動態展開
- 單一分數最佳化
- 最後一幀或單一路徑做目標評分

目前**沒有**的能力：

### 1. 沒有 phase-aware objective

缺少：
- `Phase 1 / Phase 2 / Phase 3` 的正式概念
- 每段獨立目標
- 每段時間窗

### 2. 沒有 trajectory scoring

缺少：
- 關鍵幀評分
- 時間窗評分
- 多段分數聚合

### 3. 沒有 morphology event analyzer

缺少：
- 連通區數量
- 主體面積
- 單主體 / 雙主體 / 融合回單主體的事件判斷

### 4. 沒有 narrative selection

缺少：
- `1 -> 2 -> 1` 的序列約束
- 「符合敘事」的最終候選選拔機制

---

## 二、目標能力定義

### Narrative Goal

要達成的不是單純「細胞樣動態」，而是：

```text
Phase 1: Birth
  0 -> 1 dominant cell-like body

Phase 2: Split
  1 dominant body -> 2 separated dominant bodies

Phase 3: Fusion
  2 dominant bodies -> 1 dominant fused body
```

### 完成後使用者應能得到

- 一段 `gif`
- 其中關鍵時間點可被解釋為：
  - 誕生
  - 分裂
  - 融合
- 並且 `summary.json` 或等價 artifact 能記錄這三段是否達成

---

## 三、建議新增的架構層

### 1. Narrative Config Layer

建議新增新的 config schema，例如：

```yaml
prompt: "a microscopy image of a biological cell, phase contrast"

substrate:
  name: boids
  params: {...}

runtime:
  steps: 180
  substeps: 1

narrative:
  phases:
    - name: birth
      frame_range: [0, 50]
      target_components: 1
      target_profile: single_cell_birth
      weight: 0.3
    - name: split
      frame_range: [51, 110]
      target_components: 2
      target_profile: dual_cell_split
      weight: 0.4
    - name: fusion
      frame_range: [111, 179]
      target_components: 1
      target_profile: fused_single_cell
      weight: 0.3
```

用途：
- 明確告訴引擎各段時間應發生什麼事件
- 不再依賴單一 prompt 去隱含整段敘事

### 2. Morphology Analyzer Layer

建議新增：

- `research/asal_engine/morphology.py`

最小能力：
- 二值化 / 密度閾值化
- connected components 數量
- 最大主體面積
- 前兩大主體面積
- 主體中心距離
- 主體圓形度 / 緊密度

至少要支援：

```python
analyze_frame(img) -> {
    "num_components": int,
    "largest_area": float,
    "second_area": float,
    "centroid_distance": float | None,
    "dominant_mass_ratio": float,
}
```

### 3. Phase Scoring Layer

建議新增：

- `research/asal_engine/narrative_scores.py`

職責：
- 根據不同 phase 的形態要求打分
- 分開評估：
  - `birth_score`
  - `split_score`
  - `fusion_score`
- 最後再聚合成：
  - `narrative_total_score`

### 4. Trajectory Selection Layer

現有做法是：
- 只看單一終點分數

新做法應升級為：
- 評分整段軌跡
- 檢查敘事順序是否成立

也就是 candidate selection 變成：

```text
high morphology plausibility
+ phase order validity
+ trajectory-level narrative score
```

---

## 四、三段目標如何定義

### Phase 1: Birth

目標：
- 從背景中形成單一主體
- 不希望一開始就有多個分裂主體

建議條件：
- `num_components == 1`
- `largest_area` 逐步增加
- `dominant_mass_ratio` 高

評分關注：
- 單一主體是否形成
- 是否從低質量背景成長出明顯細胞

### Phase 2: Split

目標：
- 由單一主體分裂成兩個主要主體

建議條件：
- `num_components == 2`
- 兩主體大小接近
- 兩主體距離高於最小分離門檻

評分關注：
- 是否真的變成兩個主體
- 不是一堆碎片
- 不是仍然單一黏連團塊

### Phase 3: Fusion

目標：
- 兩個主體重新聚合成單一主體

建議條件：
- 早段曾達成 `num_components == 2`
- 後段回到 `num_components == 1`
- `largest_area` 回升
- `dominant_mass_ratio` 回升

評分關注：
- 是否真有「兩者重新合一」
- 不是簡單消失一個主體
- 不是碎裂後只剩一小塊

---

## 五、分段 scoring 設計

### 現況

目前 scoring 類型基本上是：

- semantic similarity
- morphology similarity
- 但本質仍接近單階段評分

### 新需求

要改成：

```python
def narrative_score(frames, phase_specs, analyzer):
    birth = score_birth(frames[phase1_range], analyzer)
    split = score_split(frames[phase2_range], analyzer)
    fusion = score_fusion(frames[phase3_range], analyzer)
    return {
        "birth_score": ...,
        "split_score": ...,
        "fusion_score": ...,
        "total_score": weighted_sum(...),
    }
```

### 為什麼不能只看最後一幀

因為：
- 最後一幀看起來像單細胞，不代表中間有經過分裂與融合
- 使用者要的是**事件序列**
- 不是單點視覺相似

---

## 六、render / selection 要怎麼變

### 1. Render

目前 render 的角色是：
- 給 foundation model 看
- 給使用者看

未來還要多一個角色：
- 給 morphology analyzer 做結構判定

這代表 render 至少要：
- 主體可分割
- 邊界清楚
- 背景與主體對比足夠

### 2. Selection

新 selection 不應只看：
- `best_score`

而應改成：

- `narrative_total_score`
- `phase_order_valid`
- `component_transition_valid`

### 3. 必須檢查的結構事件

至少：
- Phase 1 目標是 `1`
- Phase 2 目標是 `2`
- Phase 3 再回到 `1`

如果沒有達成這個序列，就算某張圖看起來漂亮，也不應算 narrative success。

---

## 七、建議新增的 artifact

目前 artifact 不足以說明敘事是否達成。

建議新增：

- `phase_keyframes/`
  - `phase1_birth.png`
  - `phase2_split.png`
  - `phase3_fusion.png`

- `narrative_summary.json`

建議內容：

```json
{
  "birth_score": 0.82,
  "split_score": 0.74,
  "fusion_score": 0.79,
  "total_score": 0.78,
  "phase_order_valid": true,
  "component_counts": {
    "birth": 1,
    "split": 2,
    "fusion": 1
  }
}
```

- `best.gif`
  - 保留完整演化序列

---

## 八、最低可行版本（MVP）

如果要先做最小可用版本，建議：

### Phase N1
- 只做 `boids` substrate
- 只做 morphology-based narrative judge
- 不先混 OpenCLIP

### Phase N2
- 加入 OpenCLIP / morphology 混合評分

### Phase N3
- 再考慮更強 substrate
  - improved boids
  - reaction-diffusion narrative
  - future hybrid substrate

一句話：

> 第一版先求敘事正確，不先求 foundation-model score 最漂亮。

---

## 九、建議實作切分

### N1 — Morphology Analyzer

新增：
- `research/asal_engine/morphology.py`

### N2 — Narrative Score Functions

新增：
- `research/asal_engine/narrative_scores.py`

### N3 — Narrative Config

新增：
- `configs/asal/target_cell_fusion_narrative.yaml`

### N4 — Engine Extension

修改：
- `research/asal_engine/engine.py`

讓引擎支援：
- phase config
- multi-phase trajectory score
- narrative artifact export

### N5 — Validation

新增：
- narrative smoke run
- keyframe artifact validation
- sequence validity checks

---

## 十、Definition of Done

這個功能要算完成，至少要滿足：

- config 可定義三段敘事
- engine 可對整段軌跡分段評分
- 有 morphology analyzer
- 能檢查 `1 -> 2 -> 1`
- 能輸出 phase keyframes
- 能輸出 narrative summary
- `best.gif` 具備可解釋的：
  - birth
  - split
  - fusion

---

## 十一、現在不要做的事

- 不要先只追求 OpenCLIP 分數
- 不要先把這功能誤解成單張圖最佳化
- 不要先假設任意 substrate 都自然能長出 `1 -> 2 -> 1`
- 不要用單一最後一幀來宣稱敘事成功

---

## 一句話結論

**目前 ASAL 沒有敘事型細胞融合模擬能力；若要做到「誕生 -> 分裂 -> 融合」，必須新增 phase-aware config、trajectory scoring、morphology analyzer 與 `1 -> 2 -> 1` 結構事件判定。**
