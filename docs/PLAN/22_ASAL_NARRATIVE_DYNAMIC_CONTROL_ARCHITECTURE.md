# 22_ASAL_NARRATIVE_DYNAMIC_CONTROL_ARCHITECTURE

## 目的

本文件定義 `ASAL narrative cell fusion` 的第二版技術方向。

第一版已完成：

- morphology analyzer
- narrative scoring
- phase keyframes
- `1 -> 2 -> 1` 可檢查的軌跡資料

但第一版也已經明確揭露目前真實瓶頸：

- `boids` 會碎裂成大量小主體
- 關掉 `keep_largest_component` 之後，split 並沒有變成乾淨的 `2`，
  而是變成 `many`
- 也就是說，現在缺的不是觀測，而是**動態控制**

一句話：

> 第二版的目標不是修圖，而是把 substrate dynamics 從「自由碎裂」升級成「受 phase 控制的細胞樣結構演化」。

---

## 一、為什麼需要第二版

第一版 run 的真實結果顯示：

- 有些軌跡會被壓成 `1 -> 1 -> 1`
  原因是 render heuristic 抹掉了第二主體
- 去掉 heuristic 後，軌跡會變成 `28 -> 37 -> 35`
  原因是 substrate 本身沒有穩定單主體、雙主體、再融合的機制

因此第二版必須往這四個方向修：

1. 壓低 `boids` 碎裂
2. 先穩定形成單一主體
3. 再鼓勵乾淨二分
4. 最後保留雙主體一段時間再融合

---

## 二、第二版的核心原則

### 1. 從 static theta 升級為 phase-controlled dynamics

目前 `boids` 的問題是：

- 全程使用同一組控制參數
- 缺少「在某段時間做 birth / split / fusion」的主動驅動

第二版必須改成：

- base theta 仍由 evolutionary search 負責
- 但每個 phase 再套上不同的控制規則

也就是：

```text
final control = base theta + phase controller
```

### 2. 不是新增 fake postprocess，而是新增 real dynamics constraints

第二版禁止把 split / fusion 假裝成 render trick。

必須在動力學層處理：

- center pull
- split push
- cohort cohesion
- fusion recovery
- velocity damping

---

## 三、建議技術架構

### A. Narrative Controller Layer

新增概念：

- `narrative_controller`

用途：

- 根據目前時間步所在 phase，回傳一組動態控制 scale

建議位置：

- `substrate.params.narrative_controller`

資料形狀示意：

```yaml
substrate:
  name: boids
  params:
    num_boids: 100
    width: 128
    height: 128
    keep_largest_component: false
    narrative_controller:
      enabled: true
      split_axis: horizontal
      base_center_pull: 0.015
      base_velocity_damping: 0.08
      phases:
        - name: birth
          frame_range: [0, 50]
          separation_scale: 0.35
          cohesion_scale: 1.35
          alignment_scale: 1.10
          center_pull: 0.030
          split_push: 0.0
          cohort_pull: 0.0
          damping: 0.12
        - name: split
          frame_range: [51, 110]
          separation_scale: 0.90
          cohesion_scale: 0.95
          alignment_scale: 1.00
          center_pull: 0.006
          split_push: 0.040
          cohort_pull: 0.020
          split_offset: 18.0
          damping: 0.08
        - name: fusion
          frame_range: [111, 179]
          separation_scale: 0.30
          cohesion_scale: 1.30
          alignment_scale: 1.05
          center_pull: 0.035
          split_push: 0.0
          cohort_pull: 0.0
          damping: 0.14
```

---

### B. Cohort-Aware Split Layer

如果要穩定地從 `1` 變 `2`，必須有持續的雙群機制。

建議：

- `boids.reset()` 時固定把 boids 分成兩群 `cohort A / cohort B`
- split phase 不直接讓每隻粒子自由亂跑
- 而是讓兩群各自受到：
  - 相反方向 split push
  - 各自的 cohort anchor pull

效果：

- 比較容易形成兩個主體
- 不會直接碎成很多小島

---

### C. Centered Birth / Fusion Stabilizer

`birth` 與 `fusion` 的共同需求是：

- 收斂成單一主體
- 避免多團塊發散

建議增加：

- `center_pull`
- `velocity_damping`

說明：

- `center_pull`：把整體往 arena 中心拉，利於先形成單核，再重新融合
- `velocity_damping`：降低高速碎裂

---

### D. Phase Schedule Integration

控制器必須知道：

- 現在是哪一個 phase
- 對應該 phase 的控制參數是什麼

建議做法：

1. `engine` 啟動前把：
   - `runtime.steps`
   - `narrative.phases`
   傳給 substrate
2. substrate 每一步根據 `current_step`
   找到 active phase
3. 每一步使用 active phase control 更新 dynamics

---

## 四、建議修改點

### 1. `research/asal_engine/substrates/boids.py`

需要新增：

- `narrative_controller` config
- `configure_narrative(runtime_steps, phases)`
- stable cohort assignment
- active phase lookup
- additional forces:
  - `center_pull`
  - `split_push`
  - `cohort_pull`
  - `damping`

### 2. `research/asal_engine/engine.py`

需要新增：

- 在 substrate 建立後，如果 narrative enabled：
  - 把 runtime steps 與 phase specs 送進 substrate

### 3. `configs/asal/target_cell_fusion_narrative.yaml`

需要擴充：

- narrative controller params
- `cpu_tiny` / `cpu_smoke` phase controller profile

### 4. tests

至少應補：

- controller phase selection test
- cohort split force smoke test
- no-controller vs controller 的 component-count 差異測試

---

## 五、第二版實作分段

### TASK GROUP D1 — Phase-Controlled Boids Runtime

目標：

- 讓 `boids` 支援 phase-aware control schedule

完成條件：

- 同一組 theta 不再是全程固定 dynamics
- birth/split/fusion 可切換不同 control scale

### TASK GROUP D2 — Cohort Split Mechanism

目標：

- 讓雙主體形成是「受控二分」而不是無控制碎裂

完成條件：

- split phase 有穩定雙群分離趨勢

### TASK GROUP D3 — Fusion Recovery Controls

目標：

- 讓雙主體在 fusion phase 有明確回收機制

完成條件：

- `2 -> 1` 的回收路徑比第一版更可行

### TASK GROUP D4 — Narrative Validation Runs

目標：

- 用真實 run 驗證 component sequence 是否開始接近 `1 -> 2 -> 1`

完成條件：

- 至少出現：
  - 單主體穩定 birth
  - split phase 低碎裂、接近雙主體
  - fusion phase 主體數下降

---

## 六、Definition of Done

第二版要算有價值，至少要達成：

1. 有 phase-controlled boids
2. 有 cohort-based split control
3. 有 birth/fusion stabilizer
4. narrative run 的實際 component sequence 不再是：
   - `1 -> 1 -> 1`
   - 或 `28 -> 37 -> 35`
5. 至少接近：
   - `1 -> 2..4 -> 1..2`

注意：

- 第二版不要求一次就完美達到 `1 -> 2 -> 1`
- 但必須明確比第一版更接近目標敘事

---

## 七、一句話結論

**第二版的本質不是做更漂亮的圖，而是把 `ASAL` 的細胞敘事從「能觀測失敗」推進到「能開始控制動力學」，讓真正的誕生、二分、再融合變成可優化的 substrate 行為。**
