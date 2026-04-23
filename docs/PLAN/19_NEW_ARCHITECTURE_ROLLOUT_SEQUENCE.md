# 19_NEW_ARCHITECTURE_ROLLOUT_SEQUENCE

## 目的

本文件整理 `16 / 17 / 18` 三份文件，形成一份可直接執行的
**新架構落地順序圖**。

它的作用不是取代原文件，而是回答三個更實際的問題：

1. 這三份文件各自負責什麼層級？
2. 現在應該先做哪一段？
3. 哪些事情不能混在同一階段一起做？

---

## 一、三份文件的角色定位

### 16：模型戰略層

文件：
- `16_OWN_LLM_MODEL_STRATEGY.md`

回答的問題：
- 我們未來要擁有什麼樣的模型體系？
- Gemma 在整個平台裡到底是 bootstrap 還是終局？
- Clone / Judge / Planner 應不應該分成不同模型線？

這份文件的核心結論：
- `Gemma` 是 bootstrap model，不是最終唯一模型
- 中期應建立三條自有模型線：
  - `clone_model`
  - `judge_model`
  - `planner_model`
- 目前不要做 scratch base model

### 17：工程落地層

文件：
- `17_Codex_TASKLIST_OWN_MODEL_PHASE_1.md`

回答的問題：
- 現在要補哪些工程基礎，才能讓「自有模型」變成 repo 內的正式資產？
- Phase 1 應該如何分組實作？

這份文件的核心結論：
- Phase 1 先不訓練模型
- 先做：
  - `ModelSpec`
  - `model_specs/`
  - `model_registry`
  - `model_id` factory flow
  - `prompt_profile`
  - training scaffold
  - dataset contracts
  - lineage tracking

### 18：新 backend 接入層

文件：
- `18_GEMINI_NEW_MODEL_IMPLEMENTATION_SPEC_CLEAN.md`

回答的問題：
- 未來如果要接新的模型 family，例如 `Llama-3` / `Mistral` / `Phi-3`
  應該怎麼接？
- 怎麼確保新模型不破壞 shared adapter / Phase 0 / artifact flow？

這份文件的核心結論：
- 新模型接入必須遵守：
  - `BaseLLMAdapter`
  - shared factory
  - config-driven backend selection
  - artifact-first workflow
  - heuristics / metadata / summary 規範

---

## 二、總體架構升級方向

從 `10~14` 的 LLM integration 規劃往前走時，平台原本的重點是：

- 先把一條真實本地 LLM runtime 跑通
- 先讓 `GenAI` 與 `Digital Clone` 能共用 adapter

到 `16~18` 後，架構目標已經升級為：

### 1. Runtime Layer
負責「怎麼跑」

例如：
- `llama_cpp`
- future `transformers`
- future `vllm`

### 2. Model Layer
負責「跑哪顆模型」

例如：
- `gemma_bootstrap_v1`
- `clone_lora_v1`
- `judge_lora_v1`
- `planner_lora_v1`

### 3. Capability Layer
負責「這顆模型做什麼」

例如：
- `genai_chat`
- `digital_clone`
- `judge`
- `planner`

一句話說明：

> 平台接下來要從「只有 backend」升級成「backend + model asset + role/capability mapping」。

---

## 三、正確落地順序

### Phase R1：先穩定既有 bootstrap 路徑

目標：
- 把現有 `Gemma + llama.cpp` shared adapter 路徑視為 bootstrap base
- 保持 `GenAI` / `Digital Clone` / evaluation 可運作

這一段已基本完成，但仍是所有後續工作的前提。

輸出：
- working local bootstrap model path
- shared LLM adapter
- evaluation baseline

對應文件：
- `10`
- `11`
- `12`
- `13`
- `14`

### Phase R2：建立模型資產層

目標：
- 把模型身份從 backend config 中抽出
- 讓模型成為 repo 內正式可追蹤資產

必做項：
- `M1` `ModelSpec`
- `M2` `model_specs/`
- `M3` `model_registry`
- `M4` factory 支援 `model_id`
- `M5` prompt profile 擴充點

這是目前最優先的主線。

對應文件：
- `16`
- `17`

輸出：
- `model_specs/*.yaml`
- `core/model_spec.py`
- `core/model_registry.py`
- `model_id` driven factory flow

### Phase R3：建立 owned-model 工程骨架

目標：
- 讓 clone / judge / planner specialization 有正式工程落點

必做項：
- `M6` training scaffold
- `M7` dataset contracts
- `M8` lineage tracking
- `M9` docs/context 同步
- `M10` tests

對應文件：
- `17`

輸出：
- `training/`
- dataset schema docs
- lineage docs / metadata
- owned-model infrastructure tests

### Phase R4：再接第二個新模型 backend

目標：
- 在已有 `model_id` / `model_registry` / model asset layer 的前提下，
  新增第二個非 Gemma family backend

這時候 `18` 才真正開始發揮價值。

對應文件：
- `18`

輸出：
- `genai/llm/backends/new_model.py`
- new backend configs
- smoke tests
- shared integration without business-logic leakage

---

## 四、不要混在一起做的事

### 1. 不要在還沒有 `ModelSpec` 前就大量接新模型 family

原因：
- 會讓 backend config 繼續擴散
- 之後還要再做一次抽象重構

### 2. 不要在 `owned model infrastructure` 尚未建立前就直接進入 LoRA 訓練

原因：
- 模型 lineage 會不可追蹤
- dataset contract 會持續漂移
- 最後只會得到一批檔名不清楚的權重

### 3. 不要同時做 `17` 與 `18` 的重實作主線

原因：
- `17` 解的是模型資產管理
- `18` 解的是新 family 接入
- 同時開工會造成抽象邊界來回改動

### 4. 不要把 Gemma-only 假設再次寫回業務層

原因：
- 這會直接抵消 `16` 的戰略升級
- 也會讓後續 `clone / judge / planner` 切模型變得昂貴

---

## 五、建議執行順序圖

```text
Stage 0
  10~14 完成 shared LLM bootstrap
    ->
Stage 1
  16 確立 owned-model strategy
    ->
Stage 2
  17.M1 ~ 17.M5
  先做 ModelSpec / registry / model_id / prompt_profile
    ->
Stage 3
  17.M6 ~ 17.M10
  補 training scaffold / dataset contracts / lineage / tests
    ->
Stage 4
  18 接第二個新 backend family
    ->
Stage 5
  clone / judge / planner specialization
```

---

## 六、目前最合理的開工點

若以目前 repo 狀態判斷，下一個實作起點應是：

### 第一優先
- `17` 的 `M1`
- `17` 的 `M2`
- `17` 的 `M3`
- `17` 的 `M4`

原因：
- 這四項會把「模型」正式變成平台資產
- 是後續一切 specialization 與 backend 擴充的共用前提

### 第二優先
- `17` 的 `M5 ~ M10`

原因：
- 這些讓 owned-model strategy 不只停留在 schema，而能進入真正可追蹤的工程流程

### 第三優先
- `18`

原因：
- 只有在 model asset layer 已建立後，第二個新 backend 才不會只是再堆一條孤立 runtime path

---

## 七、Definition Of Success

這份新架構真正落地，不是只看有沒有接上第二個模型，而是看平台是否達成以下條件：

- `Gemma` 已明確降格為 bootstrap model
- repo 內存在正式 `ModelSpec`
- `model_specs/` 成為模型資產來源
- factory 能用 `model_id`
- business logic 不再依賴 family-specific path
- `clone / judge / planner` 已有正式模型線規劃位置
- 新 backend 可在不破壞 shared architecture 的前提下接入

---

## 一句話結論

**16 定義方向，17 定義現在要補的工程基礎，18 定義未來如何擴新模型 family；正確落地順序是先做 17，再做 18，而不是反過來。**
