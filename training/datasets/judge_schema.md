# judge Dataset Contract / judge 資料契約

ZH：這是已量測形態特徵的文字評估契約，沒有原始影像輸入。score 必須介於 0 與 1，feedback 只存文字；序列化時才加上 score= 與 feedback=，避免重複前綴。

EN: This is a text evaluation contract for measured morphology features, without raw image input. Score must be between 0 and 1; feedback stores plain text. Serialization adds score= and feedback= exactly once.

## Schema / 結構規格

ZH：機器可讀規格以 [`judge.schema.json`](judge.schema.json) 為準。每個 JSONL 行需獨立通過 Draft 7 與 `training.datasets.contracts.validate_row` 跨欄位檢查；拒絕額外欄位與非有限數字。

EN: The canonical machine-readable contract is [`judge.schema.json`](judge.schema.json). Each JSONL row must pass Draft 7 and the cross-field checks in `training.datasets.contracts.validate_row`; extra fields and nonfinite numbers are rejected.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "JudgeTrainingRow",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "image_metadata": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "num_components": {
          "type": "integer",
          "minimum": 0
        },
        "largest_circularity_proxy": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      },
      "required": [
        "num_components",
        "largest_circularity_proxy"
      ]
    },
    "rubric": {
      "type": "string",
      "minLength": 1
    },
    "score": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "feedback": {
      "type": "string",
      "minLength": 1,
      "not": {
        "pattern": "^score="
      }
    }
  },
  "required": [
    "image_metadata",
    "rubric",
    "score",
    "feedback"
  ]
}
```

## Example / 範例

ZH：下列一筆樣本同時存於 `examples/judge.jsonl`，僅用於契約驗證，不是完成的訓練資料集。

EN: This single example also appears in `examples/judge.jsonl` for contract validation; it is not a completed training dataset.

```json
{
  "image_metadata": {
    "num_components": 1,
    "largest_circularity_proxy": 0.85
  },
  "rubric": "Prefer one coherent circular body; this row contains measured features, not raw visual input.",
  "score": 0.92,
  "feedback": "One component has a circularity proxy of 0.85; temporal continuity still needs trajectory evidence."
}
```
