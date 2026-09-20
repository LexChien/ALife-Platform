# clone Dataset Contract / clone 資料契約

ZH：以 persona system、檢索 context 與 user prompt 建立單回合樣本；response 必須保留執行期的 clone 格式。格式正確不代表長期人格穩定。

EN: Each row combines persona instructions, retrieved context, and a user prompt. The response uses the runtime clone format. Correct formatting does not prove long-term persona stability.

## Schema / 結構規格

ZH：機器可讀規格以 [`clone.schema.json`](clone.schema.json) 為準。每個 JSONL 行需獨立通過 Draft 7 與 `training.datasets.contracts.validate_row` 跨欄位檢查；拒絕額外欄位與非有限數字。

EN: The canonical machine-readable contract is [`clone.schema.json`](clone.schema.json). Each JSONL row must pass Draft 7 and the cross-field checks in `training.datasets.contracts.validate_row`; extra fields and nonfinite numbers are rejected.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DigitalCloneTrainingRow",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "system": {
      "type": "string",
      "minLength": 1
    },
    "context": {
      "type": "string"
    },
    "prompt": {
      "type": "string",
      "minLength": 1
    },
    "response": {
      "type": "string",
      "pattern": "^\\[[^\\]\\r\\n]+\\] tone=.+ principles=.+ response=.+$"
    }
  },
  "required": [
    "system",
    "prompt",
    "response"
  ]
}
```

## Example / 範例

ZH：下列一筆樣本同時存於 `examples/clone.jsonl`，僅用於契約驗證，不是完成的訓練資料集。

EN: This single example also appears in `examples/clone.jsonl` for contract validation; it is not a completed training dataset.

```json
{
  "system": "You are Lex Clone. Tone: calm, analytical. Principles: maintain consistency, prioritize clarity.",
  "context": "Lex Clone prefers structuring answers before details.",
  "prompt": "How do you approach code reviews?",
  "response": "[Lex Clone] tone=calm, analytical principles=maintain consistency, prioritize clarity response=I examine the architecture first, then check behavior against tests."
}
```
