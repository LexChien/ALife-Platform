# planner Dataset Contract / planner 資料契約

ZH：階段限定 birth／split／fusion，每階段至少兩幀，target_components 必須對應 1／2／1。theta 上下界各五維、每維 low ≤ high，數值必須有限；next_step 為必填。計畫不會自動套用到模擬器。

EN: Phases are birth/split/fusion with at least two frames each and target counts of 1/2/1. Theta bounds contain five finite dimensions with low <= high in each dimension; next_step is required. Plans are not automatically applied to the simulator.

## Schema / 結構規格

ZH：機器可讀規格以 [`planner.schema.json`](planner.schema.json) 為準。每個 JSONL 行需獨立通過 Draft 7 與 `training.datasets.contracts.validate_row` 跨欄位檢查；拒絕額外欄位與非有限數字。

EN: The canonical machine-readable contract is [`planner.schema.json`](planner.schema.json). Each JSONL row must pass Draft 7 and the cross-field checks in `training.datasets.contracts.validate_row`; extra fields and nonfinite numbers are rejected.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PlannerTrainingRow",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "instruction": {
      "type": "string",
      "minLength": 1
    },
    "expected_phases": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "phase": {
            "type": "string",
            "enum": [
              "birth",
              "split",
              "fusion"
            ]
          },
          "steps": {
            "type": "integer",
            "minimum": 2
          },
          "target_components": {
            "type": "integer",
            "enum": [
              1,
              2
            ]
          }
        },
        "required": [
          "phase",
          "steps",
          "target_components"
        ],
        "allOf": [
          {
            "if": {
              "properties": {
                "phase": {
                  "const": "split"
                }
              }
            },
            "then": {
              "properties": {
                "target_components": {
                  "const": 2
                }
              }
            },
            "else": {
              "properties": {
                "target_components": {
                  "const": 1
                }
              }
            }
          }
        ]
      }
    },
    "hyperparameters": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "theta_low": {
          "type": "array",
          "minItems": 5,
          "maxItems": 5,
          "items": {
            "type": "number"
          }
        },
        "theta_high": {
          "type": "array",
          "minItems": 5,
          "maxItems": 5,
          "items": {
            "type": "number"
          }
        }
      },
      "required": [
        "theta_low",
        "theta_high"
      ]
    },
    "next_step": {
      "type": "string",
      "minLength": 1
    }
  },
  "required": [
    "instruction",
    "expected_phases",
    "hyperparameters",
    "next_step"
  ]
}
```

## Example / 範例

ZH：下列一筆樣本同時存於 `examples/planner.jsonl`，僅用於契約驗證，不是完成的訓練資料集。

EN: This single example also appears in `examples/planner.jsonl` for contract validation; it is not a completed training dataset.

```json
{
  "instruction": "Plan a cell division search over 60 frames.",
  "expected_phases": [
    {
      "phase": "birth",
      "steps": 20,
      "target_components": 1
    },
    {
      "phase": "split",
      "steps": 40,
      "target_components": 2
    }
  ],
  "hyperparameters": {
    "theta_low": [
      0,
      0,
      0,
      18,
      1
    ],
    "theta_high": [
      2,
      2,
      2,
      128,
      6.5
    ]
  },
  "next_step": "Validate the configuration, run fixed-seed search, then inspect lossless trajectories."
}
```
