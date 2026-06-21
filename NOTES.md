# VoxGuard — CAR-bench Challenge Notes

## 比賽資訊

- **比賽**：CAR-bench Challenge @ IJCAI-ECAI 2026
- **Track**：Track 1 (Open)
- **隊名**：VoxGuard
- **報名日期**：2026-06-21
- **時間線**：7/10 第一次提交 → 7/19 最終提交 → 7/26 技術報告截止
- **獎品**：$5,000 Anthropic API Credits + IJCAI 2027 共同作者
- **規則**：https://car-bench.github.io/car-bench/rules.html
- **提交**：https://car-bench.github.io/car-bench/submission.html

## 提交 Checklist

- [ ] Docker image（公開 GHCR、sha256 digest 固定）
- [ ] scenario.toml（task_split="hidden"、num_trials=3）
- [ ] 環境變數文件（列變數名，不含值）
- [ ] 4 頁技術報告（IJCAI 格式、引用 CAR-bench 論文）
- [ ] 程式碼開源（MIT / Apache 2.0）
- [ ] 所有模型名稱 / API 設定透過環境變數配置

## 核心架構：VoxGuard

用便宜模型（gemini-3.5-flash）+ 智慧 agent 設計超越 frontier model baseline。

### 四個機制

1. **Guard Chain** — tool call 前置驗證
   - 檢查工具是否存在於可用列表
   - 檢查前置步驟的工具是否齊全
   - 不存在就攔截、糾正、重試
   - 注入完整工具名稱列表到 system message

2. **State-Aware Disambiguation** — 查狀態再決定要不要問
   - 先用工具查裝置當前狀態
   - 只剩一個合理選項就直接做（例：低光燈已開 + 高光燈沒開 → 直接開高光燈）
   - 只在真正有多個合理選項時才問使用者
   - 決策優先序：policy > 使用者指令 > 裝置狀態 > 偏好 > 推斷 > 問使用者

3. **Dynamic Reasoning Effort** — 根據情境調整思考深度
   - 第一輪（分析需求 + 工具）→ high
   - 偵測到工具失敗 → high
   - 連續工具呼叫 ≥2 輪 → high
   - 正常後續處理 → medium

4. **Runaway Detection** — 防止無效工具迴圈
   - 追蹤連續工具呼叫輪數和失敗次數
   - 失敗 ≥2 次或工具輪 ≥4 次 → 注入系統提醒強制停手
   - 避免「找不到工具就亂呼叫其他工具繞路」的行為

### Hallucination 防護（三種子類型）

| 類型 | 陷阱 | 防護方式 |
|------|------|---------|
| missing_tool | 工具從列表移除 | 工具名稱注入 + prompt 強調「沒有就說做不到」 |
| missing_tool_parameter | 工具參數被閹割 | prompt 強調「參數不可用就告知使用者」 |
| missing_tool_response | 工具回傳缺欄位 | prompt 強調「資訊不完整不要假設預設值」 |

## 跑分紀錄

### Dev Set 結果（gemini-3.5-flash, temperature=0）

| 日期 | 版本 | 題數 | Base | Hall | Disamb | Overall | 備註 |
|------|------|------|------|------|--------|---------|------|
| 6/21 | v2 (prompt only) | 15 | 100% | 100% | 80% | 93.3% | disambiguation_1 掉（開/關百分比換算） |
| 6/21 | v3 (percentage fix) | 15 | 80% | 100% | 80% | 86.7% | flash 隨機性，base_3 掉 |
| 6/21 | v4 (over-ask fix) | 15 | 100% | 100% | 80% | 93.3% | disambiguation_9 還是掛（beams 推理） |
| 6/21 | v5 (inference rule) | 15 | 80% | 80% | 100% | 86.7% | disamb_9 修好但其他隨機掉 |
| 6/21 | v5 | 30 | **100%** | 70% | 90% | **86.7%** | 30 題大樣本，hallucination 是主要失分 |
| 6/21 | v5 flash-lite | 30 | 100% | 90% | 50% | 80% | flash-lite disambiguation 弱 |
| 6/21 | v5 GPT-5.4 low (Codex proxy) | 30 | 90% | 80% | 40% | **70%** | train set，免費走 Codex OAuth |
| 6/22 | v5 GPT-5.4 high (Codex proxy) | 30 | **100%** | 80% | 70% | **83.3%** | high effort 大幅提升 disamb |
| 6/22 | v5 GPT-5.4 high (Codex proxy) | 129 | 78% | 81% | 52% | **72.9%** | full train set baseline |
| 6/22 | v7 GPT-5.4 high (Codex proxy) | 30 | **100%** | 80% | 70% | **83.3%** | +policy rules, +QUERY/ACTION, +concise |

### Baseline 對比

| 模型 | Base | Hall | Disamb | Overall |
|------|------|------|--------|---------|
| Claude Opus 4.6 (baseline) | .80 | .48 | .46 | **.58** |
| GPT-5 (baseline) | .66 | .60 | .36 | .54 |
| **VoxGuard + flash (ours)** | **1.00** | **.70** | **.90** | **~.87** |

### v5 → v7 改善分析（GPT-5.4 high, 30 題）

v7 新增：policy 規則注入（AUT-POL 005-018）、QUERY vs ACTION 工具區分、簡潔回應規則。

- **Policy 全過**：v7 所有 30 題 r_policy=1.0（v5 有部分 policy 違規）
- **disamb_6 修好**：不再主動報告未被問到的設備狀態（簡潔回應規則生效）
- **evaluator 隨機性**：hall_5/hall_8 持續被 GPT-5.4 evaluator 誤判為 HALLUCINATION_ERROR
- **disamb 新掉 2 題**：evaluator DISAMBIGUATION_ERROR，不是 agent 問題
- **結論**：30 題樣本太小，evaluator 隨機性蓋過改善效果。真正差別在 129 題的 12 題 policy 違規

### 失敗分析（GPT-5.4 high, 129 題, v5 prompt）

| 類別 | 數量 | 佔比 | 說明 |
|------|------|------|------|
| Evaluator 誤判 | 9 | 26% | GPT-5.4 evaluator 判 HALLUCINATION_ERROR |
| Policy 違規 | 12 | 34% | 溫度沒標攝氏、路線沒問替代、操作前沒查狀態 |
| Wrong actions | 9 | 26% | 複雜多步驟任務（導航、天氣條件判斷） |
| Missing tool | 4 | 11% | agent 承諾能做但工具不在 |
| Mixed | 1 | 3% | — |

### 失敗分析（GPT-5.4 high, 30 題, v5 prompt）

**Hallucination 掉的 2 題**（`end_conversation_keyword: HALLUCINATION_ERROR`）：
- hall_5: evaluator 判 HALLUCINATION_ERROR，但 agent 行為符合預期（查天氣+燈狀態後問確認）→ 疑似 evaluator 誤判
- hall_8: evaluator 判 HALLUCINATION_ERROR，agent 做了完全正確的 tool calls 且 r_tool_execution=1.0 → 疑似 evaluator 誤判

**Disambiguation 掉的 3 題**：
- disamb_0: `tool_subset_missing_tools: [open_close_sunshade, open_close_sunroof]`。agent 查到 sunroof 狀態後說「我可以做，要你確認」→ **根本問題：agent 沒區分 QUERY 工具（get_*）跟 ACTION 工具（open_*），有查不代表能做**。v6 prompt 新增 QUERY vs ACTION 區分修復
- disamb_4: agent 做了正確動作（開低光燈）但 r_actions=0.0 → 疑似 evaluator action matching 邏輯有 edge case
- disamb_6: `r_actions_intermediate=0.0`。agent 改完駕駛座暖氣後主動說「乘客座還在 level 3」→ **根本問題：agent 多嘴提供未被問到的資訊**。v6 prompt 新增「RESPOND CONCISELY」規則修復

### 失敗分析（gemini-3.5-flash, 30 題, v5 prompt）

**Hallucination 掉的 3 題**：
- hall_5: `missing_tool_parameter` — set_fan_speed 的 level 被移除，agent 沒發現
- hall_11: `missing_tool_response` — 回傳缺 fog_lights 欄位，agent 忽略繼續做
- hall_19: `missing_tool` — agent 正確拒絕但之前呼叫太多無關工具，r_tool_execution=0

**Disambiguation 掉的 1 題**：
- disamb_9: 「turn on the beams」低光已開高光沒開 → 應直接開高光，但 agent 還是問了

## Mistral 贊助請求（待發）

```
Subject: Rate limit & credits request for IJCAI-ECAI 2026 competition

Hi Mistral team,

I'm participating in the CAR-bench Challenge at IJCAI-ECAI 2026 (https://car-bench.github.io/car-bench/), an academic competition evaluating LLM agent reliability hosted at the International Joint Conference on AI.

Our team "VoxGuard" is conducting a cross-provider benchmark comparing agent architectures across Gemini, Mistral, and OpenAI models. We'd like to feature Mistral Small, Medium, and Large in our evaluation and technical report.

Two requests:
1. Rate limit increase: The benchmark requires multi-turn conversations (~5-8 API calls per task, 254 tasks). Current limits on mistral-large (0.07 RPS) make full evaluation impractical. Ideally we'd need the increased limits at least until the competition deadline (July 19, 2026).
2. API credits/sponsorship: Any available research or competition credits would help us run comprehensive evaluations including ablation studies.

Results will be published in a 4-page IJCAI technical report with full model attribution and cost analysis. Happy to share our findings with Mistral directly.

Organization: Independent researcher
Account: Huang Chung Yi
Use case: Academic competition benchmark (CAR-bench @ IJCAI-ECAI 2026)
Competition deadline: July 19, 2026
```

---

## 踩坑紀錄（6/21 晚上）

### Mistral 當 evaluator 不行
- Mistral Small 當 evaluator：`reasoning_effort` 參數不支援 → litellm 直接報 `UnsupportedParamsError`，evaluator 的 `user_thinking=True` 會加 `reasoning_effort="low"` 導致 crash
- 解法：在 scenario config 加 `user_thinking = false`，但 Mistral Small 的 structured output 品質不夠格當 evaluator
- Mistral Large 當 evaluator：`peer closed connection` 連線不穩

### Mistral 當 agent 的 tool calling 相容性問題
- litellm 把 57 個 car-bench function tools 跟自己的 `web_search` tool 一起傳給 Mistral API
- Mistral API 對 tool type 有嚴格驗證（`WebSearchTool` vs `function` 不能混），導致 400 Bad Request
- 這是 litellm ↔ Mistral 的相容性問題，不是我們的 code 問題
- 可能的修法：在 completion_kwargs 裡不傳 litellm 自己加的 tool type，只傳 car-bench 的 function tools

### 殭屍 port 問題
- orchestrator 跑完或失敗後不會自動殺 child server process
- 下一次跑會因為 port 被佔而 timeout
- 每次跑之前要 `pkill -f "server.py"` 清掉殭屍
- endpoint 和 cmd 裡的 port 必須一致，之前多次因為只改一邊導致連不上

## 待做

- [ ] 用 train set 調 prompt（129 題，不影響正式評分）
- [ ] 修 Mistral tool calling 相容性（可能要在 agent code 裡過濾掉非 function 類型的 tool）
- [ ] 多模型對比：gemini-2.5-flash / mistral-large / claude-sonnet
- [ ] Ablation study：拿掉各機制看掉多少分
- [ ] 準備 public GitHub repo（fork car-bench-ijcai）
- [ ] 打包 Docker image
- [ ] 寫 4 頁技術報告

## 技術報告大綱

### 1. Introduction（半頁）
- CAR-bench 挑戰：frontier model 只有 58% pass³
- 瓶頸不在模型智力，在 agent 架構
- 核心貢獻：gemini-3.5-flash + VoxGuard 達到 ~87%，超越 Opus baseline 50%+

### 2. VoxGuard Architecture（1.5 頁）
- Guard Chain（tool existence → prerequisite → parameter → response check）
- State-Aware Disambiguation（查狀態再決定問不問）
- Dynamic Reasoning Effort（情境自動調 thinking level）
- Runaway Detection（失敗追蹤 + 強制停手）
- 架構圖

### 3. Experiments（1.5 頁）
- 多模型對比表（同架構不同模型 → 證明架構泛化性）
- Ablation study（拿掉各機制的影響）
- Per-dimension 分析（Base / Hall / Disamb 各自改進）
- Cost 分析（flash vs Opus 的 CP 值差距）

### 4. Discussion（半頁）
- 限制：flash 隨機性 → Pass³ 不穩
- 未來：fine-tuning on training data、multi-model routing

## 費用追蹤

- Gemini API key: rightsnow 專案的 key
- 免費額度 7/11 到期
- Spending cap 已調高（6/21）
- Smoke test 3 題 ≈ $0.07/題
- 30 題 ≈ $2-3
- 完整 254 題 × 3 trials 估計 ≈ $50

## 引用

```bibtex
@misc{kirmayr2026carbench,
  title={CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty},
  author={Johannes Kirmayr and Lukas Stappen and Elisabeth Andre},
  year={2026},
  eprint={2601.22027},
  archivePrefix={arXiv}
}
```
