【角色設定】
你是一個專業的「電腦視覺工程師」，精通 Python、YOLO、OpenCV 以及多目標追蹤演算法。

【任務目標】
請為我撰寫一段完整的 Python 類別（Class）或腳本。系統將即時接收 YOLO 模型的偵測數據（包含 `person`, `helmet`, `reflective_vest` 的絕對像素座標 `x1, y1, x2, y2` 與 confidence），透過空間關聯邏輯判斷工人裝備合規性，並具備時序防閃爍與多目標追蹤機制。

【核心演算法與邏輯規範】

1. 多目標追蹤 (Multi-Object Tracking)：
   - 請提供整合 ByteTrack 或 DeepSORT 的程式碼框架。
   - 每一幀的 `person` 偵測框必須先餵給追蹤器，取得穩定的 `Track ID`。後續的合規性判斷與時序過濾皆綁定此 `Track ID`。

2. 邊界情況過濾 (Edge Cases & Filtering)：
   - **過濾過小物件**：若 `person` 框的高度（`y2 - y1`）小於 30 像素，直接丟棄不處理（距離過遠）。
   - **過濾過大或異常物件**：
     - 若 `person` 框的面積超過輸入畫面總面積的 70%，視為距離過近或誤判，直接忽略。
     - 若 `person` 框的長寬比異常（如 `(x2 - x1) / (y2 - y1) > 1.5`），視為非正常站立狀態或誤判，直接忽略。
   - **孤立裝備過濾**：未成功關聯到任何 `person` ROI 的 `helmet` 或 `vest`，視為背景物件，不予處理。

3. ROI 空間關聯 (Spatial Association)：
   - 針對通過過濾的 `person` 框，依據其高度進行垂直區域切割：
     - **Head ROI**：該 `person` 框的頂部 1/3 區域（`y1` 到 `y1 + height/3`）。
     - **Torso ROI**：該 `person` 框的中段 1/2 區域（從 `y1 + height/3` 開始向下延伸）。
   - **判定條件**：計算裝備框（`helmet` 或 `vest`）的中心點 `(cx, cy)`。
     - 若 `helmet` 的中心點落在該工人的框內，視為該工人「單幀配戴安全帽」。
     - 若 `reflective_vest` 的中心點落在該工人的框內，視為該工人「單幀穿戴反光背心」。

4. 合規狀態判定 (Compliance Status)：
   每個 `Track ID` 在單幀中會得到以下四種狀態之一：
   - `COMPLIANT`：同時關聯到安全帽與反光背心。
   - `NO_HELMET`：有反光背心，但無安全帽。
   - `NO_VEST`：有安全帽，但無反光背心。
   - `NO_PPE`：兩者皆無。

5. 滑動窗口確認機制與記憶體管理 (Temporal Smoothing & Memory Cleanup)：
   - 使用 `dict` 搭配 `collections.deque(maxlen=120)` 來儲存每個 `Track ID` 過去 120 幀的單幀狀態。
   - **觸發警告條件**：只有當某一特定的違規狀態（例如 `NO_HELMET`）在過去 120 幀中累積出現 >= 100 幀時，才正式確認該狀態並觸發違規警告輸出。
   - **記憶體清理**：若某個 `Track ID` 已經連續 300 幀未出現在追蹤器的活躍列表內（代表工人已離開畫面），必須將該 ID 從快取字典中完全移除，避免記憶體洩漏。

【輸出要求】
1. 請以物件導向 (Class) 的方式實作（例如 `ConstructionSafetyMonitor`），結構需清晰、解耦。
2. 在關鍵邏輯處（如 ROI 切割計算、中心點判定、時序計數、記憶體清理）加上詳細的繁體中文註解。
3. 請預留一個 `visualize` 函數，使用 OpenCV 在影像上繪製工人的 `Track ID`、各個 ROI 框，並用文字標註其最终確認的合規狀態。