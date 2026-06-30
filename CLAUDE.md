【角色設定】
你是一位資深的 Python 後端工程師與電腦視覺專家。

【專案背景】
我目前有一個運行在 Ubuntu 上的工人 PPE（安全帽、反光背心）合規偵測專案。目前的架構包含：
1. `detector/yolo_engine.py`：負責 YOLO 雙模型推理與 ByteTrack 追蹤。
2. `ppe/compliance.py`：負責單幀的空間幾何過濾與裝備關聯。
3. `ppe/smoother.py`：基於 `track_id` 提供 120 幀滑動窗口的時序確認機制。
4. `ui/main_window.py`：目前使用 PySide6 顯示畫面與右側結果面板。

【任務目標】
請幫我將這個專案的執行入口從「桌面端 GUI (PySide6)」改造成「後台伺服器 (Web Server)」，並附帶一個「簡易的 Web 前端頁面」供我進行驗證與測試。請推薦使用 FastAPI 或 Flask 來實作。

【需求規格】

1. **影像來源支援**：
   - 支援讀取 USB 攝影機（例如：`cv2.VideoCapture(0)`）。
   - 支援讀取 RTSP 或一般 URL 影片串流。
   - 可以在 Web 前端頁面輸入來源並啟動。

2. **核心邏輯無縫接軌**：
   - 必須維持原有的 `engine.predict` -> `check_compliance` -> `smoother.update` 的資料流。
   - 呼叫原有的 `annotator.py` 在影像上畫框（Bounding Box）。

3. **警報與去重複機制 (Alert Debouncing)**：
   - 當 `smoother.update` 回傳的工人狀態為違規（例如 `NO_HELMET`, `NO_VEST`, `NO_PPE`）時，觸發提醒。
   - **請實作一個 `AlertManager`**：利用 `track_id` 控制發報頻率，確保同一位工人的同一個違規狀態，在特定時間內（例如 60 秒）只會觸發一次警報，避免「警報風暴」。

4. **簡易 Web 前端驗證頁面 (Testing Dashboard)**：
   - 提供一個簡單的 `index.html`（可以直接寫在 Python 字串中回傳，或獨立成 HTML 檔）。
   - **即時影像區塊**：實作 MJPEG 串流路由（例如 `/video_feed`），讓前端可以使用 `<img src="/video_feed">` 即時看到有畫框的影像。
   - **即時警報區塊**：實作 WebSocket 或 SSE (Server-Sent Events) 路由，當 `AlertManager` 觸發警報時，立刻推播文字訊息到前端頁面顯示（例如："Track #3 缺少安全帽"）。

【輸出要求】
1. 請提供完整的 `server.py`（包含 Web 路由、背景影像處理執行緒或非同步邏輯）。
2. 請提供 `AlertManager` 的類別實作與整合方式。
3. 請提供前端測試頁面 `index.html` 的程式碼。
4. 程式碼需具備良好的繁體中文註解，特別是在影像串流產生 (Generator) 與 WebSocket 推播結合的地方。