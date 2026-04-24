# load_local_scene.py 使用說明

這份專案提供一支最小化示範腳本，用來在 Isaac Sim 中載入本地 USD 場景。

對應檔案：
- load_local_scene.py

## 功能

- 啟動 Isaac Sim 視窗模式（headless=False）
- 以本地路徑開啟 USD 場景
- 等待 Stage 載入完成後再開始模擬
- 建立 World 並進行固定秒數的渲染迴圈

## 需求

- 已安裝 Isaac Sim
- 可使用 Isaac Sim 的 Python 啟動器（環境變數 ISAACSIM_PYTHON）
- Linux 桌面環境（X11/Wayland 可顯示視窗）

## 執行方式

在專案目錄執行：

$ISAACSIM_PYTHON load_local_scene.py

## 設定場景路徑

請編輯 load_local_scene.py 內的路徑設定：

- BASE_DIR：腳本所在資料夾
- USD_PATH：要開啟的 USD 檔案

目前預設為相對路徑寫法：

USD_PATH = str(BASE_DIR / "20250403.usd")

你也可以改成子資料夾，例如：

USD_PATH = str(BASE_DIR / "assets" / "warehouse.usd")

建議優先使用相對路徑，專案搬移到其他機器時不需要改絕對路徑。

## 程式流程

1. 初始化 SimulationApp（必須早於 omni 模組 import）
2. 等待應用程式 ready
3. open_stage 載入 USD
4. 輪詢 stage_loading_status 直到載入完成
5. 建立 World 並 reset
6. 更新與渲染迴圈
7. 關閉 SimulationApp

## 常見問題

1. 執行後黑畫面或無畫面
- 確認不是在純 SSH 無 GUI 環境下執行
- 確認 DISPLAY 已設定（X11）
- 確認 headless=False

2. 卡在 Loading
- 若使用雲端資產路徑，第一次載入可能較久
- 本腳本已加入 wait_for_stage_loaded，會等待載入完成

3. 檔案找不到
- 檢查 USD_PATH 指向的檔案是否存在
- 建議先把 USD 放在專案資料夾內並使用相對路徑

## 教學用途建議

如果你要教學示範「從本地路徑匯入 USD 當場景」，建議流程：

1. 先展示專案資料夾中的 USD 檔案位置
2. 修改 USD_PATH 為相對路徑
3. 執行腳本
4. 解釋 wait_for_stage_loaded 為何必要（避免尚未載入完就進入模擬）

以上流程可以穩定重現，不依賴互動式選單。
