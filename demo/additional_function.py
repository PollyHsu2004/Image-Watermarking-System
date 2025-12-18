import cv2
import numpy as np
import os

def bgr_to_ycrcb(img_bgr):
    """
    手動實作 BGR -> YCrCb 轉換 (ITU-R BT.601)
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cr = (R-Y)*0.713 + 128
    Cb = (B-Y)*0.564 + 128
    """
    img_float = img_bgr.astype(np.float32)
    B, G, R = cv2.split(img_float)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128

    # 合併並保持 float 供後續運算
    return cv2.merge([Y, Cr, Cb])

def ycrcb_to_bgr(img_ycrcb):
    """
    手動實作 YCrCb -> BGR 轉換
    R = Y + 1.403*(Cr-128)
    G = Y - 0.714*(Cr-128) - 0.344*(Cb-128)
    B = Y + 1.773*(Cb-128)
    """
    Y, Cr, Cb = cv2.split(img_ycrcb)

    # 減去偏移量
    Cr_cen = Cr - 128.0
    Cb_cen = Cb - 128.0

    R = Y + 1.403 * Cr_cen
    G = Y - 0.714 * Cr_cen - 0.344 * Cb_cen
    B = Y + 1.773 * Cb_cen

    # 限制數值範圍並轉回 uint8
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    return cv2.merge([np.uint8(B), np.uint8(G), np.uint8(R)])


# --- 手寫 DSP 算法：Nearest Neighbor Resize (取代 cv2.resize) ---
def resize(img, target_size):
    """
    手動實作 Nearest Neighbor Interpolation (最鄰近插值)。
    邏輯參考自 resize.py，但針對 Python 效能進行了向量化優化 (Vectorization)。
    
    Args:
        img: 輸入圖片 (H, W, C) 或 (H, W)
        target_size: 目標尺寸 tuple (width, height) -> 注意 OpenCV 格式是 (w, h)
    """
    # 1. 取得原始與目標尺寸
    h_old, w_old = img.shape[:2]
    w_new, h_new = target_size
    
    # 2. 計算縮放比例 (Scale Factors)
    # 對應 resize.py: y_old = y_new / scale
    # 這裡我們反過來算 scaling factor 用於坐標映射
    x_scale = w_old / w_new
    y_scale = h_old / h_new

    # --- 方法 A: 雙層迴圈版 (邏輯完全同 resize.py，但 Python 跑起來極慢) ---
    # 如果你堅持要用原本 resize.py 的 for loop 寫法，請解開下面註解
    # 但在網頁上可能會因為跑太久而 Timeout (幾秒鐘 vs 幾毫秒的差別)
    """
    if len(img.shape) == 3:
        out = np.zeros((h_new, w_new, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((h_new, w_new), dtype=img.dtype)
        
    for y in range(h_new):
        for x in range(w_new):
            # Backward Warping + Rounding
            src_y = int(min(round(y * y_scale), h_old - 1))
            src_x = int(min(round(x * x_scale), w_old - 1))
            out[y, x] = img[src_y, src_x]
    return out
    """

    # --- 方法 B: 向量化版 (DSP 推薦寫法) ---
    # 這是把雙層迴圈變成矩陣運算，原理完全一樣，但速度快 100 倍
    
    # 1. 產生網格座標 (0 ~ h_new-1)
    # 對應 resize.py 中的 y_new, x_new 迴圈
    row_indices = np.arange(h_new)
    col_indices = np.arange(w_new)

    # 2. Backward Warping (反向映射)
    # 對應 resize.py: y_nearest = int(np.round(y_old))
    src_rows = (row_indices * y_scale).round().astype(int)
    src_cols = (col_indices * x_scale).round().astype(int)

    # 3. 邊界檢查 (Boundary Check)
    # 對應 resize.py: if (1 <= y_nearest <= h)...
    src_rows = np.clip(src_rows, 0, h_old - 1)
    src_cols = np.clip(src_cols, 0, w_old - 1)

    # 4. 像素賦值 (Pixel Assignment)
    # 利用 Numpy 的 Advanced Indexing 一次搬移所有像素
    # out[y, x] = img[src_y, src_x]
    out = img[src_rows[:, None], src_cols]

    return out