import cv2
import numpy as np
import os
from additional_function import ycrcb_to_bgr, bgr_to_ycrcb, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 輔助函數：計算浮水印位置的 ROI 座標 ---
def get_roi_coordinates(main_img_shape, wm_img_shape, position):
    """根據位置參數計算浮水印應放置的起始 (x, y) 座標"""
    main_h, main_w = main_img_shape[:2]
    wm_h, wm_w = wm_img_shape[:2]
    
    # 邊緣留白 (Padding)
    pad = 30
    
    if position == "左上角":
        x = pad
        y = pad
    elif position == "右上角":
        x = main_w - wm_w - pad
        y = pad
    elif position == "左下角":
        x = pad
        y = main_h - wm_h - pad
    elif position == "右下角":
        x = main_w - wm_w - pad
        y = main_h - wm_h - pad
    elif position == "中央":
        x = (main_w - wm_w) // 2
        y = (main_h - wm_h) // 2
    else:
        # 預設左上角
        x = pad
        y = pad

    # 確保座標不為負
    x = max(0, x)
    y = max(0, y)
    
    # 裁剪浮水印，確保不超出主圖邊界
    wm_h = min(wm_h, main_h - y)
    wm_w = min(wm_w, main_w - x)
    
    return x, y, x + wm_w, y + wm_h, wm_h, wm_w


# --- 核心函數 A: 自適應浮雕去背處理 (基於您最後的優化版本) ---
def add_embossed_watermark(main_img, watermark_img, target_width, opacity, padding_value, wm_position):
    """執行自適應浮雕、Sobel邊緣提取、動態閾值去背，並混合到主圖上。"""
    
    try:
        # 修正點：檢查通道數
        if watermark_img.ndim == 3 and watermark_img.shape[2] == 3:
            # 輸入是 BGR (彩色)，安全地轉換為灰階
            gray_w = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2GRAY)
        elif watermark_img.ndim == 2 or watermark_img.shape[2] == 1:
            # 輸入已經是灰階 (單通道)，直接使用
            gray_w = watermark_img
        else:
            # 處理 BGRA (4通道) 圖片，先轉 BGR 再轉灰階 (不應該發生，但作為防禦)
            gray_w = cv2.cvtColor(watermark_img[:,:,:3], cv2.COLOR_BGR2GRAY)
        h, w = gray_w.shape[:2]
        target_height = int(target_width * (h / w))
        
        # 1. 浮水印尺寸計算和縮放
        resized_gray = resize(gray_w, (target_width, target_height))
        wm_h, wm_w = resized_gray.shape[:2]

        # 2. 自動計算 Background Threshold
        sample_size = 10 
        sample_areas = np.concatenate([
            resized_gray[:sample_size, :sample_size].flatten(), 
            resized_gray[:sample_size, wm_w - sample_size:].flatten(), 
            resized_gray[wm_h - sample_size:, :sample_size].flatten(), 
            resized_gray[wm_h - sample_size:, wm_w - sample_size:].flatten()
        ])
        
        mean_bg_lightness = np.mean(sample_areas)
        
        if mean_bg_lightness > 128:
            final_threshold = max(0, int(mean_bg_lightness - padding_value))
            threshold_type = cv2.THRESH_BINARY_INV # 淺色背景: 保留暗部
        else:
            final_threshold = min(255, int(mean_bg_lightness + padding_value))
            threshold_type = cv2.THRESH_BINARY # 深色背景: 保留亮部

        # 3. 建立 Alpha 遮罩 (去背)
        _, mask_binary = cv2.threshold(resized_gray, final_threshold, 1, threshold_type)
        resized_mask_float = mask_binary.astype(np.float32)
        final_alpha = resized_mask_float * opacity
        alpha_3_channel = cv2.merge([final_alpha, final_alpha, final_alpha])
        
        # 4. 浮雕效果模擬 (Sobel)
        grad_x = cv2.Sobel(resized_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_y = cv2.Sobel(resized_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        combined_grad = (grad_x + grad_y) / 2.0 
        adjust_factor = 0.5 
        scaled_grad = combined_grad * adjust_factor
        
        # 5. 計算 ROI 並疊加 (HSV 雕刻感)
        x1, y1, x2, y2, wm_h, wm_w = get_roi_coordinates(main_img.shape, (wm_h, wm_w), wm_position)
        
        # 裁剪 ROI 到實際浮水印大小
        roi_bgr = main_img[y1:y2, x1:x2].astype(np.float32)
        roi_hsv = cv2.cvtColor(roi_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(roi_hsv)
        
        grad_resized = resize(scaled_grad, (wm_w, wm_h))
        
        # 混合 V 通道 (亮度)
        v_blended = v + grad_resized
        v_final = v_blended * final_alpha[:wm_h, :wm_w] + v * (1.0 - final_alpha[:wm_h, :wm_w])
        v_final = np.clip(v_final, 0, 255)
        
        # 合併 H, S, V 通道並轉回 BGR
        blended_hsv = cv2.merge([h, s, v_final])
        blended_bgr = cv2.cvtColor(blended_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 將結果放回主圖
        main_img[y1:y2, x1:x2] = blended_bgr
        
        return main_img, (x1, y1, x2, y2)

    except Exception as e:
        print(f"浮雕處理失敗: {e}")
        return None

# --- 核心函數 B: 透明彩色疊加處理 (最終優化版本) ---
def add_transparent_color_watermark(main_img, watermark_img, target_width, opacity, wm_position):
    """
    執行穩健的彩色浮水印疊加：使用浮雕函數的智能去背邏輯，並允許 Y 通道參與混合。
    """
    try:
        # 0. 參數設定 (調整這裡的 Y_MIX_RATIO 來控制浮水印的深度和對比度)
        Y_MIX_RATIO = 0.4  # <<--- 提高到 0.4，讓浮水印更實體化
        PADDING_VALUE = 20 # 繼承自浮雕函數，用於去背的緩衝值
        
        # 1. 浮水印尺寸計算和縮放
        h_wm_raw, w_wm_raw = watermark_img.shape[:2]
        target_height = int(target_width * (h_wm_raw / w_wm_raw))
        wm_resized = resize(watermark_img, (target_width, target_height))
        
        wm_alpha = None
        h_wm, w_wm = wm_resized.shape[:2]
        wm_bgr = wm_resized if wm_resized.ndim == 3 else cv2.cvtColor(wm_resized, cv2.COLOR_GRAY2BGR)

        # 2. **替換去背邏輯**：使用自適應角落採樣
        if wm_resized.shape[2] == 4:
            # PNG 內建 Alpha
            wm_alpha = wm_resized[:, :, 3] / 255.0
        else:
            # JPG/灰階 -> 智能動態閾值去背
            gray_wm = cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2GRAY)
            
            # (a) 邊角採樣計算平均背景亮度
            sample_size = 10 
            sample_areas = np.concatenate([
                gray_wm[:sample_size, :sample_size].flatten(), 
                gray_wm[:sample_size, w_wm - sample_size:].flatten(), 
                gray_wm[h_wm - sample_size:, :sample_size].flatten(), 
                gray_wm[h_wm - sample_size:, w_wm - sample_size:].flatten()
            ])
            mean_bg_lightness = np.mean(sample_areas)
            
            # (b) 設定自適應閾值和類型
            if mean_bg_lightness > 128:
                final_threshold = max(0, int(mean_bg_lightness - PADDING_VALUE))
                threshold_type = cv2.THRESH_BINARY_INV # 淺色背景：保留暗部
            else:
                final_threshold = min(255, int(mean_bg_lightness + PADDING_VALUE))
                threshold_type = cv2.THRESH_BINARY # 深色背景：保留亮部

            _, mask = cv2.threshold(gray_wm, final_threshold, 255, threshold_type)
            
            # (c) 邊緣柔化並正規化
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            wm_alpha = mask / 255.0

        # 3. 計算 ROI 位置並裁剪浮水印
        x1, y1, x2, y2, roi_h, roi_w = get_roi_coordinates(main_img.shape, (h_wm, w_wm), wm_position)
        
        # 裁剪浮水印和 Alpha 遮罩
        wm_bgr_crop = wm_bgr[:roi_h, :roi_w]
        wm_alpha_crop = wm_alpha[:roi_h, :roi_w] 
        roi_bgr = main_img[y1:y2, x1:x2]

        # 4. YCbCr 轉換與混合
        roi_ycrcb = bgr_to_ycrcb(roi_bgr)
        wm_ycrcb = bgr_to_ycrcb(wm_bgr_crop)

        roi_y, roi_cr, roi_cb = cv2.split(roi_ycrcb)
        wm_y, wm_cr, wm_cb = cv2.split(wm_ycrcb)

        # 最終混合強度 (色度)
        blend_factor = wm_alpha_crop * opacity 
        
        # Y 通道混合：確保亮度信息能顯示 (增強可見度)
        y_blend_factor = wm_alpha_crop * opacity * Y_MIX_RATIO 
        blended_y = roi_y * (1.0 - y_blend_factor) + wm_y * y_blend_factor

        # Cr/Cb 通道混合 (色度)
        blended_cr = roi_cr * (1.0 - blend_factor) + wm_cr * blend_factor
        blended_cb = roi_cb * (1.0 - blend_factor) + wm_cb * blend_factor

        # 5. 重建與輸出
        blended_y = np.clip(blended_y, 0, 255)
        blended_cr = np.clip(blended_cr, 0, 255)
        blended_cb = np.clip(blended_cb, 0, 255)

        res_ycrcb = cv2.merge([np.uint8(blended_y), np.uint8(blended_cr), np.uint8(blended_cb)])
        res_bgr = ycrcb_to_bgr(res_ycrcb)
        
        main_img[y1:y2, x1:x2] = res_bgr
        
        return main_img, (x1, y1, x2, y2)
        
    except Exception as e:
        print(f"YCbCr 處理失敗: {e}")
        return main_img

def add_text_watermark(main_img, text, font_scale, opacity, wm_position, color, target_width):
    """
    將文字浮水印疊加到主圖上，並根據背景亮度自動調整文字顏色。

    回傳: (final_img, roi_coords)
    """
    
    # 0. 變數初始化，防止 NameError
    # 預設深灰色作為備用顏色
    text_color = (50, 50, 50) 
    
    main_h, main_w = main_img.shape[:2]
    
    # 1. 計算文字粗細 (與 font_scale 比例固定)
    thickness_factor = 2 
    thickness = max(1, round(thickness_factor * font_scale)) 
    
    # 2. 計算文字大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    # (text_w, text_h) 是文字包圍盒的寬高，baseline 是從 y=0 到基線的距離
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 理論浮水印尺寸
    wm_h_theory = text_h + baseline 
    wm_w_theory = text_w
    
    # 3. 計算 ROI 位置 (使用 get_roi_coordinates 獲取理論位置)
    # x1_theory, y1_theory, x2_theory, y2_theory 是理論上的 ROI 座標
    x1_theory, y1_theory, x2_theory, y2_theory, roi_h, roi_w = get_roi_coordinates(
        main_img.shape, (wm_h_theory, wm_w_theory), wm_position
    )

    # ❗ 步驟 3.1: 修正 ROI 座標以精確對齊 cv2.putText ❗
    
    # cv2.putText 繪圖起點是文字的左下角 (baseline)
    text_x_start = x1_theory
    text_y_start = y1_theory + text_h # y1_theory (頂) + text_h = 基線 (baseline)
    
    # 實際用於裁剪和測量的 ROI 座標 (確保包含所有繪圖範圍)
    x1, y1 = x1_theory, y1_theory
    x2, y2 = x2_theory, y2_theory # 直接使用 get_roi_coordinates 處理邊界後的結果
    
    # [新增邏輯] 3.5. 根據背景亮度自動決定文字顏色
    # 定義深色和淺色的 BGR 基礎值 (作為浮水印的「光譜」兩端)
    COLOR_DARK = np.array([50, 50, 50], dtype=np.float32)   # 深灰色
    COLOR_LIGHT = np.array([200, 200, 200], dtype=np.float32) # 淺灰色
    
    # 檢查 ROI 區域是否有效 (防止切片失敗)
    if y2 > y1 and x2 > x1:
        # 取得 ROI 區域 (此 ROI 已經被 get_roi_coordinates 確保在邊界內)
        roi = main_img[y1:y2, x1:x2]
        
        # 確保 ROI 是有效的彩色圖像，再進行轉換
        if roi.size > 0 and roi.ndim == 3:
            # 1. 取得 ROI 區域的平均 BGR 顏色 (這是文字正下方的「底色」)
            average_orig_color_bgr = np.mean(roi, axis=(0, 1)).astype(np.float32)
            
            # 2. 轉換為灰度並計算平均亮度
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            average_brightness = np.mean(roi_gray)
            
            # 計算淺色權重 (讓背景越暗，權重越高，顏色越偏淺)
            weight_light = 1.0 - (average_brightness / 255.0)
            
            # 基礎灰色 (在深灰和淺灰之間過渡的顏色)
            base_gray_float = (COLOR_LIGHT * weight_light) + \
                            (COLOR_DARK * (1.0 - weight_light))
                            
            # 4. 進行原圖顏色與基礎灰色的最終混合
            
            # 混合公式： (基礎灰 * 基礎灰權重) + (原圖顏色 * 原圖權重)
            # 這裡我們使用一種簡單的「加權平均」，讓文字顏色同時具備原圖色彩和深淺灰的對比度。
            
            # 我們讓基礎灰色的色彩強度影響力高一些，以確保文字清晰可見 (例如 70%)
            BASE_WEIGHT = 0.9 
            ORIG_WEIGHT = 1.0 - BASE_WEIGHT
            
            text_color_float = (base_gray_float * BASE_WEIGHT) + \
                            (average_orig_color_bgr * ORIG_WEIGHT)

            # 5. 轉換為 BGR 整數元組
            text_color = tuple(np.uint8(text_color_float).tolist())
    
    # 4. 繪製文字浮水印
    overlay = main_img.copy()
    
    # print(text_color)
    # print(font_scale)

    cv2.putText(
        overlay, 
        text, 
        (text_x_start, text_y_start), # 使用精確計算的 baseline 繪圖起點
        font, 
        font_scale, 
        text_color, # 使用自動決定的文字顏色
        thickness,
        cv2.LINE_AA
    )
    
    # 5. 混合 (實現透明度)
    # final_img = main_img * (1 - opacity) + overlay * opacity (簡化版)
    final_img = cv2.addWeighted(main_img, 1.0 - opacity, overlay, opacity, 0)
    
    # 6. 回傳最終圖像和精確 ROI 座標
    return final_img, (x1, y1, x2, y2)

def measure_image_quality_roi(original_img, processed_img, roi_coords=None, save_roi_prefix=None):
    """計算兩張圖片之間在 ROI 區域的 PSNR 和 SSIM。"""
    
    if original_img.shape != processed_img.shape:
        return None, None
    
    # ------------------- 1. 裁剪 ROI -------------------
    if roi_coords is not None:
        x1, y1, x2, y2 = roi_coords
        
        orig_roi = original_img[y1:y2, x1:x2]
        proc_roi = processed_img[y1:y2, x1:x2]
        
        if orig_roi.size == 0 or proc_roi.size == 0:
            return None, None
    else:
        orig_roi = original_img
        proc_roi = processed_img
        
    # ------------------- 1.5. (新增) 儲存 ROI 圖片 -------------------
    if save_roi_prefix is not None:
        try:
            # 確保儲存的 ROI 圖片是 BGR 格式 (cv2.imwrite 預期格式)
            cv2.imwrite(f"{save_roi_prefix}_orig.jpg", orig_roi)
            cv2.imwrite(f"{save_roi_prefix}_proc.jpg", proc_roi)
            print(f"成功儲存 ROI 圖片至: {save_roi_prefix}_orig.jpg 和 {save_roi_prefix}_proc.jpg")
        except Exception as e:
            print(f"警告: 儲存 ROI 圖片失敗: {e}")
            
    # ------------------- 2. 轉換與正規化 -------------------
    orig_roi = orig_roi.astype(np.float64) / 255.0
    proc_roi = proc_roi.astype(np.float64) / 255.0

    # ------------------- 3. 計算 PSNR -------------------
    try:
        psnr_value = psnr(orig_roi, proc_roi, data_range=1.0)
    except Exception:
        psnr_value = 0.0

    # ------------------- 4. 計算 SSIM -------------------
    is_multichannel = len(orig_roi.shape) == 3
    
    win_size_limit = min(orig_roi.shape[0], orig_roi.shape[1])
    win_size = min(7, win_size_limit - 1)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        ssim_value = 0.0
    else:
        try:
            ssim_value = ssim(orig_roi, proc_roi, 
                              data_range=1.0, 
                              multichannel=is_multichannel, 
                              channel_axis=2 if is_multichannel else None,
                              win_size=win_size
                              )
        except Exception:
            ssim_value = 0.0
        
    return psnr_value, ssim_value

def test_jpeg_compression(img, quality_factor):
    """
    對帶浮水印圖片進行 JPEG 壓縮測試。
    
    參數:
    - img (np.ndarray): 待壓縮的圖像 NumPy 陣列 (通常是帶浮水印的原圖)。
    - quality_factor (int): JPEG 品質因子 (0-100，越高越好)。
    
    回傳:
    - np.ndarray | None: 經過 JPEG 壓縮和解壓縮後的圖像 NumPy 陣列，如果失敗則回傳 None。
    """

    if img is None:
        print("錯誤：輸入圖像為 None。")
        return None

    # 1. 進行 JPEG 編碼（壓縮）
    # 參數：cv2.IMWRITE_JPEG_QUALITY, 範圍 0-100 (越高越好)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    
    # 輸出為記憶體的緩衝區 (compressed_img_data 是一個位元組陣列)
    result, compressed_img_data = cv2.imencode('.jpg', img, encode_param)

    if result:
        # 2. 將壓縮後的數據解碼回圖片格式 (NumPy 陣列)
        compressed_img = cv2.imdecode(compressed_img_data, cv2.IMREAD_COLOR)
        
        # ❗ 移除 cv2.imwrite 的儲存邏輯 ❗
        
        # 3. 回傳 NumPy 陣列
        print(f"壓縮 QF={quality_factor} 測試完成，直接回傳 NumPy 陣列。")
        return compressed_img
    else:
        print("壓縮失敗。")
        return None