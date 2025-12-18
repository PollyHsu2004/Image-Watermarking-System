import os
import cv2
import numpy as np
import time 
from flask import Flask, render_template, request, redirect
from flask import url_for, send_file, flash, session, send_from_directory
from werkzeug.utils import secure_filename

from watermark_word import add_embossed_watermark, add_transparent_color_watermark
from watermark_word import add_text_watermark, measure_image_quality_roi, test_jpeg_compression

# --- Flask Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.secret_key = 'your_strong_secret_key_here_for_flash' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Unified Image Processing Function ---
def process_image(main_path, wm_path, user_text, wm_type, wm_size, wm_position):
    
    main_img = cv2.imread(main_path)
    
    if main_img is None:
        return None 

    main_width = main_img.shape[1]
    main_copy = main_img.copy()
    
    if wm_size == "大":
        target_width = int(main_width * 0.30)
    elif wm_size == "中":
        target_width = int(main_width * 0.20)
    else: 
        target_width = int(main_width * 0.10)
    
    if wm_type == "文字浮水印":
        if not user_text:
            return None
            
        processed_img, roi_coords = add_text_watermark(
            main_img, 
            user_text, 
            font_scale=target_width/250, 
            opacity=0.9, 
            wm_position=wm_position,
            color=(0, 0, 0),
            target_width=target_width 
        )

    elif wm_type in ["浮雕去背", "透明彩色"]:
        watermark_img = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
        if watermark_img is None:
            return None 

        if wm_type == "浮雕去背":
            processed_img, roi_coords = add_embossed_watermark(main_img, watermark_img, target_width, 
                                                   opacity=0.8, padding_value=20, wm_position=wm_position)
        
        elif wm_type == "透明彩色":
            processed_img, roi_coords = add_transparent_color_watermark(main_img, watermark_img, target_width, 
                                                            opacity=0.7, wm_position=wm_position)
    
    else:
        return None 

    if processed_img is not None:
        unique_filename = f"watermark_{int(time.time())}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], unique_filename)
        cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        psnr_original, ssim_original = measure_image_quality_roi(main_copy, processed_img, roi_coords)
        compressed_img_QF40 = test_jpeg_compression(processed_img, quality_factor=40)
        psnr_40, ssim_40 = measure_image_quality_roi(processed_img, compressed_img_QF40, roi_coords)
        compressed_img_QF30 = test_jpeg_compression(processed_img, quality_factor=30)
        psnr_30, ssim_30 = measure_image_quality_roi(processed_img, compressed_img_QF30, roi_coords)
        compressed_img_QF20 = test_jpeg_compression(processed_img, quality_factor=20)
        psnr_20, ssim_20 = measure_image_quality_roi(processed_img, compressed_img_QF20, roi_coords)

        psnr_results = [
        f"{psnr_original:.2f}",  
        f"{psnr_40:.2f}",
        f"{psnr_30:.2f}",
        f"{psnr_20:.2f}"
        ]

        ssim_results = [
            f"{ssim_original:.2f}",
            f"{ssim_40:.2f}",
            f"{ssim_30:.2f}",
            f"{ssim_20:.2f}"
        ]
        # 返回一個包含所有結果的字典
        return {
            'path': output_path,
            'psnr': psnr_results,
            'ssim': ssim_results,
            'filename': unique_filename
        }
    return None

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. 獲取表單數據和檔案
        wm_type = request.form.get('wm_type')
        wm_size = request.form.get('wm_size')
        wm_position = request.form.get('wm_position')
        user_text = request.form.get('user_text')
        
        # 安全地獲取檔案物件，避免 KeyError
        main_file = request.files.get('main_image')
        wm_file = request.files.get('watermark_image') # 在文字模式下可能為 None

        # 2. 輸入驗證
        
        # 驗證主圖 (所有模式都需要主圖)
        if not main_file or main_file.filename == '' or not allowed_file(main_file.filename):
            flash("請選擇有效的主圖檔案，並確保格式為 .jpg, .jpeg 或 .png。", 'error')
            return redirect(request.url)

        # 驗證文字浮水印模式
        if wm_type == "文字浮水印":
            if not user_text:
                flash("選擇文字浮水印模式時，請輸入文字內容。", 'error')
                return redirect(request.url)
            # 在文字模式下，不需要檢查 wm_file
        
        # 驗證圖片浮水印模式 ("浮雕去背" 或 "透明彩色")
        elif wm_type in ["浮雕去背", "透明彩色"]:
            if not wm_file or wm_file.filename == '' or not allowed_file(wm_file.filename):
                flash("請上傳有效的浮水印圖檔案，並確保格式為 .jpg, .jpeg 或 .png。", 'error')
                return redirect(request.url)
        
        # 3. 儲存檔案
        
        # 儲存主圖
        main_filename = secure_filename(main_file.filename)
        main_path = os.path.join(app.config['UPLOAD_FOLDER'], f"main_{int(time.time())}_{main_filename}")
        main_file.save(main_path)

        # 儲存浮水印圖 (僅在需要時)
        wm_path = None
        if wm_type != "文字浮水印":
            wm_filename = secure_filename(wm_file.filename)
            wm_path = os.path.join(app.config['UPLOAD_FOLDER'], f"wm_{int(time.time())}_{wm_filename}")
            wm_file.save(wm_path)
            
        # 4. 執行核心處理
        output_file_path = process_image(main_path, wm_path, user_text, wm_type, wm_size, wm_position)
        
        # 5. 清理臨時上傳檔案
        try:
            os.remove(main_path)
            if wm_path:
                os.remove(wm_path)
        except Exception as e:
            print(f"Cleanup failed (uploaded files): {e}")

        if output_file_path: # output_file_path 現在是一個字典
            
            # 將陣列轉換為逗號分隔的字串
            psnr_str = ",".join(output_file_path['psnr'])
            ssim_str = ",".join(output_file_path['ssim'])

            flash("圖像處理成功！請查看結果。", 'success')
            
            # 導向新的 results 路由，並傳遞結果數據
            return redirect(url_for('results', 
                                    filename=output_file_path['filename'],
                                    psnr=psnr_str,
                                    ssim=ssim_str))
        else:
            flash("圖像處理失敗，請檢查圖片格式、內容或處理參數。", 'error')
            return redirect(request.url)

    # GET 請求時，渲染主頁面
    return render_template('index.html')


@app.route('/results/<filename>')
def results(filename):
    # 從 URL 參數中獲取結果字串
    psnr_str = request.args.get('psnr', '')
    ssim_str = request.args.get('ssim', '')

    # 將字串轉換回列表
    psnr_results = psnr_str.split(',')
    ssim_results = ssim_str.split(',')

    image_url = url_for('get_output_file', filename=filename)    
    # 確保所有訊息（如 'success'）能夠顯示
    
    return render_template('index.html', 
                           results_filename=filename,
                           image_url=image_url, 
                           psnr_data=psnr_results, 
                           ssim_data=ssim_results)

@app.route('/output_files/<filename>')
def get_output_file(filename):
    # send_from_directory 是 Flask 內建的安全方法
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        flash("找不到輸出檔案，可能已被刪除或處理失敗。", 'error')
        return redirect(url_for('index')) # 導回主頁面

    try:
        # 發送檔案給使用者
        response = send_file(file_path, as_attachment=True, download_name=filename)
        
        # 在檔案發送後執行清理
        @response.call_on_close
        def delete_file():
            try:
                os.remove(file_path)
                print(f"Cleaned up output file: {file_path}")
            except Exception as e:
                # 如果使用者下載失敗或中途取消，這裡可能會失敗，但這是正常的
                print(f"Cleanup failed for output file {filename}: {e}")
        
        return response
    except Exception as e:
        flash(f"下載時發生錯誤: {e}", 'error')
        return redirect(url_for('index')) # 導回主頁面


if __name__ == '__main__':
    print("Flask server starting...")
    app.run(debug=True)