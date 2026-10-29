import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import cv2
from PIL import Image
import pytesseract
import re

# ====================== Matplotlib 全局配置 ======================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

st.set_page_config(page_title="高频电子线路虚拟仿真与研判平台", layout="wide")

# ====================== 页面样式 ======================
st.markdown(
    """
    <style>
    .stRadio > label {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    .stRadio div[role="radiogroup"] label {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("高频电子线路虚拟仿真与研判平台")
st.header("**AM调幅实验**")

menu = st.radio(
    "模块选择",
    ["A模块：波形仿真", "E模块：智能研判"],
)


# ====================== 图像处理工具函数【升级OCR区域裁剪+文字增强】 ======================
def remove_reflection(img_cv):
    """抗反光预处理：降低高光、增强波形线条和文字对比度。"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.absdiff(gray, blur)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.convertScaleAbs(norm, alpha=1.8, beta=10)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh


def find_oscilloscope_screen(img_cv):
    """通过轮廓检测找到示波器屏幕区域。"""
    processed = remove_reflection(img_cv)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)
    if best_rect is not None:
        x, y, w, h = best_rect
        return img_cv[y:y+h, x:x+w]
    return img_cv


def perspective_correction(img_cv):
    """透视矫正：将倾斜拍摄的示波器屏幕自动转正为矩形。"""
    h, w = img_cv.shape[:2]
    processed = remove_reflection(img_cv)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        pts = pts[np.argsort(pts[:, 0])]
        left = pts[:2]
        right = pts[2:]
        left = left[np.argsort(left[:, 1])]
        right = right[np.argsort(right[:, 1])]
        src = np.array([left[0], right[0], left[1], right[1]], dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(img_cv, M, (w, h))
        return corrected
    return img_cv


def crop_scope_text_region(img_cv):
    """单独裁剪示波器底部文字档位区域，避开波形"""
    h, w = img_cv.shape[:2]
    text_img = img_cv[int(h * 0.85): h, :]
    return text_img


def enhance_text_img(img_cv):
    """文字图像增强：去摩尔纹、锐化、二值化，专门用于OCR识别"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
    return th


def parse_scope_settings(ocr_text):
    """容错解析档位，兼容μ/us 各种OCR识别错字"""
    ocr_text = ocr_text.replace("μ", "u").replace("µ", "u").replace("，", " ").replace(",", " ")
    # 匹配时间档位
    pat_time = re.search(r'(\d+\.?\d*)\s*(us|ms|s)', ocr_text.lower())
    # 匹配电压档位，排除误识别的VB
    pat_volt = re.search(r'(\d+\.?\d*)\s*(mv|v)(?!b)', ocr_text.lower())

    time_per_div = None
    volt_per_div = None
    unit_map_time = {
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1
    }
    if pat_time:
        val = float(pat_time.group(1))
        unit = pat_time.group(2)
        time_per_div = val * unit_map_time[unit]
    if pat_volt:
        val = float(pat_volt.group(1))
        unit = pat_volt.group(2)
        if unit == "mv":
            val = val / 1000
        volt_per_div = val
    return time_per_div, volt_per_div


def extract_waveform_from_image(corrected_img, time_per_div, volt_per_div):
    """从矫正后的图片提取波形曲线"""
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.absdiff(gray, blur)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.convertScaleAbs(norm, alpha=2.0, beta=15)
    _, thresh = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY_INV)

    h_img, w_img = thresh.shape
    x_pixel = np.arange(w_img)
    y_pixel = np.zeros(w_img)

    for x in range(w_img):
        col = thresh[:, x]
        idx = np.where(col > 0)[0]
        if len(idx) > 0:
            y_pixel[x] = np.mean(idx)
        else:
            y_pixel[x] = np.nan
    # 插值补断线
    mask_nan = np.isnan(y_pixel)
    y_pixel[mask_nan] = np.interp(x_pixel[mask_nan], x_pixel[~mask_nan], y_pixel[~mask_nan])

    # 像素坐标转为物理量
    t_meas = x_pixel / w_img * (10 * time_per_div)
    sig_meas = ((h_img / 2 - y_pixel) / (h_img / 10)) * volt_per_div
    return t_meas, sig_meas


# ====================== A模块：波形仿真 ======================
if menu == "A模块：波形仿真":
    st.sidebar.header("**参数调节区**")
    fc = st.sidebar.number_input("载波频率 fc (Hz)", value=450000, min_value=100, max_value=20000000)
    fm = st.sidebar.number_input("调制信号频率 fm (Hz)", value=1000, min_value=1, max_value=20000)
    m = st.sidebar.slider("调制系数 m", 0.0, 2.0, 0.6, 0.01)

    sample_per_carrier = 20
    show_mod_cycle = 2
    t_end = show_mod_cycle / fm
    sample_rate = fc * sample_per_carrier
    max_points = 15000
    n_points = int(sample_rate * t_end)
    if n_points > max_points:
        n_points = max_points
    t = np.linspace(0, t_end, n_points)

    carrier = np.cos(2 * np.pi * fc * t)
    mod_signal = np.cos(2 * np.pi * fm * t)
    am_signal = (1 + m * mod_signal) * carrier

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    ax1.plot(t, mod_signal, color="#2E86AB")
    ax1.set_title("调制信号")
    ax1.set_ylabel("幅度")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, carrier, color="#A23B72")
    ax2.set_title("载波信号")
    ax2.set_ylabel("幅度")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, am_signal, color="#F18F01")
    ax3.set_title(f"AM调幅信号，调制度 m={m}")
    ax3.set_xlabel("时间 t (s)")
    ax3.set_ylabel("幅度")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig)


# ====================== E模块：智能研判 ======================
elif menu == "E模块：智能研判":
    st.info("**探究任务：对比实测波形与理想AM波形，判断波形类别**")
    st.markdown(
        "**操作说明：**\n"
        "① CSV模式：上传示波器导出CSV；\n"
        "② 拍照模式：上传示波器屏幕照片，程序自动裁剪、透视矫正，识别档位；OCR识别不稳定时可手动填写档位。"
    )
    with st.sidebar:
        st.header("**输入参数区**")
        ref_sel = st.radio(
            "**预设波形类别（用于对照判定）**",
            ["m=0.6正常调幅", "m=1.0临界调幅", "m=1.4过调幅"]
        )
        input_mode = st.radio("数据来源", ["CSV文件导出", "拍照/图片识别波形"])
        fc_e = st.number_input("载波频率(Hz)", value=465000)
        fm_e = st.number_input("调制信号频率(Hz)", value=2000)

    # -------- CSV分支 --------
    if input_mode == "CSV文件导出":
        upload_file = st.file_uploader("上传示波器CSV文件（两列：时间，电压）", type="csv")
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file)
                if len(df) < 2:
                    st.error("❌ CSV数据过少！")
                else:
                    t_meas = df.iloc[:, 0].values
                    sig_meas = df.iloc[:, 1].values
                    analytic = hilbert(sig_meas)
                    envelope = np.abs(analytic)
                    env_max = np.max(envelope)
                    env_min = np.min(envelope)
                    ma_estimate = (env_max - env_min) / (env_max + env_min)

                    t_start = np.min(t_meas)
                    t_end_csv = np.max(t_meas)
                    t_ref = np.linspace(t_start, t_end_csv, 20000)
                    if ref_sel == "m=0.6正常调幅":
                        am_ref = (1 + 0.6 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                        ref_m = 0.6
                        ref_name = "正常调幅"
                    elif ref_sel == "m=1.0临界调幅":
                        am_ref = (1 + 1.0 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                        ref_m = 1.0
                        ref_name = "临界调幅"
                    else:
                        am_ref = (1 + 1.4 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                        ref_m = 1.4
                        ref_name = "过调幅"

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
                    ax1.plot(t_meas * 1000, sig_meas, label="实测波形")
                    ax1.plot(t_meas * 1000, envelope, "r--", label="包络")
                    ax1.set_title("实测波形与包络")
                    ax1.set_xlabel("时间(ms)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    ax2.plot(t_ref * 1000, am_ref, color="orange", label=f"参考 {ref_sel}")
                    ax2.set_title("理想参考波形")
                    ax2.set_xlabel("时间(ms)")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    st.subheader("🤖研判结果")
                    st.write(f"估算实测调制度 ma = {ma_estimate:.3f}")
                    if ref_m <1:
                        if abs(ma_estimate - ref_m) < 0.15:
                            st.success(f"✅匹配。上传波形为{ref_name}，和选择类别一致。")
                        else:
                            st.error(f"❌不匹配。你选择的是{ref_name}，当前上传波形不是，请上传对应类别的波形。")
                    elif ref_m ==1.0:
                        if abs(ma_estimate - ref_m) <0.15:
                            st.success(f"✅匹配。上传波形为{ref_name}，和选择类别一致。")
                        else:
                            st.error(f"❌不匹配。你选择的是{ref_name}，当前上传波形不是，请上传对应类别的波形。")
                    else:
                        #过调幅不依赖ma估算，用包络最小值判断
                        if env_min < 0.15 * env_max:
                            st.success(f"✅匹配。上传波形为{ref_name}，和选择类别一致。")
                        else:
                            st.error(f"❌不匹配。你选择的是{ref_name}，当前上传波形不是，请上传对应类别的波形。")
            except Exception as err:
                st.error(f"文件解析失败：{err}")
                st.info("CSV文件要求：第1列时间，第2列电压，不要多余表头")

    # -------- 拍照图片分支 --------
    else:
        st.markdown("##### 方式1：摄像头拍摄示波器屏幕")
        img_cam = st.camera_input("拍摄示波器屏幕，保证底部档位文字完整")
        st.markdown("##### 方式2：上传图片文件")
        img_upload = st.file_uploader("上传示波器截图/照片", type=["jpg", "png", "jpeg"])
        img_buffer = img_cam if img_cam else img_upload
        if img_buffer is not None:
            img_pil = Image.open(img_buffer)
            st.image(img_pil, caption="原始图片", use_container_width=True)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            #图像流水线：裁剪屏幕→透视矫正
            crop_img = find_oscilloscope_screen(img_cv)
            corrected_img = perspective_correction(crop_img)
            st.image(corrected_img, caption="自动裁剪并透视矫正后的示波器屏幕", use_container_width=True)

            # 单独截取底部文字区域做OCR
            text_region = crop_scope_text_region(corrected_img)
            text_enh = enhance_text_img(text_region)
            ocr_raw = pytesseract.image_to_string(text_enh, lang="chi_sim+eng")
            st.markdown(f"OCR识别原文：`{ocr_raw}`")
            time_div, volt_div = parse_scope_settings(ocr_raw)

            st.markdown("### 档位确认（OCR识别失败可手动填写兜底）")
            col1, col2 = st.columns(2)
            with col1:
                if time_div is not None:
                    time_per_div = st.number_input("每格时间(s)", value=float(time_div), format="%.7f")
                else:
                    time_per_div = st.number_input("每格时间(s)", value=200e-6, format="%.7f")
            with col2:
                if volt_div is not None:
                    volt_per_div = st.number_input("每格电压(V)", value=float(volt_div))
                else:
                    volt_per_div = st.number_input("每格电压(V)", value=2.0)

            #提取波形
            t_meas, sig_meas = extract_waveform_from_image(corrected_img, time_per_div, volt_per_div)
            analytic = hilbert(sig_meas)
            envelope = np.abs(analytic)
            env_max = np.max(envelope)
            env_min = np.min(envelope)
            ma_estimate = (env_max - env_min) / (env_max + env_min)

            t_start = np.min(t_meas)
            t_end_img = np.max(t_meas)
            t_ref = np.linspace(t_start, t_end_img, 20000)
            if ref_sel == "m=0.6正常调幅":
                am_ref = (1 + 0.6 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                ref_m = 0.6
                ref_name = "正常调幅"
            elif ref_sel == "m=1.0临界调幅":
                am_ref = (1 + 1.0 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                ref_m = 1.0
                ref_name = "临界调幅"
            else:
                am_ref = (1 + 1.4 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                ref_m = 1.4
                ref_name = "过调幅"

            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,7), constrained_layout=True)
            ax1.plot(t_meas*1000, sig_meas, label="提取实测波形")
            ax1.plot(t_meas*1000, envelope, "r--", label="包络")
            ax1.set_title("图片提取实测波形与包络")
            ax1.set_xlabel("时间(ms)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(t_ref*1000, am_ref, color="orange", label=f"参考 {ref_sel}")
            ax2.set_title("理想参考波形")
            ax2.set_xlabel("时间(ms)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.subheader("🤖研判结果")
            st.write(f"估算实测调制度 ma = {ma_estimate:.3f}")
            if ref_m <1:
                if abs(ma_estimate - ref_m) < 0.15:
                    st.success(f"✅匹配。上传波形为{ref_name}，和选择类别一致。")
                else:
                    st.error(f"❌不匹配。你选择的是{ref_name}，当前上传波形不是，请上传对应类别的波形。")
            elif ref_m ==1.0:
                if abs(ma_estimate - ref_m) <0.15:
                    st.success(f"✅匹配。上传波形为{ref_name}，和选择类别一致。")
                else:
                    st.error(f"❌不匹配。你选择的是{ref_name}，当前上传波形不是，请上传对应类别的波形。")
            else:
                if env_min < 0.15 * env_max:
                    st.success(f"✅匹配。上传波形为{ref_name}，和选择类别一致。")
                else:
                    st.error(f"❌不匹配。你选择的是{ref_name}，当前上传波形不是，请上传对应类别的波形。")

