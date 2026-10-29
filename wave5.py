import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
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


# ====================== 图像处理工具函数 ======================

def remove_reflection(img_cv):
    """抗反光预处理：降低高光、增强波形线条和文字对比度。"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 用大核均值估计屏幕反光区域
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.absdiff(gray, blur)

    # 归一化并增强对比度
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.convertScaleAbs(norm, alpha=1.8, beta=10)

    # 局部二值化，增强文字和波形线条
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

    # 如果近似为四边形，做透视变换
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


def parse_scope_settings(ocr_text):
    """从OCR文字中解析时间/电压档位。"""
    ocr_text = ocr_text.replace("，", " ").replace(",", " ")

    # 匹配时间档位：200μs、0.5ms、1ms、50us等
    pat_time = re.search(
        r'(\d+\.?\d*)\s*(us|μs|ms|s)',
        ocr_text.lower()
    )
    # 匹配电压档位：0.5V、1V、2V、5V等
    pat_volt = re.search(
        r'(\d+\.?\d*)\s*v',
        ocr_text.lower()
    )

    time_per_div = None
    volt_per_div = None

    unit_map_time = {
        "us": 1e-6,
        "μs": 1e-6,
        "ms": 1e-3,
        "s": 1
    }

    if pat_time:
        val = float(pat_time.group(1))
        unit = pat_time.group(2)
        time_per_div = val * unit_map_time[unit]

    if pat_volt:
        volt_per_div = float(pat_volt.group(1))

    return time_per_div, volt_per_div


def extract_waveform_from_image(crop_img, time_per_div, volt_per_div):
    """从裁剪并矫正后的示波器图片中提取波形像素曲线，并转换为时间/电压序列。"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 抗反光增强
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

    # 插值补全断线
    mask = np.isnan(y_pixel)
    y_pixel[mask] = np.interp(x_pixel[mask], x_pixel[~mask], y_pixel[~mask])

    # 像素坐标 -> 物理量
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7))
    ax1.plot(t, mod_signal, color="#2E86AB")
    ax1.set_title("Modulating Signal")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, carrier, color="#A23B72")
    ax2.set_title("Carrier Signal")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, am_signal, color="#F18F01")
    ax3.set_title(f"AM Modulated Waveform (m = {m:.2f})")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time t (s)")
    ax3.grid(True, alpha=0.3)

    fig.subplots_adjust(top=0.92, hspace=0.45)
    st.pyplot(fig)


# ====================== E模块：智能研判 ======================
elif menu == "E模块：智能研判":
    st.info("**探究任务：对比实测波形与理想波形差异，校验实测波形是否正确。**")
    st.markdown(
        "**操作说明：**\n"
        "① CSV模式：上传示波器导出CSV文件；\n"
        "② 拍照模式：直接拍摄示波器屏幕，程序自动裁剪、透视矫正、抗反光增强，并识别档位。"
    )

    with st.sidebar:
        st.header("**输入参数区**")
        ref_sel = st.radio(
            "**待测类型选择（用于对照检测）**",
            ["m=0.6正常调幅", "m=1.0临界调幅", "m=1.4过调幅"]
        )
        input_mode = st.radio(
            "数据来源",
            ["CSV文件导出", "拍照/图片识别波形"]
        )
        fc_e = st.number_input("载波频率(Hz)", value=465000)
        fm_e = st.number_input("调制信号频率(Hz)", value=2000)
        st.markdown("---")
        st.subheader("包络平滑滤波")
        use_filter = st.checkbox("启用包络平滑降噪", value=True)
        win_size = st.slider("平滑窗口大小", min_value=3, max_value=60, value=21, step=2)

    # ---------- CSV分支 ----------
    if input_mode == "CSV文件导出":
        uploaded_file = st.file_uploader("**上传示波器CSV文件，两列：时间，电压**", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if len(df) < 2:
                    st.error("❌ **CSV文件数据太少！至少需要2个采样点，请检查示波器导出文件。**")
                else:
                    t_meas = df.iloc[:, 0].values
                    sig_meas = df.iloc[:, 1].values

                    dt_arr = np.diff(t_meas)
                    if np.mean(dt_arr) <= 0:
                        st.warning("⚠️ **采样时间异常，时间序列不能重复或倒序！**")
                    elif np.std(dt_arr) / np.mean(dt_arr) > 0.15:
                        st.warning("⚠️ **检测到采样时间间隔不均匀，包络提取效果可能变差！优先使用示波器等间隔采样导出**")

                    analytic = hilbert(sig_meas)
                    envelope = np.abs(analytic)
                    # 平滑滤波
                    if use_filter:
                        envelope = uniform_filter1d(envelope, size=win_size)

                    env_max = np.max(envelope)
                    env_min = np.min(envelope)
                    ma_meas = (env_max - env_min) / (env_max + env_min)

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

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
                    ax1.plot(t_meas * 1000, sig_meas, label="Measured AM Waveform")
                    ax1.plot(t_meas * 1000, envelope, "r--", label="Envelope Detection (smoothed)")
                    ax1.set_title("Measured AM Waveform\nEnvelope Detection")
                    ax1.set_xlabel("Time(ms)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    ax2.plot(t_ref * 1000, am_ref, color="orange", label=f"Reference Waveform {ref_sel}")
                    ax2.set_title(f"Reference Waveform Carrier {fc_e/1000:.1f}kHz, Mod {fm_e}Hz, m={ref_m:.1f}")
                    ax2.set_xlabel("Time(ms)")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                    fig.subplots_adjust(top=0.92, hspace=0.45)
                    st.pyplot(fig)

                    st.subheader("**🤖 AI智能研判结果**")
                    st.write(f"**估算实测调制度：** $m_a$ = {ma_meas:.3f}")

                    # 自动判定调幅状态
                    if ma_meas < 0.95:
                        state_info = "✅ 欠调幅，无相位反转，适合包络检波"
                    elif 0.95 <= ma_meas <= 1.05:
                        state_info = "⚡ 临界调幅，接近不失真上限"
                    else:
                        state_info = "❌ 过调幅，载波零点相位反转，包络检波失真！"
                    st.info(state_info)

                    # 和选择参考模板对比
                    if abs(ma_meas - ref_m) < 0.15:
                        st.success(f"✅ **匹配。实测m={ma_meas:.3f}，与选择【{ref_sel}】基本一致。**")
                    else:
                        st.error(f"❌ **不匹配。你选择的是{ref_name}，实测m={ma_meas:.3f}，波形类型不一致。**")

            except Exception as err:
                st.error(f"**文件解析失败：{err}**")
                st.info("请确认CSV只有两列：第一列时间(s)，第二列电压(V)，不要带多余表头说明行，文件至少包含多组采样点！")

    # ---------- 拍照图片分支【希尔伯特 + 侧边可控平滑滤波】 ----------
    else:
        st.markdown("##### 方式1：摄像头直接拍摄示波器屏幕")
        img_camera = st.camera_input("拍摄示波器（建议包含波形区和底部档位文字）")
        st.markdown("##### 方式2：上传示波器截图图片")
        img_upload = st.file_uploader("上传波形图片", type=["jpg", "jpeg", "png"])
        img_buffer = img_camera if img_camera else img_upload

        if img_buffer is not None:
            img_pil = Image.open(img_buffer)
            st.image(img_pil, caption="原始拍摄图片", use_container_width=True)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Step 1：自动裁剪示波器屏幕
            cropped = find_oscilloscope_screen(img_cv)

            # Step 2：透视矫正
            corrected = perspective_correction(cropped)

            st.image(corrected, caption="自动裁剪并透视矫正后的示波器屏幕", use_container_width=True)

            # Step 3：OCR识别档位，支持中文和英文
            ocr_raw = pytesseract.image_to_string(
                corrected,
                lang="chi_sim+eng"
            )
            st.markdown(f"OCR识别原文：`{ocr_raw}`")

            time_per_div, volt_per_div = parse_scope_settings(ocr_raw)

            # 识别失败时允许手动输入
            if time_per_div is None or volt_per_div is None:
                st.warning("⚠️ OCR未能完整识别档位，请手动输入示波器网格参数！")
                time_per_div = st.number_input("横向：每格时间(s)", value=0.0002)
                volt_per_div = st.number_input("纵向：每格电压(V)", value=0.5)
            else:
                st.success(f"✅ OCR识别成功：每格时间={time_per_div:.6f} s，每格电压={volt_per_div} V")

            # Step 4：提取波形像素序列
            t_meas, sig_meas = extract_waveform_from_image(corrected, time_per_div, volt_per_div)

            # 希尔伯特提取包络 + 平滑
            analytic = hilbert(sig_meas)
            envelope = np.abs(analytic)
            if use_filter:
                envelope = uniform_filter1d(envelope, size=win_size)

            env_max = np.max(envelope)
            env_min = np.min(envelope)
            ma_meas = (env_max - env_min) / (env_max + env_min)

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

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
            ax1.plot(t_meas * 1000, sig_meas, label="Measured AM Waveform")
            ax1.plot(t_meas * 1000, envelope, "r--", label="Envelope Detection (smoothed)")
            ax1.set_title("Measured AM Waveform\nEnvelope Detection")
            ax1.set_xlabel("Time(ms)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(t_ref * 1000, am_ref, color="orange", label=f"Reference Waveform {ref_sel}")
            ax2.set_title(f"Reference Waveform Carrier {fc_e/1000:.1f}kHz, Mod {fm_e}Hz, m={ref_m:.1f}")
            ax2.set_xlabel("Time(ms)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.subplots_adjust(top=0.92, hspace=0.45)
            st.pyplot(fig)

            st.subheader("**🤖 AI智能研判结果**")
            st.write(f"**估算实测调制度：** $m_a$ = {ma_meas:.3f}")

            # 自动判定调幅状态
            if ma_meas < 0.95:
                state_info = "✅ 欠调幅，无相位反转，适合包络检波"
            elif 0.95 <= ma_meas <= 1.05:
                state_info = "⚡ 临界调幅，接近不失真上限"
            else:
                state_info = "❌ 过调幅，载波零点相位反转，包络检波失真！"
            st.info(state_info)

            # 和选择参考模板对比
            if abs(ma_meas - ref_m) < 0.15:
                st.success(f"✅ **匹配。实测m={ma_meas:.3f}，与选择【{ref_sel}】基本一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的是{ref_name}，实测m={ma_meas:.3f}，波形类型不一致。**")

