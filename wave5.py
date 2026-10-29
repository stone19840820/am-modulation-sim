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
# 全部图表使用英文，注释掉中文字体，避免云端标题丢失
# plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
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
    """【升级增强版】从图片提取波形，中位数取点+尖峰过滤，提升识别率"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 更强降噪
    blur = cv2.GaussianBlur(gray, (81, 81), 0)
    diff = cv2.absdiff(gray, blur)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.convertScaleAbs(norm, alpha=2.5, beta=12)

    _, thresh = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY_INV)

    h_img, w_img = thresh.shape
    x_pixel = np.arange(w_img)
    y_pixel = np.zeros(w_img)

    for x in range(w_img):
        col = thresh[:, x]
        idx = np.where(col > 0)[0]
        if len(idx) > 0:
            y_pixel[x] = np.median(idx)  # 中位数代替均值，抑制噪点尖峰
        else:
            y_pixel[x] = np.nan

    # 插值补断线
    mask = np.isnan(y_pixel)
    y_pixel[mask] = np.interp(x_pixel[mask], x_pixel[~mask], y_pixel[~mask])

    # 突变尖峰剔除
    y_diff = np.diff(y_pixel, prepend=y_pixel[0])
    spike_th = np.std(y_diff)*3
    y_pixel[np.abs(y_diff)>spike_th] = np.nan
    mask2 = np.isnan(y_pixel)
    y_pixel[mask2] = np.interp(x_pixel[mask2], x_pixel[~mask2], y_pixel[~mask2])

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
        "② 拍照/图片模式：直接拍摄示波器屏幕或者上传已拍图片。"
    )

    with st.sidebar:
        st.header("**输入参数区**")
        ref_sel = st.radio(
            "**待测类型选择（用于对照检测）**",
            ["m=0.6正常调幅", "m=1.0临界调幅", "m=1.4过调幅"]
        )
        input_mode = st.radio(
            "数据来源",
            ["CSV文件", "拍照/图片"]
        )
        fc_e = st.number_input("载波频率(Hz)", value=465000)
        fm_e = st.number_input("调制信号频率(Hz)", value=2000)

    # ========== CSV研判函数（严格阈值） ==========
    def judge_csv(ref_sel, ma_meas, env_min_norm):
        st.subheader("**🤖 AI智能研判结果**")
        if ref_sel == "m=0.6正常调幅":
            if 0.40 <= ma_meas <= 0.80 and env_min_norm > 0.15:
                st.success("✅ **匹配。上传波形为正常调幅，与选择类型一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的正常调幅，当前波形不是正常调幅，请上传正常调幅波形。**")
        elif ref_sel == "m=1.0临界调幅":
            if 0.80 <= ma_meas <= 1.20 and env_min_norm < 0.18:
                st.success("✅ **匹配。上传波形为临界调幅，与选择类型一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的临界调幅，当前波形不是临界调幅，请上传临界调幅波形。**")
        elif ref_sel == "m=1.4过调幅":
            if env_min_norm < 0.12 and ma_meas > 0.95:
                st.success("✅ **匹配。上传波形为过调幅，与选择类型一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的过调幅，当前波形不是过调幅波形，请上传过调幅波形。**")

    # ========== 图片研判函数（宽松阈值，适配拍照噪声） ==========
    def judge_image(ref_sel, ma_meas, env_min_norm):
        st.subheader("**🤖 图片模式研判结果（仅课堂参考）**")
        if ref_sel == "m=0.6正常调幅":
            # 图片放宽，包络谷底>0.08就判定正常，调制度范围扩大
            if 0.30 <= ma_meas <= 0.90 and env_min_norm > 0.08:
                st.success("✅ **形态匹配。提取波形包络特征符合正常调幅。**")
            else:
                st.error(f"❌ **形态不匹配。包络特征不符合正常调幅，请上传正常调幅波形。**")
        elif ref_sel == "m=1.0临界调幅":
            if 0.70 <= ma_meas <= 1.30 and 0.02 < env_min_norm <0.22:
                st.success("✅ **形态匹配。提取波形包络特征符合临界调幅。**")
            else:
                st.error(f"❌ **形态不匹配。包络特征不符合临界调幅，请上传临界调幅波形。**")
        elif ref_sel == "m=1.4过调幅":
            if env_min_norm < 0.08 and ma_meas > 0.90:
                st.success("✅ **形态匹配。提取波形包络特征符合过调幅。**")
            else:
                st.error(f"❌ **形态不匹配。包络特征不符合过调幅，请上传过调幅波形。**")

    # ---------- CSV分支 ----------
    if input_mode == "CSV文件":
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
                    env_max = np.max(envelope)
                    env_min = np.min(envelope)
                    ma_meas = (env_max - env_min) / (env_max + env_min)
                    env_min_norm = env_min / env_max

                    # 理想波形：固定用侧边栏fc_e fm_e，时间长度取3个调制周期
                    Tm_ref = 1 / fm_e
                    t_ref = np.linspace(0, 3.2 * Tm_ref, 20000)

                    if ref_sel == "m=0.6正常调幅":
                        am_ref = (1 + 0.6 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                    elif ref_sel == "m=1.0临界调幅":
                        am_ref = (1 + 1.0 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                    else:
                        am_ref = (1 + 1.4 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(t_ref * 1000, am_ref, color="orange", label=f"Reference Waveform {ref_sel}")
                    ax.set_title(f"Reference Waveform Carrier {fc_e/1000:.1f}kHz, Mod {fm_e}Hz, m={ref_sel[2:4]}")
                    ax.set_xlabel("Time(ms)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.subplots_adjust(top=0.92)
                    st.pyplot(fig)

                    judge_csv(ref_sel, ma_meas, env_min_norm)

            except Exception as err:
                st.error(f"**文件解析失败：{err}**")
                st.info("请确认CSV只有两列：第一列时间(s)，第二列电压(V)，不要带多余表头说明行，文件至少包含多组采样点！")

    # ---------- 拍照图片分支（修改：HSV筛选蓝色波形，包络对比） ----------
    else:
        st.warning("📷 图片模式：仅提取图中蓝色波形进行包络分析，受屏幕摩尔纹、曝光影响，结论仅供课堂参考，精准判定请使用CSV文件！")
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

            # Step 3：OCR识别档位
            ocr_raw = pytesseract.image_to_string(
                corrected,
                lang="chi_sim+eng"
            )
            time_per_div, volt_per_div = parse_scope_settings(ocr_raw)

            if time_per_div is None or volt_per_div is None:
                st.warning("⚠️ OCR未能完整识别档位，请手动输入示波器网格参数！")
                time_per_div = st.number_input("横向：每格时间(s)", value=0.0002)
                volt_per_div = st.number_input("纵向：每格电压(V)", value=2.0)

            # ========== 【改动重点：HSV筛选蓝色波形，只提取蓝色线条】 ==========
            hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
            # 蓝色阈值，适配示波器常见蓝色波形
            lower_blue = np.array([90, 60, 40])
            upper_blue = np.array([130, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            # 形态学降噪，去掉细碎噪点
            kernel = np.ones((2, 2), np.uint8)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

            h_img, w_img = mask_blue.shape
            x_pixel = np.arange(w_img)
            y_pixel = np.zeros(w_img)

            for x in range(w_img):
                col = mask_blue[:, x]
                idx = np.where(col > 0)[0]
                if len(idx) > 0:
                    y_pixel[x] = np.median(idx)
                else:
                    y_pixel[x] = np.nan

            # 插值补断线
            mask_nan = np.isnan(y_pixel)
            y_pixel[mask_nan] = np.interp(x_pixel[mask_nan], x_pixel[~mask_nan], y_pixel[~mask_nan])

            # 剔除突变尖峰噪声
            y_diff = np.diff(y_pixel, prepend=y_pixel[0])
            spike_th = np.std(y_diff)*3
            y_pixel[np.abs(y_diff)>spike_th] = np.nan
            mask_nan2 = np.isnan(y_pixel)
            y_pixel[mask_nan2] = np.interp(x_pixel[mask_nan2], x_pixel[~mask_nan2], y_pixel[~mask_nan2])

            # 像素坐标转物理信号
            t_meas = x_pixel / w_img * (10 * time_per_div)
            sig_meas = ((h_img / 2 - y_pixel) / (h_img / 10)) * volt_per_div

            # Step 4：包络计算
            analytic = hilbert(sig_meas)
            envelope = np.abs(analytic)
            env_max = np.max(envelope)
            env_min = np.min(envelope)
            ma_meas = (env_max - env_min) / (env_max + env_min)
            env_min_norm = env_min / env_max

            # 理想参考波形
            Tm_ref = 1 / fm_e
            t_ref = np.linspace(0, 3.2 * Tm_ref, 20000)
            if ref_sel == "m=0.6正常调幅":
                am_ref = (1 + 0.6 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                m_ref_val = 0.6
            elif ref_sel == "m=1.0临界调幅":
                am_ref = (1 + 1.0 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                m_ref_val = 1.0
            else:
                am_ref = (1 + 1.4 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                m_ref_val = 1.4

            # 图1：理想参考波形
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t_ref * 1000, am_ref, color="orange", label=f"Reference Waveform m={m_ref_val:.2f}")
            ax.set_title(f"Reference Waveform Carrier {fc_e/1000:.1f}kHz, Mod {fm_e}Hz, m={m_ref_val:.2f}")
            ax.set_xlabel("Time(ms)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.subplots_adjust(top=0.92)
            st.pyplot(fig)

            # 图2：提取蓝色波形 + 包络，方便对比包络形状
            fig2, ax2 = plt.subplots(figsize=(12,4))
            ax2.plot(t_meas*1000, sig_meas, color="#1f77b4", label="Extracted Blue Waveform")
            ax2.plot(t_meas*1000, envelope, color="red", lw=2, label="Envelope")
            ax2.set_title("Extracted Waveform & Envelope")
            ax2.set_xlabel("Time(ms)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.subplots_adjust(top=0.92)
            st.pyplot(fig2)

            # 图片分支调用宽松的图片研判函数
            judge_image(ref_sel, ma_meas, env_min_norm)
