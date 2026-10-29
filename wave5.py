import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
import cv2
from PIL import Image
import pytesseract
import re

# ====================== Matplotlib 全局配置 ======================
# 全部图表使用英文，避免云端标题丢失
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


def parse_scope_settings(ocr_text):
    """从OCR文字中解析时间/电压档位。"""
    ocr_text = ocr_text.replace("，", " ").replace(",", " ")
    pat_time = re.search(
        r'(\d+\.?\d*)\s*(us|μs|ms|s)',
        ocr_text.lower()
    )
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
    """从图片提取波形，中位数取点+尖峰过滤"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
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
            y_pixel[x] = np.median(idx)
        else:
            y_pixel[x] = np.nan
    mask = np.isnan(y_pixel)
    y_pixel[mask] = np.interp(x_pixel[mask], x_pixel[~mask], y_pixel[~mask])
    y_diff = np.diff(y_pixel, prepend=y_pixel[0])
    spike_th = np.std(y_diff)*3
    y_pixel[np.abs(y_diff)>spike_th] = np.nan
    mask2 = np.isnan(y_pixel)
    y_pixel[mask2] = np.interp(x_pixel[mask2], x_pixel[~mask2], y_pixel[~mask2])
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
        "② 拍照/图片模式：直接上传示波器屏幕照片。"
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

    # ---------- 拍照图片分支：屏幕定位 + 波形分离 + 包络研判 ----------
    else:
        st.warning("📷 图片模式：基于波形包络形状定性研判，受屏幕摩尔纹影响，结果仅作参考！精准判定请切换CSV模式。")
        st.markdown("##### 上传波形图片")
        img_upload = st.file_uploader("上传波形图片", type=["jpg", "jpeg", "png"])

        if img_upload is not None:
            img_pil = Image.open(img_upload).convert("RGB")
            st.image(img_pil, caption="原始拍摄图片", use_container_width=True)

            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # ---------- 1. 屏幕定位：优先找"亮绿色/高亮"区域，退化时用最大四边形 ----------
            def locate_screen(img):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((15, 15), np.uint8)
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    return img, None
                best, best_score = None, -1
                H, W = img.shape[:2]
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 0.05 * H * W:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    ar = w / max(h, 1)
                    if 1.0 < ar < 3.0:
                        score = area
                        if score > best_score:
                            best_score, best = score, (x, y, w, h)
                if best is None:
                    c = max(cnts, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    best = (x, y, w, h)
                x, y, w, h = best
                pad_x, pad_y = int(w * 0.05), int(h * 0.05)
                x, y = x + pad_x, y + pad_y
                w, h = w - 2 * pad_x, h - 2 * pad_y
                return img[y:y+h, x:x+w], (x, y, w, h)

            screen, box = locate_screen(img_cv)
            if box is None:
                st.error("❌ 未能定位示波器屏幕，请上传更清晰的正面照片。")
                st.stop()
            st.image(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), caption="定位到的屏幕区域", use_container_width=True)

            # ---------- 2. 波形提取：按"高亮/彩色"分离波形，剔除网格 ----------
            def extract_waveform(screen_bgr):
                hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
                H, W = screen_bgr.shape[:2]
                sat = hsv[:, :, 1]
                val = hsv[:, :, 2]
                color_mask = ((sat > 80) & (val > 120)).astype(np.uint8) * 255
                gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
                _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if color_mask.sum() > 0.005 * H * W * 255:
                    mask = color_mask
                else:
                    mask = bright_mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                sig = np.full(W, np.nan, dtype=np.float32)
                for col in range(W):
                    ys = np.where(mask[:, col] > 0)[0]
                    if len(ys) > 0:
                        sig[col] = np.median(ys)
                valid = ~np.isnan(sig)
                if valid.sum() < 10:
                    return None, None
                idx = np.arange(W)
                sig = np.interp(idx, idx[valid], sig[valid])
                v = (H / 2 - sig) / (H / 2)
                return idx, v

            col_idx, sig_norm = extract_waveform(screen)
            if sig_norm is None:
                st.error("❌ 未能从图片中提取到波形，请换一张对比度更高的照片。")
                st.stop()

            # ---------- 3. 去直流 + 轻平滑，再算包络 ----------
            sig_norm = sig_norm - np.mean(sig_norm)
            if len(sig_norm) > 51:
                sig_norm = savgol_filter(sig_norm, window_length=51, polyorder=3)

            # 先画提取到的波形，方便用户确认是否提取成功
            fig_w, ax_w = plt.subplots(figsize=(12, 3))
            ax_w.plot(col_idx, sig_norm, color="#d62728", linewidth=0.8)
            ax_w.set_title("从图片提取的波形（已去直流）")
            ax_w.set_xlabel("Pixel column")
            ax_w.set_ylabel("Normalized amplitude")
            ax_w.grid(True, alpha=0.3)
            fig_w.subplots_adjust(top=0.9)
            st.pyplot(fig_w)

            # ---------- 4. 包络提取（Hilbert）+ 包络平滑 ----------
            analytic_img = hilbert(sig_norm)
            env_img = np.abs(analytic_img)
            if len(env_img) > 31:
                env_img = savgol_filter(env_img, 31, 3)

            env_img_max = np.max(env_img)
            env_img_min = np.min(env_img)
            if env_img_max < 1e-6:
                st.error("❌ 包络幅值异常，无法研判。")
                st.stop()
            ma_img_est = (env_img_max - env_img_min) / (env_img_max + env_img_min)
            env_min_norm_img = env_img_min / env_img_max

            st.subheader("🤖 图片包络形态研判结果（仅供参考）")
            st.write(f"包络估算调制度：$m_a$ ≈ {ma_img_est:.3f}")
            st.write(f"归一化包络最小值：{env_min_norm_img:.3f}")

            # ---------- 5. 形态分类（阈值来自 (1-m)/(1+m) 理论值）----------
            if env_min_norm_img > 0.08:
                shape_type = "正常调幅（包络谷底远离0）"
            elif env_min_norm_img > 0.03:
                shape_type = "临界调幅（包络谷底接近0）"
            else:
                shape_type = "过调幅（包络谷底凹陷/相位翻转）"
            st.write(f"从包络形态判断：波形属于【{shape_type}】")

            target_type = ref_sel
            if target_type == "m=0.6正常调幅":
                if env_min_norm_img > 0.08:
                    st.success("✅ 形态匹配：包络特征符合正常调幅。")
                else:
                    st.error("❌ 形态不匹配：包络形态不符合正常调幅。")
            elif target_type == "m=1.0临界调幅":
                if 0.03 < env_min_norm_img <= 0.08:
                    st.success("✅ 形态匹配：包络特征符合临界调幅。")
                else:
                    st.error("❌ 形态不匹配：包络形态不符合临界调幅。")
            elif target_type == "m=1.4过调幅":
                if env_min_norm_img <= 0.03:
                    st.success("✅ 形态匹配：包络特征符合过调幅。")
                else:
                    st.error("❌ 形态不匹配：包络形态不符合过调幅。")

            # ========== 叠加：提取波形包络 VS 理想参考包络 ==========
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
            env_ref = np.abs(hilbert(am_ref))

            fig_compare, ax = plt.subplots(figsize=(12,4))
            ax.plot(col_idx, env_img, color="red", lw=2, label="提取波形包络")
            ax.plot(np.linspace(0,col_idx[-1], len(env_ref)), env_ref, color="orange", lw=2, label=f"理想包络 m={m_ref_val}")
            ax.set_title("提取包络 VS 理论理想包络")
            ax.set_xlabel("Pixel")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig_compare.subplots_adjust(top=0.9)
            st.pyplot(fig_compare)
