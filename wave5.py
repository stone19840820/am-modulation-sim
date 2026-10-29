import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import cv2
import pytesseract
from PIL import Image
import re

# ====================== 全局Matplotlib中文配置 ======================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

st.set_page_config(page_title="高频电子线路虚拟仿真与研判平台", layout="wide")

# 1级标题（平台总名称，后续所有实验都不变）
st.title("高频电子线路虚拟仿真与研判平台")
# 2级标题（切换实验时只修改这一行文字）
st.subheader("AM调幅实验")

menu = st.radio("模块选择", ["A模块：波形仿真", "E模块：AI智能研判"])

# ====================== A模块：波形仿真 ======================
if menu == "A模块：波形仿真":
    st.header("A模块｜AM调幅波形仿真")
    st.sidebar.header("参数调节区")
    fc = st.sidebar.number_input("载波频率 fc (Hz)", value=450000, min_value=100, max_value=20000000)
    fm = st.sidebar.number_input("调制信号频率 fm (Hz)", value=1000, min_value=1, max_value=20000)
    m = st.sidebar.slider("调制系数 m", 0.0, 2.0, 0.6, 0.01)

    # 时间轴，取3个调制周期
    Tm = 1 / fm
    t = np.linspace(0, 3.2 * Tm, 30000)
    carrier = np.cos(2 * np.pi * fc * t)
    mod_signal = np.cos(2 * np.pi * fm * t)
    am_signal = (1 + m * mod_signal) * carrier

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t * 1000, am_signal, color="#1f77b4")
    ax.set_title(f"AM调幅波形，载波={fc/1000:.1f}kHz，调制={fm}Hz，m={m:.2f}")
    ax.set_xlabel("Time(ms)")
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(top=0.92)
    st.pyplot(fig)

    # 导出CSV
    df_out = pd.DataFrame({"time(s)": t, "voltage(V)": am_signal})
    csv_data = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("📥 导出当前波形CSV（用于E模块研判）", data=csv_data, file_name="am_wave.csv", mime="text/csv")


# ====================== E模块：AI智能研判 ======================
elif menu == "E模块：AI智能研判":
    st.info("**探究任务：对比实测波形与理想波形差异，校验实测波形是否正确。**")
    st.markdown(
        "**操作说明：**\n"
        "① CSV模式（推荐，高精度AI研判）：上传仿真/示波器导出CSV文件；\n"
        "② 拍照/图片模式：提取波形包络做形态研判，受摩尔纹、像素失真影响，结果仅作课堂参考。"
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

    # ========== 封装统一研判函数，仅CSV分支调用 ==========
    def judge_result(ref_sel, ma_meas, env_min_norm):
        st.subheader("**🤖 AI智能研判结果**")
        st.write(f"**估算实测调制度：** $m_a$ = {ma_meas:.3f}")
        st.write(f"**归一化包络最小值：** {env_min_norm:.3f}")
        if ref_sel == "m=0.6正常调幅":
            if 0.30 <= ma_meas <= 0.90 and env_min_norm > 0.10:
                st.success("✅ **匹配。上传波形为正常调幅，与选择类型一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的正常调幅，当前波形不是正常调幅，请上传正常调幅波形。**")
        elif ref_sel == "m=1.0临界调幅":
            if 0.70 <= ma_meas <= 1.30 and env_min_norm < 0.22:
                st.success("✅ **匹配。上传波形为临界调幅，与选择类型一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的临界调幅，当前波形不是临界调幅波形。**")
        elif ref_sel == "m=1.4过调幅":
            if env_min_norm < 0.15 and ma_meas > 0.90:
                st.success("✅ **匹配。上传波形为过调幅，与选择类型一致。**")
            else:
                st.error(f"❌ **不匹配。你选择的过调幅，当前波形不是过调幅波形。**")

    # ---------- CSV分支：完整保留波形提取+自动研判 ----------
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
                    env_max = np.max(envelope)
                    env_min = np.min(envelope)
                    ma_meas = (env_max - env_min) / (env_max + env_min)
                    env_min_norm = env_min / env_max

                    # 理想波形
                    Tm_ref = 1 / fm_e
                    t_ref = np.linspace(0, 3.2 * Tm_ref, 20000)
                    m_ref_val = float(re.search(r"m=(\d+\.\d+)", ref_sel).group(1))
                    if ref_sel == "m=0.6正常调幅":
                        am_ref = (1 + 0.6 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                    elif ref_sel == "m=1.0临界调幅":
                        am_ref = (1 + 1.0 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
                    else:
                        am_ref = (1 + 1.4 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(t_ref * 1000, am_ref, color="orange", label=f"Reference Waveform {ref_sel}")
                    ax.set_title(f"Reference Waveform Carrier {fc_e/1000:.1f}kHz, Mod {fm_e}Hz, m={m_ref_val:.2f}")
                    ax.set_xlabel("Time(ms)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.subplots_adjust(top=0.92)
                    st.pyplot(fig)

                    judge_result(ref_sel, ma_meas, env_min_norm)

            except Exception as err:
                st.error(f"**文件解析失败：{err}**")
                st.info("请确认CSV只有两列：第一列时间(s)，第二列电压(V)，不要带多余表头说明行！")

    # ---------- 拍照图片分支：提取包络，宽松形态研判 ----------
    else:
        st.warning("📷 图片模式：基于波形包络形状定性研判，受屏幕摩尔纹影响，结果仅作参考！精准判定请切换CSV模式。")
        st.markdown("##### 上传波形图片")
        img_upload = st.file_uploader("上传波形图片", type=["jpg", "jpeg", "png"])

        if img_upload is not None:
            img_pil = Image.open(img_upload)
            st.image(img_pil, caption="原始拍摄图片", use_container_width=True)

            # 图片预处理函数（你原来的函数保留）
            def find_oscilloscope_screen(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5),0)
                edge = cv2.Canny(blur,50,150)
                cnts,_ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts)==0:
                    return img
                cnt = max(cnts, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(cnt)
                return img[y:y+h, x:x+w]
            def perspective_correction(img):
                return img
            def parse_scope_settings(ocr_text):
                return None, None
            def extract_waveform_from_image(img, t_div, v_div):
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
                h,w = binary.shape
                sig = []
                for col in range(w):
                    col_data = binary[:,col]
                    y_idx = np.where(col_data>0)[0]
                    if len(y_idx)>0:
                        y_mean = np.mean(y_idx)
                        sig.append( (h/2 - y_mean) / (h/2)*v_div )
                    else:
                        sig.append(0)
                sig = np.array(sig)
                t = np.linspace(0, w*t_div, w)
                return t, sig

            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cropped = find_oscilloscope_screen(img_cv)
            corrected = perspective_correction(cropped)

            ocr_raw = pytesseract.image_to_string(corrected, lang="chi_sim+eng")
            time_per_div, volt_per_div = parse_scope_settings(ocr_raw)
            if time_per_div is None or volt_per_div is None:
                st.warning("⚠️ OCR未能识别档位，请手动输入网格参数")
                time_per_div = st.number_input("横向：每格时间(s)", value=0.0002)
                volt_per_div = st.number_input("纵向：每格电压(V)", value=2.0)

            t_meas, sig_meas = extract_waveform_from_image(corrected, time_per_div, volt_per_div)

            # 包络提取
            analytic_img = hilbert(sig_meas)
            env_img = np.abs(analytic_img)
            env_img_max = np.max(env_img)
            env_img_min = np.min(env_img)
            ma_img_est = (env_img_max - env_img_min) / (env_img_max + env_img_min)
            env_min_norm_img = env_img_min / env_img_max

            st.subheader("🤖 图片包络形态研判结果（仅供参考）")
            st.write(f"包络估算调制度：$m_a$ ≈ {ma_img_est:.3f}")
            st.write(f"归一化包络最小值：{env_min_norm_img:.3f}")

            # 形态分类规则
            if env_min_norm_img > 0.12:
                shape_type = "正常调幅（包络谷底不接近0）"
            elif 0.03 < env_min_norm_img <= 0.12:
                shape_type = "临界调幅（包络谷底接近0）"
            else:
                shape_type = "过调幅（包络谷底凹陷、发生相位翻转）"
            st.write(f"从包络形态判断：波形属于【{shape_type}】")

            target_type = ref_sel
            if target_type == "m=0.6正常调幅":
                if env_min_norm_img > 0.12:
                    st.success("✅ 形态匹配：包络特征符合正常调幅。")
                else:
                    st.error("❌ 形态不匹配：包络形态不符合正常调幅。")
            elif target_type == "m=1.0临界调幅":
                if 0.03 < env_min_norm_img <= 0.12:
                    st.success("✅ 形态匹配：包络特征符合临界调幅。")
                else:
                    st.error("❌ 形态不匹配：包络形态不符合临界调幅。")
            elif target_type == "m=1.4过调幅":
                if env_min_norm_img <= 0.03:
                    st.success("✅ 形态匹配：包络特征符合过调幅。")
                else:
                    st.error("❌ 形态不匹配：包络形态不符合过调幅。")
            st.info("💡 提示：图片识别存在摩尔纹噪声干扰，该结论仅用于课堂目视辅助验证，实验考核请使用CSV文件！")

            # 绘制理想参考波形
            Tm_ref = 1 / fm_e
            t_ref = np.linspace(0, 3.2 * Tm_ref, 20000)
            m_ref_val = float(re.search(r"m=(\d+\.\d+)", ref_sel).group(1))
            if ref_sel == "m=0.6正常调幅":
                am_ref = (1 + 0.6 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
            elif ref_sel == "m=1.0临界调幅":
                am_ref = (1 + 1.0 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)
            else:
                am_ref = (1 + 1.4 * np.cos(2 * np.pi * fm_e * t_ref)) * np.cos(2 * np.pi * fc_e * t_ref)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t_ref * 1000, am_ref, color="orange", label=f"Reference Waveform {ref_sel}")
            ax.set_title(f"Reference Waveform Carrier {fc_e/1000:.1f}kHz, Mod {fm_e}Hz, m={m_ref_val:.2f}")
            ax.set_xlabel("Time(ms)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.subplots_adjust(top=0.92)
            st.pyplot(fig)

            # 提取波形+红色包络图
            fig2, ax2 = plt.subplots(figsize=(12,4))
            ax2.plot(t_meas*1000, sig_meas, color="#1f77b4", label="提取波形(蓝色)")
            ax2.plot(t_meas*1000, env_img, color="red", lw=2, label="提取包络")
            ax2.set_xlabel("Time(ms)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.subplots_adjust(top=0.92)
            st.pyplot(fig2)
