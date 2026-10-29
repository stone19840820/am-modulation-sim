import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import cv2

# ====================== 全局Matplotlib中文配置 ======================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

st.set_page_config(page_title="高频电子线路虚拟仿真与研判平台", layout="wide")

# 1级标题（平台总名称，后续所有实验都不变）
st.title("高频电子线路虚拟仿真与研判平台")
# 2级标题
st.subheader("AM调幅实验")

menu = st.radio("模块选择", ["A模块：波形仿真", "E模块：AI智能研判"])

# ========== A模块：波形仿真 ==========
if menu == "A模块：波形仿真":
    st.header("A模块｜AM调幅波形仿真")
    st.sidebar.header("参数调节区")
    fc = st.sidebar.number_input("载波频率 fc (Hz)", value=450000, min_value=100, max_value=20000000)
    fm = st.sidebar.number_input("调制信号频率 fm (Hz)", value=1000, min_value=1, max_value=20000)
    m = st.sidebar.slider("调制系数 m", 0.0, 2.0, 0.6, 0.01)

    t = np.linspace(0, 0.002, 4000)
    carrier = np.cos(2 * np.pi * fc * t)
    mod_signal = np.cos(2 * np.pi * fm * t)
    am_signal = (1 + m * mod_signal) * carrier

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,7), constrained_layout=True)
    ax1.plot(t, mod_signal)
    ax1.set_title("调制信号")
    ax1.grid(alpha=0.3)
    ax2.plot(t, carrier)
    ax2.set_title("载波信号")
    ax2.grid(alpha=0.3)
    ax3.plot(t, am_signal)
    ax3.set_title(f"AM已调信号，调制系数 m={m:.2f}")
    ax3.grid(alpha=0.3)
    st.pyplot(fig)

# ========== E模块：AI智能研判 ==========
if menu == "E模块：AI智能研判":
    st.header("E模块｜AI智能研判 · 实测波形对比")
    ref_sel = st.radio("请选择你预期的波形类型", ["m=0.6正常调幅", "m=1.0临界调幅", "m=1.4过调幅"])
    upload_type = st.radio("数据来源", ["上传CSV文件（定量精确研判）", "上传示波器截图（图片提取波形）"])

    if upload_type == "上传CSV文件（定量精确研判）":
        st.subheader("CSV波形定量研判")
        uploaded_csv = st.file_uploader("上传示波器导出csv", type="csv")
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            t_data = df.iloc[:,0].values
            sig_data = df.iloc[:,1].values

            analytic = hilbert(sig_data)
            env = np.abs(analytic)
            env_max = np.max(env)
            env_min = np.min(env)
            ma_estimate = (env_max - env_min)/(env_max + env_min)
            st.write(f"估算调制度 ma = {ma_estimate:.3f}")

            fig_csv, ax_csv = plt.subplots(figsize=(12,4), constrained_layout=True)
            ax_csv.plot(t_data, sig_data, label="实测波形", alpha=0.5)
            ax_csv.plot(t_data, env, "r", linewidth=2, label="包络")
            ax_csv.set_title("CSV波形与包络")
            ax_csv.grid(True, alpha=0.3)
            ax_csv.legend()
            st.pyplot(fig_csv)

            st.markdown("### 研判结论")
            if ref_sel == "m=0.6正常调幅":
                if 0.4 < ma_estimate < 0.85:
                    st.success("✅波形匹配，为正常调幅")
                else:
                    st.error("❌不匹配，不是正常调幅")
            elif ref_sel == "m=1.0临界调幅":
                if 0.9 < ma_estimate < 1.1:
                    st.success("✅波形匹配，为临界调幅")
                else:
                    st.error("❌不匹配，不是临界调幅")
            elif ref_sel == "m=1.4过调幅":
                if ma_estimate > 1.1:
                    st.success("✅波形匹配，为过调幅")
                else:
                    st.error("❌不匹配，不是过调幅")

    else:
        st.subheader("示波器截图 → 提取波形并研判")
        st.info("⚠️由图片提取生成波形数据，受屏幕网格、拍摄摩尔纹干扰，存在误差，仅教学演示。正式测量请上传示波器CSV文件。")
        uploaded_img = st.file_uploader("上传示波器波形截图", type=["jpg","png"])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # 蓝色波形提取
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([85,70,40])
            upper_blue = np.array([130,255,255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            kernel = np.ones((3,3),np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            H,W = mask.shape
            sig_img = np.full(W, np.nan)
            for col in range(W):
                points_y = np.where(mask[:,col]>0)[0]
                if len(points_y)>0:
                    sig_img[col] = np.median(points_y)
            valid_idx = ~np.isnan(sig_img)
            x_img = np.arange(W)
            sig_img = np.interp(x_img, x_img[valid_idx], sig_img[valid_idx])
            sig_img = (H/2 - sig_img)/(H/2)

            # 构造虚拟CSV
            t_img = np.linspace(0, 1, len(sig_img))
            df_virtual = pd.DataFrame({"time": t_img, "signal": sig_img})
            csv_download = df_virtual.to_csv(index=False).encode('utf-8')
            st.download_button("📥下载图片提取得到的CSV", csv_download, "img_extracted.csv", "text/csv")

            # 和CSV完全一样的研判逻辑
            t_data = df_virtual["time"].values
            sig_data = df_virtual["signal"].values

            analytic = hilbert(sig_data)
            env = np.abs(analytic)
            env_max = np.max(env)
            env_min = np.min(env)
            ma_estimate = (env_max - env_min)/(env_max + env_min)
            st.write(f"估算调制度 ma = {ma_estimate:.3f}")

            # 绘图：提取波形+包络+理想包络叠加
            fig_img, ax = plt.subplots(figsize=(12,4), constrained_layout=True)
            ax.plot(t_data, sig_data, color="#1f77b4", alpha=0.4, label="提取波形")
            ax.plot(t_data, env, color="#1f77b4", linewidth=2, label="提取波形包络")

            if ref_sel == "m=0.6正常调幅":
                ma_ref =0.6
            elif ref_sel == "m=1.0临界调幅":
                ma_ref =1.0
            else:
                ma_ref =1.4
            env_ideal = 1 + ma_ref * np.cos(2*np.pi * 1 * t_data)
            env_ideal = env_ideal / np.max(env_ideal) * np.max(env)
            ax.plot(t_data, env_ideal, color="#ff4444", linestyle="--", linewidth=2, label=f"理想包络 {ref_sel}")

            ax.set_title("包络对比图")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig_img)

            st.markdown("### 研判结论")
            if ref_sel == "m=0.6正常调幅":
                if 0.4 < ma_estimate < 0.85:
                    st.success("✅波形匹配，为正常调幅")
                else:
                    st.error("❌不匹配，不是正常调幅")
            elif ref_sel == "m=1.0临界调幅":
                if 0.9 < ma_estimate < 1.1:
                    st.success("✅波形匹配，为临界调幅")
                else:
                    st.error("❌不匹配，不是临界调幅")
            elif ref_sel == "m=1.4过调幅":
                if ma_estimate > 1.1:
                    st.success("✅波形匹配，为过调幅")
                else:
                    st.error("❌不匹配，不是过调幅")

