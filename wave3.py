import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ====================== 全局Matplotlib中文配置 ======================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 12   # 放大图表标题
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

st.set_page_config(page_title="高频电子线路虚拟仿真与研判平台", layout="wide")

# ===== 新增全局样式，放这里！ =====
st.markdown(
    """
    <style>
    /* 模块选择 radio 标题 */
    .stRadio > label {
        font-size:24px !important;
        font-weight:bold !important;
    }
    /* radio单选选项 A模块 E模块 */
    .stRadio div[role="radiogroup"] label {
        font-size:20px !important;
        font-weight:bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==================================

# 1级标题（平台总名称，后续所有实验都不变）
st.title("高频电子线路虚拟仿真与研判平台")
# 2级标题（切换实验时只修改这一行文字）
st.header("**AM调幅实验**")

# ============ 这里替换成上面新增代码 ============
menu = st.radio(
    "模块选择",
    ["A模块：波形仿真", "E模块：智能研判"],
)


# ========== A模块 ==========
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
    ax3.set_title(f"AM已调波形 (m = {m:.2f})")
    ax3.set_ylabel("幅度")
    ax3.set_xlabel("时间 t (s)")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig)

# ========== E模块：智能研判 ==========
elif menu == "E模块：智能研判":
    st.info("**探究任务：对比实测波形与理想波形差异，校验实测波形是否正确。**")
    st.markdown("**操作说明**：上传示波器导出CSV（时间,电压），先在左侧选择待测类型，再自动提取包络，与对应标准波形对比检测。")

    # ========== 左侧输入区 ==========
    with st.sidebar:
        st.header("**输入参数区**")
        ref_sel = st.radio("**待测类型选择（用于对照检测）**", ["m=0.6正常调幅", "m=1.0临界调幅", "m=1.4过调幅"])
        uploaded_file = st.file_uploader("**上传示波器CSV文件，两列：时间，电压**", type="csv")
        fc_e = st.number_input("载波频率(Hz)", value=465000)
        fm_e = st.number_input("调制信号频率(Hz)", value=2000)

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
                ax1.plot(t_meas * 1000, sig_meas, label="实测AM波形")
                ax1.plot(t_meas * 1000, envelope, "r--", label="模拟包络检波")
                ax1.set_title("实测AM波形\n模拟包络检波")
                ax1.set_xlabel("时间(ms)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(t_ref * 1000, am_ref, color="orange", label=f"标准参考波形 {ref_sel}")
                ax2.set_title(f"标准参考波形 载波{fc_e/1000:.1f}kHz，调制{fm_e}Hz，m={ref_m:.1f}")
                ax2.set_xlabel("时间(ms)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.subheader("**🤖 AI智能研判结果**")

                if ref_m == 0.6:
                    st.write(f"**估算实测调制度：** $m_a$ = {ma_meas:.3f}")
                    if ma_meas < 0.95:
                        st.success("✅ **匹配。上传波形为正常调幅，与选择类型一致。**")
                    else:
                        st.error(f"❌ **不匹配。你选择的是{ref_name}，当前上传波形不是正常调幅，请上传正常调幅波形。**")
                elif ref_m == 1.0:
                    st.write(f"**估算实测调制度：** $m_a$ = {ma_meas:.3f}")
                    if 0.95 <= ma_meas <= 1.05:
                        st.success("✅ **匹配。上传波形为临界调幅，与选择类型一致。**")
                    else:
                        st.error(f"❌ **不匹配。你选择的是{ref_name}，当前上传波形不是临界调幅，请上传临界调幅波形。**")
                elif ref_m == 1.4:
                    env_min_norm = env_min / env_max
                    if env_min_norm < 0.15:
                        st.success("✅ **匹配。上传波形为过调幅，与选择类型一致。**")
                    else:
                        st.error(f"❌ **不匹配。你选择的是{ref_name}，当前上传波形不是过调幅，请上传过调幅波形。**")

        except Exception as err:
            st.error(f"**文件解析失败：{err}**")
            st.info("请确认CSV只有两列：第一列时间(s)，第二列电压(V)，不要带多余表头说明行，文件至少包含多组采样点！")
