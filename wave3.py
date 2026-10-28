import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import pandas as pd

# ========== 解决中文乱码（Streamlit云端专用）==========
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 页面标题（放到图外面，不会被裁剪，申报截图更稳）==========
st.title("AM调制波形与包络检测对比平台")
st.markdown("#### 功能说明：上传示波器导出CSV(时间,电压)，提取包络，与标准AM波形比对")

# ========== 参数设置 ==========
col1, col2, col3 = st.columns(3)
fc = col1.number_input("载波频率 kHz", value=465.0, min_value=10.0)
fm = col2.number_input("调制信号频率 Hz", value=2000, min_value=10)
m = col3.slider("调幅系数 m", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

fs = 10 * fc * 1000   # 采样率
t_total = 1.0e-3      # 总时长 1ms
t = np.linspace(0, t_total, int(fs * t_total))

# 生成标准AM信号
Ac = 1
carrier = Ac * np.cos(2 * np.pi * fc * 1000 * t)
mod = m * Ac * np.cos(2 * np.pi * fm * t)
am_signal = (1 + mod) * carrier

# 希尔伯特变换提取包络
analytic = hilbert(am_signal)
env = np.abs(analytic)

# ========== 绘图 ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7))

# 上图：AM波形 + 包络
ax1.plot(t*1000, am_signal, label="AM调制波形")
ax1.plot(t*1000, env, "r--", label="提取包络")
ax1.set_xlabel("时间 (ms)")
ax1.set_ylabel("电压")
ax1.set_title(f"AM调制波形，载波{fc}kHz，调制信号{fm}Hz，m={m:.2f}")
ax1.legend()
ax1.grid(True)

# 下图：包络波形
ax2.plot(t*1000, env, color="orange", label=f"包络 m={m:.2f}")
ax2.set_xlabel("时间 (ms)")
ax2.set_ylabel("包络幅度")
ax2.legend()
ax2.grid(True)

# 留出顶部空间，防止标题被裁剪
fig.subplots_adjust(top=0.9, hspace=0.4)

st.pyplot(fig)

# ========== CSV上传模块 ==========
st.subheader("示波器CSV上传比对")
upload_file = st.file_uploader("上传示波器CSV文件，格式：时间,电压", type="csv")
if upload_file is not None:
    df = pd.read_csv(upload_file)
    t_osc = df.iloc[:,0]
    v_osc = df.iloc[:,1]
    analytic_osc = hilbert(v_osc)
    env_osc = np.abs(analytic_osc)

    fig2, ax = plt.subplots(figsize=(10,4))
    ax.plot(t_osc, v_osc, label="实测波形")
    ax.plot(t_osc, env_osc, "r--", label="实测包络")
    ax.set_title("示波器实测波形与包络提取")
    ax.set_xlabel("时间")
    ax.set_ylabel("电压")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig2)
