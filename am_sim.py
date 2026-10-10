import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AM调幅可视化仿真", layout="wide")
st.title("AI可视化：AM调幅信号仿真教学平台")

# 参数设置
st.sidebar.header("参数调节区")
fc = st.sidebar.slider("载波频率 fc (Hz)", 20, 200, 100)
fm = st.sidebar.slider("调制信号频率 fm (Hz)", 1, 30, 10)
m = st.sidebar.slider("调制系数 m", 0.0, 2.0, 0.6, 0.01)

# 时间轴
t = np.linspace(0, 1, 2000)
# 载波、调制信号
carrier = np.cos(2 * np.pi * fc * t)
mod_signal = np.cos(2 * np.pi * fm * t)
# AM调幅公式
am_signal = (1 + m * mod_signal) * carrier

# 绘图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), dpi=100)

ax1.plot(t, mod_signal, color="#2E86AB")
ax1.set_title("基带调制信号")
ax1.set_ylabel("幅度")
ax1.grid(True)

ax2.plot(t, carrier, color="#A23B72")
ax2.set_title("载波信号")
ax2.set_ylabel("幅度")
ax2.grid(True)

ax3.plot(t, am_signal, color="#F18F01")
ax3.set_title(f"AM已调波形 (m = {m:.2f})")
ax3.set_ylabel("幅度")
ax3.set_xlabel("时间 t (s)")
ax3.grid(True)

plt.tight_layout()
st.pyplot(fig)

# 提示信息
if m < 1:
    st.success(f"✅ 当前调制系数 m={m:.2f}，为正常调幅，无失真")
elif m == 1:
    st.info(f"ℹ️ 当前调制系数 m={m:.2f}，为临界调幅")
else:
    st.error(f"❌ 当前调制系数 m={m:.2f}，产生过调幅失真！通信出现混叠干扰")

st.markdown("""
### 教学说明
1. 拖动滑块可实时观察参数对波形的影响  
2. **m＜1**：正常调幅，包络无失真  
3. **m＞1**：发生过调幅失真，是工程禁忌参数  
""")
# 隐藏 Streamlit 默认底部标识
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
