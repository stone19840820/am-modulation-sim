import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 初始化session_state默认值
if "fc" not in st.session_state:
    st.session_state.fc = 100
if "fm" not in st.session_state:
    st.session_state.fm = 10
if "m" not in st.session_state:
    st.session_state.m = 0.6

st.title("AI可视化：AM调幅信号仿真教学平台")

st.markdown("#### 参数调节区")

# ===== 载波频率 fc =====
col1, col2 = st.columns([2,1])
with col1:
    fc_slider = st.slider("载波频率 fc (Hz)", min_value=10, max_value=500, value=st.session_state.fc, key="fc_slider")
with col2:
    fc_input = st.number_input("手动输入 fc", min_value=10.0, max_value=500.0, value=float(fc_slider), step=1.0, key="fc_input")
st.session_state.fc = fc_input

# ===== 调制信号频率 fm =====
col3, col4 = st.columns([2,1])
with col3:
    fm_slider = st.slider("调制信号频率 fm (Hz)", min_value=1, max_value=50, value=st.session_state.fm, key="fm_slider")
with col4:
    fm_input = st.number_input("手动输入 fm", min_value=1.0, max_value=50.0, value=float(fm_slider), step=0.5, key="fm_input")
st.session_state.fm = fm_input

# ===== 调制系数 m =====
col5, col6 = st.columns([2,1])
with col5:
    m_slider = st.slider("调制系数 m", min_value=0.1, max_value=1.8, value=st.session_state.m, step=0.05, key="m_slider")
with col6:
    m_input = st.number_input("手动输入 m", min_value=0.1, max_value=1.8, value=float(m_slider), step=0.01, key="m_input")
st.session_state.m = m_input

# 赋值简写
fc = st.session_state.fc
fm = st.session_state.fm
m = st.session_state.m

# 生成波形
fs = 5000
t = np.linspace(0, 1, fs)
mt = np.sin(2 * np.pi * fm * t)    # 调制信号
ct = np.sin(2 * np.pi * fc * t)    # 载波信号
am = (1 + m * mt) * ct             # AM信号

# 绘图
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
ax[0].plot(t, mt)
ax[0].set_title("调制信号")
ax[0].set_ylabel("幅度")
ax[1].plot(t, ct, c="#993380")
ax[1].set_title("载波信号")
ax[1].set_ylabel("幅度")
ax[2].plot(t, am, c="#f29c30")
ax[2].set_title(f"AM调幅信号 (m={m:.2f})")
ax[2].set_ylabel("幅度")
ax[2].set_xlabel("t (s)")
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
1. 拖动滑块 **或** 在右侧输入框手动录入参数，可实时观察参数对波形的影响
2. **m < 1**：正常调幅，包络无失真
3. **m > 1**：发生过调幅失真，是工程禁忌参数
""")

# 隐藏streamlit自带按钮（纸飞机、菜单、底部）
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
.stDeployButton {display: none !important;}
div[data-testid="stToolbar"] {visibility: hidden !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
