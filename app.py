import matplotlib.pyplot as plt

# 中文标题 / 坐标轴标签（Windows 常见字体优先）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import logging
import os
import sys
import traceback
import warnings



def _safe_exc_text(e: Exception) -> str:
    try:
        msg = str(e)
    except Exception:
        msg = repr(e)
    try:
        return msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        return repr(e)

class _DropMissingScriptRunContext(logging.Filter):
    """Streamlit 在「无会话上下文」线程上会打 WARNING；get_logger 还会把 level 设回 INFO，覆盖事先的 setLevel。"""
    def filter(self, record: logging.LogRecord) -> bool:
        return "missing ScriptRunContext" not in record.getMessage()

_SRC_CTX = "streamlit.runtime.scriptrunner_utils.script_run_context"
logging.getLogger(_SRC_CTX).addFilter(_DropMissingScriptRunContext())

for _name in (
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner_utils",
    _SRC_CTX,
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.state",
    "streamlit.runtime.state.session_state_proxy",
    "cmdstanpy",
    "utilsforecast",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

def _mute_streamlit_console_noise() -> None:
    for _name in (
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner_utils",
        _SRC_CTX,
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.state",
        "streamlit.runtime.state.session_state_proxy",
    ):
        logging.getLogger(_name).setLevel(logging.ERROR)

import pandas as pd
import streamlit as st

_mute_streamlit_console_noise()

try:
    from chronomind.agent import chronomind
    from chronomind.models.utils.base_forecaster import Forecaster as FcstPlotter
except ModuleNotFoundError:
    # 在包目录内直接 `streamlit run app.py` 时的路径回退。
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    from chronomind.agent import chronomind
    from chronomind.models.utils.base_forecaster import Forecaster as FcstPlotter

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="时序助手",
    page_icon="⏱️",
    layout="wide",
)

st.title("⏱️  时序助手")
st.caption("生成式AI预测引擎 · 大语言模型与时间序列基础模型的融合")

# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
if "tc" not in st.session_state:
    st.session_state.tc = None
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
def _chart_title(plot_kind: str, plot_ids: list | None) -> str:
    """与侧边「图表类型」文案一致，并标明当前绘制的序列，避免仅显示 unique_id=…"""
    base = {
        "series": "原始序列",
        "forecast": "预测结果",
        "anomalies": "异常检测",
    }[plot_kind]
    if plot_ids and len(plot_ids) == 1:
        return f"{base}（序列：{plot_ids[0]}）"
    return base

def _finalize_mpl_figure(fig) -> None:
    """utilsforecast 会在 figure 上放 legend，挤压标题；收紧布局并保留右侧图例空间。"""
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

def render_plot(plot_type: str) -> None:
    result = st.session_state.result
    tc = st.session_state.tc
    if result is None or tc is None:
        st.info("请先完成分析。")
        return

    dataset = tc.dataset
    fcst_df = result.fcst_df
    anomalies_df = result.anomalies_df

    # 向 utilsforecast 只传入单个 matplotlib 坐标轴时，多 id 绘图可能触发形状不匹配；
    # 单轴模式下仅绘制一条序列。
    plot_ids = None
    if "unique_id" in dataset.df.columns:
        unique_ids = dataset.df["unique_id"].dropna().unique().tolist()
        if unique_ids:
            plot_ids = [unique_ids[0]]

    fig, ax = plt.subplots(figsize=(12, 5))

    if plot_type == "series":
        FcstPlotter.plot(
            df=dataset.df,
            forecasts_df=None,
            ids=plot_ids,
            engine="matplotlib",
            max_ids=1,
            ax=ax,
        )
        ax.set_title(_chart_title("series", plot_ids), fontsize=13)

    elif plot_type == "forecast" and fcst_df is not None and not fcst_df.empty:
        models = [col for col in fcst_df.columns if col not in ["unique_id", "ds"] and "-" not in col]
        FcstPlotter.plot(
            df=dataset.df,
            forecasts_df=fcst_df,
            ids=plot_ids,
            models=models,
            engine="matplotlib",
            max_ids=1,
            ax=ax,
        )
        ax.set_title(_chart_title("forecast", plot_ids), fontsize=13)

    elif plot_type == "anomalies" and anomalies_df is not None and not anomalies_df.empty:
        FcstPlotter.plot(
            df=dataset.df,
            forecasts_df=anomalies_df,
            ids=plot_ids,
            plot_anomalies=True,
            engine="matplotlib",
            max_ids=1,
            ax=ax,
        )
        ax.set_title(_chart_title("anomalies", plot_ids), fontsize=13)

    elif (
        plot_type == "both"
        and fcst_df is not None
        and not fcst_df.empty
        and anomalies_df is not None
        and not anomalies_df.empty
    ):
        # utilsforecast 会按父 GridSpec 的行列数 reshape `ax`；
        # `plt.subplots(2,1)` 取单个子轴仍带 nrows=2，只传入一个 Axes 会在 reshape 时报错，
        # 故改用两个独立的 1×1 图。
        plt.close(fig)
        models = [col for col in fcst_df.columns if col not in ["unique_id", "ds"] and "-" not in col]

        fig_fcst, ax_fcst = plt.subplots(figsize=(12, 5))
        FcstPlotter.plot(
            df=dataset.df,
            forecasts_df=fcst_df,
            ids=plot_ids,
            models=models,
            engine="matplotlib",
            max_ids=1,
            ax=ax_fcst,
        )
        ax_fcst.set_title(_chart_title("forecast", plot_ids), fontsize=13)
        _finalize_mpl_figure(fig_fcst)
        st.pyplot(fig_fcst, width="stretch")
        plt.close(fig_fcst)

        fig_anom, ax_anom = plt.subplots(figsize=(12, 5))
        FcstPlotter.plot(
            df=dataset.df,
            forecasts_df=anomalies_df,
            ids=plot_ids,
            plot_anomalies=True,
            engine="matplotlib",
            max_ids=1,
            ax=ax_anom,
        )
        ax_anom.set_title(_chart_title("anomalies", plot_ids), fontsize=13)
        _finalize_mpl_figure(fig_anom)
        st.pyplot(fig_anom, width="stretch")
        plt.close(fig_anom)
        return

    else:
        st.warning("当前数据不支持该图表类型。")
        plt.close(fig)
        return

    _finalize_mpl_figure(fig)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ─────────────────────────────────────────────
# 侧边栏：输入配置
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("配置")

    # LLM 选择
    llm_option = st.selectbox(
        "LLM 模型",
        [
            "deepseek-v3.2",   
        ],
    )

    # API Key
    api_key_input = st.text_input(
        "API Key（可选，优先使用环境变量）",
        type="password",
        placeholder="sk-...",
    )
    base_url_input = st.text_input(
        "Base URL（可选，OpenAI 兼容网关）",
        placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    st.divider()

    # 数据来源
    st.subheader("数据")
    data_source = st.radio("数据来源", ["上传文件", "输入 URL"])

    uploaded_file = None
    data_url = None

    if data_source == "上传文件":
        uploaded_file = st.file_uploader("上传 CSV / Parquet", type=["csv", "parquet"])
        exog_file = st.file_uploader(
            "外生变量 CSV（可选）",                                                      
            type=["csv"],
            help="需包含 ds 列 + 特征列，如 holiday、promo，历史期和预测期都要有",                   
        )    
    else:
        data_url = st.text_input(
            "数据 URL",
            placeholder="https://",
        )

    st.divider()

    # 预测参数
    st.subheader("预测参数（可选）")
    freq_input = st.text_input("频率 freq", placeholder="如 D / MS / H")
    h_input = st.number_input("预测步数 h", min_value=1, value=None, step=1)
    seasonality_input = st.number_input(
        "季节性周期 seasonality", min_value=1, value=None, step=1
    )
    query_input = st.text_area(
        "自然语言 Query（可选）",
        placeholder="如：预测未来12个月的销量",
    )

    st.divider()

    run_btn = st.button(" 开始分析", type="primary", width="stretch")

# ─────────────────────────────────────────────
# 运行分析
# ─────────────────────────────────────────────
if run_btn:
    # 设置 API Key
    if api_key_input:
        provider = llm_option.split(":")[0] if ":" in llm_option else "openai"
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key_input
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key_input

    # 准备数据
    df_input = None
    exog_df = None
    if exog_file is not None:
        exog_df = pd.read_csv(exog_file)
        exog_df["ds"] = pd.to_datetime(exog_df["ds"])

    if data_source == "上传文件" and uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_parquet(uploaded_file)
    elif data_source == "输入 URL" and data_url:
        df_input = data_url
    else:
        st.error("请提供数据文件或 URL。")
        st.stop()

    freq = freq_input.strip() or None
    h = int(h_input) if h_input else None
    seasonality = int(seasonality_input) if seasonality_input else None
    query = query_input.strip() or None
    base_url = base_url_input.strip() or None

    # 重置状态
    st.session_state.result = None
    st.session_state.tc = None
    st.session_state.chat_history = []

    with st.spinner("时序助手正在分析，请稍候..."):
        try:
            llm_kwargs = {}
            if api_key_input:
                llm_kwargs["api_key"] = api_key_input
            if base_url:
                llm_kwargs["base_url"] = base_url
            tc = chronomind(llm=llm_option, **llm_kwargs)
            result = tc.analyze(
                df=df_input,
                freq=freq,
                h=h,
                seasonality=seasonality,
                query=query,
                exog_df=exog_df,  # 新增
            )
            st.session_state.tc = tc
            st.session_state.result = result
            st.success("分析完成！")
        except Exception as e:
            st.error(f"分析失败: {_safe_exc_text(e)}")
            st.stop()

# ─────────────────────────────────────────────
# 结果展示
# ─────────────────────────────────────────────
result = st.session_state.result

if result is not None:
    output = result.output

    tab1, tab2 = st.tabs([" 分析结果", "💬 对话"])

    # ── Tab 1: 分析结果 ──────────────────────
    with tab1:
        # 模型选定卡片
        col1, col2 = st.columns(2)
        with col1:
            st.metric("选定模型", output.selected_model)
            if output.selected_model == "Ensemble" and output.ensemble_models:
                st.caption("集成组成：" + " · ".join(output.ensemble_models))
        with col2:
            if output.is_better_than_seasonal_naive:
                st.success("✅ 优于 Seasonal Naive 基准")
            else:
                st.warning("️  未优于 Seasonal Naive 基准")

        st.divider()

        # 时间序列特征
        st.subheader("1. 时间序列特征")
        if result.features_df is not None:
            features_display = result.features_df.iloc[0].dropna().to_frame("值")
            features_display.index.name = "特征"
            features_display["值"] = features_display["值"].apply(lambda x: f"{x:.4f}")
            st.dataframe(features_display, width="stretch")
        st.write(output.tsfeatures_analysis)

        st.divider()

        # 模型评估
        st.subheader("2. 模型评估（MASE）")
        if result.eval_df is not None:
            eval_display = result.eval_df.copy()
            model_cols = [c for c in eval_display.columns if c != "metric"]
            scores = {m: float(eval_display[m].iloc[0]) for m in model_cols}
            scores_df = (
                pd.DataFrame.from_dict(scores, orient="index", columns=["MASE"])
                .sort_values("MASE")
            )
            scores_df.index.name = "模型"
            scores_df["MASE"] = scores_df["MASE"].apply(lambda x: f"{x:.4f}")
            st.dataframe(scores_df, width="stretch")
        st.write(output.model_comparison)

        st.divider()

        # 预测结果
        st.subheader("3. 预测结果")
        if result.fcst_df is not None and not result.fcst_df.empty:
            fcst_display = result.fcst_df.copy()
            if "ds" in fcst_display.columns:
                fcst_display["ds"] = pd.to_datetime(fcst_display["ds"]).dt.strftime(
                    "%Y-%m-%d"
                )
            st.caption(
                f"共 {len(fcst_display)} 行 · 列：{', '.join(fcst_display.columns.astype(str).tolist())}"
            )
            st.dataframe(fcst_display, width="stretch", height=320)
            st.caption("预测结果可视化（历史序列 + 预测曲线）")
            render_plot("forecast")
        else:
            st.warning("暂无预测结果表（fcst_df 为空或未生成）。")
        st.write(output.forecast_analysis)

        st.divider()

        # 异常检测
        st.subheader("4. 异常检测")
        if result.anomalies_df is not None and not result.anomalies_df.empty:
            anomaly_cols = [
                c for c in result.anomalies_df.columns if c.endswith("-anomaly")
            ]
            total = sum(int(result.anomalies_df[c].fillna(False).astype(bool).sum()) for c in anomaly_cols)
            total_pts = len(result.anomalies_df)
            rate = (total / total_pts * 100) if total_pts > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("样本点数", int(total_pts))
            col2.metric("异常点数量", int(total))
            col3.metric("异常率", f"{rate:.1f}%")

            ad_display = result.anomalies_df.copy()
            if "ds" in ad_display.columns:
                ad_display["ds"] = pd.to_datetime(ad_display["ds"]).dt.strftime(
                    "%Y-%m-%d"
                )

            st.caption("完整异常检测输出（含预测区间与各模型异常标记）")
            st.dataframe(ad_display, width="stretch", height=320)

            if anomaly_cols:
                mask = ad_display[anomaly_cols].any(axis=1)
                n_flagged = int(mask.sum())
                if n_flagged > 0:
                    with st.expander(
                        f"仅显示异常点（{n_flagged} 行）", expanded=False
                    ):
                        st.dataframe(
                            ad_display.loc[mask].reset_index(drop=True),
                            width="stretch",
                            height=min(280, 40 + n_flagged * 28),
                        )
                else:
                    st.success("当前区间内未标记异常点。")
            st.caption("异常检测可视化（观测值、置信区间与异常点）")
            render_plot("anomalies")
        else:
            st.warning("暂无异常检测结果表（anomalies_df 为空或未生成）。")
        st.write(output.anomaly_analysis)

        # Query 回答
        if output.user_query_response:
            st.divider()
            st.subheader("5. Query 回答")
            st.info(output.user_query_response)

    # ── Tab 2: 对话 ──────────────────────────
    with tab2:
        st.subheader("Follow-up 对话")

        # 展示历史
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 输入框
        user_q = st.chat_input("继续提问，如：哪个月预测值最高？")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    try:
                        tc = st.session_state.tc
                        answer = tc.query(user_q)
                        st.write(answer.output)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer.output}
                        )
                    except Exception as e:
                        st.error(f"查询失败: {_safe_exc_text(e)}")

        if st.button("清空对话历史"):
            st.session_state.chat_history = []
            if st.session_state.tc:
                st.session_state.tc.clear_conversation_history()
            st.rerun()

else:
    st.info(" 在左侧配置参数后点击「开始分析」。")