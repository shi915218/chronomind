import numpy as np
import pandas as pd
from tsfeatures.tsfeatures import _get_feats 
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, Annotated, TypedDict
from pydantic import BaseModel, Field
from .utils.experiment_dataset_parser import ExperimentDataset, ExperimentDatasetParser


# tsfresh 补充特征（包装为 tsfeatures 兼容签名）
def _make_tsfresh_feat(func_name: str, kwargs: dict) -> Callable:
    """将 tsfresh 单个特征计算器包装为 tsfeatures 兼容签名。"""
    def _wrapper(ts: Any, freq: int = 1) -> dict[str, float]:
        try:
            from tsfresh.feature_extraction import feature_calculators as fc
        except ImportError:
            raise ImportError("请先安装 tsfresh：pip install tsfresh")
        fn = getattr(fc, func_name)

        # tsfeatures 在内部会按单条序列调用特征函数，ts 通常是 ndarray
        x = np.asarray(ts, dtype=float)
        try:
            val = fn(x, **kwargs)
            if hasattr(val, "__iter__") and not isinstance(val, (float, int, np.floating, np.integer)):
                return {f"{func_name}__{k}": float(v) for k, v in val}
            return {func_name: float(val)}
        except Exception:
            return {func_name: float("nan")}

    _wrapper.__name__ = func_name
    return _wrapper

approximate_entropy_feat = _make_tsfresh_feat(
    "approximate_entropy", {"m": 2, "r": 0.3}
)
number_cwt_peaks_feat = _make_tsfresh_feat(
    "number_cwt_peaks", {"n": 5}
)
linear_trend_feat = _make_tsfresh_feat(
    "linear_trend", {"param": [{"attr": "slope"}, {"attr": "intercept"}, {"attr": "rvalue"}]}
)
augmented_dickey_fuller_feat = _make_tsfresh_feat(
    "augmented_dickey_fuller", {"attr": "teststat", "autolag": "AIC"}
)
change_quantiles_feat = _make_tsfresh_feat(
    "change_quantiles", {"ql": 0.1, "qh": 0.9, "isabs": True, "f_agg": "mean"}
)

 


from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    guerrero,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    intervals,
    lumpiness,
    nonlinearity,
    pacf_features,
    series_length,
    sparsity,
    stability,
    stl_features,
    unitroot_kpss,
    unitroot_pp,
)

from tsfeatures.tsfeatures import _get_feats
from .ensemble_forecaster import Forecaster, chronomindForecaster
from .models.prophet_forecaster import Prophet

# LangGraph 核心依赖
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from .models.statsforecast_models import (
    AutoTheta,   
    MSTL,        
    Naive,
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    HistoricAverage,
    SeasonalNaive,
    ZeroModel, 
)

# ─────────────────────────────────────────────
# 默认模型列表
# ─────────────────────────────────────────────
DEFAULT_MODELS: list[Forecaster] = [
    AutoTheta(),   
    MSTL(),                                                                            
    Naive(),       
    ADIDA(),
    AutoARIMA(),
    AutoCES(),
    AutoETS(),
    CrostonClassic(),

    HistoricAverage(),
    IMAPA(),
    SeasonalNaive(),
    ZeroModel(),
    # Prophet(),
    
]

# ─────────────────────────────────────────────
# 可用的时间序列特征函数映射
# ─────────────────────────────────────────────
TSFEATURES: dict[str, Callable] = {
    "acf_features": acf_features,
    "arch_stat": arch_stat,
    "crossing_points": crossing_points,
    "entropy": entropy,
    "flat_spots": flat_spots,
    "heterogeneity": heterogeneity,
    "holt_parameters": holt_parameters,
    "lumpiness": lumpiness,
    "nonlinearity": nonlinearity,
    "pacf_features": pacf_features,
    "stl_features": stl_features,
    "stability": stability,
    "hw_parameters": hw_parameters,
    "unitroot_kpss": unitroot_kpss,
    "unitroot_pp": unitroot_pp,
    "series_length": series_length,
    "hurst": hurst,
    "sparsity": sparsity,
    "intervals": intervals,
    "guerrero": guerrero,
    "approximate_entropy": approximate_entropy_feat,
    "number_cwt_peaks": number_cwt_peaks_feat,
    "linear_trend": linear_trend_feat,
    "augmented_dickey_fuller": augmented_dickey_fuller_feat,
    "change_quantiles": change_quantiles_feat,
}


# ─────────────────────────────────────────────
# 输出数据模型（
# ─────────────────────────────────────────────
class ForecastAgentOutput(BaseModel):
    """预测 Agent 的结构化输出。"""

    tsfeatures_analysis: str = Field(
        description=(
            "基于时间序列特征，说明数据特点及其对模型选择与预测的含义。"
        )
    )
    selected_model: str = Field(
        description="最终选定用于预测的模型名称"
    )
    model_details: str = Field(
        description=(
            "所选模型的技术要点：假设、优势与典型适用场景。"
        )
    )
    model_comparison: str = Field(
        description=(
            "各模型表现对比，说明在本序列上孰优孰劣及可能原因。"
        )
    )
    is_better_than_seasonal_naive: bool = Field(
        description="所选模型是否优于Seasonal Naive基准"
    )
    reason_for_selection: str = Field(
        description="选择该模型的理由说明"
    )
    forecast_analysis: str = Field(
        description=(
            "对预测结果的解读：趋势、模式及潜在风险或问题。"
        )
    )
    anomaly_analysis: str = Field(
        description=(
            "对检测到的异常的分析：模式、可能成因与处理建议。"
        )
    )
    user_query_response: str | None = Field(
        description=(
            "若用户提供了自然语言问题，在此给出直接回答；若无用户问题则为 null。"
        )
    )
    ensemble_models: list[str] | None = Field(
        description=(
            "若选定模型为 Ensemble，列出参与集成的具体模型名称及权重；否则为 null。"
        ),
        default=None,                                                                              
    )


# ─────────────────────────────────────────────
# 文本转换辅助函数
# ─────────────────────────────────────────────
def _transform_time_series_to_text(df: pd.DataFrame) -> str:
    df_agg = df.groupby("unique_id").agg(list)
    return (
        "以下为 JSON 格式的时间序列：键为序列标识 unique_id，值为含两个数组的对象："
        "第一个为日期列，第二个为观测值列。"
        f"{df_agg.to_json(orient='index')}"
    )


def _transform_features_to_text(features_df: pd.DataFrame) -> str:
    return (
        "以下为 JSON 格式的序列特征：键为序列标识，值为特征名到取值的映射。"
        f"{features_df.to_json(orient='index')}"
    )


def _transform_eval_to_text(eval_df: pd.DataFrame, models: list[str]) -> str:
    return ", ".join([f"{model}: {eval_df[model].iloc[0]}" for model in models])


def _transform_fcst_to_text(fcst_df: pd.DataFrame) -> str:
    df_agg = fcst_df.groupby("unique_id").agg(list)
    return (
        "以下为 JSON 格式的预测值：键为序列标识，值为含日期数组与预测值数组的对象。"
        f"{df_agg.to_json(orient='index')}"
    )


def _transform_anomalies_to_text(anomalies_df: pd.DataFrame) -> str:
    """Transform anomaly detection results to text for the agent."""
    anomaly_cols = [col for col in anomalies_df.columns if col.endswith("-anomaly")]
    if not anomaly_cols:
        return "暂无异常检测结果。"

    anomaly_summary = {}
    for unique_id in anomalies_df["unique_id"].unique():
        series_data = anomalies_df[anomalies_df["unique_id"] == unique_id]
        series_summary = {}
        for anomaly_col in anomaly_cols:
            if anomaly_col in series_data.columns:
                anomaly_count = series_data[anomaly_col].sum()
                total_points = len(series_data)
                anomaly_rate = (
                    (anomaly_count / total_points) * 100 if total_points > 0 else 0
                )
                anomalies = series_data[series_data[anomaly_col]]
                anomaly_dates = (
                    anomalies["ds"].dt.strftime("%Y-%m-%d").tolist()
                    if len(anomalies) > 0
                    else []
                )
                series_summary[anomaly_col] = {
                    "count": int(anomaly_count),
                    "rate_percent": round(anomaly_rate, 2),
                    "dates": anomaly_dates[:10],
                    "total_points": int(total_points),
                }
        anomaly_summary[unique_id] = series_summary

    return (
        "以下为 JSON 格式的异常检测结果：键为序列标识，值含各异常列的统计"
        "（数量、占比、检出日期等）。"
        f"{anomaly_summary}"
    )


# ─────────────────────────────────────────────
# LangGraph 状态定义
# ─────────────────────────────────────────────
class ForecastGraphState(TypedDict):
    """LangGraph 图的状态，贯穿整个工作流"""

    # 消息历史（支持 add_messages reducer 自动追加）
    messages: Annotated[list, add_messages]

    # 数据集上下文
    dataset: Any

    # 各阶段产出的数据
    features_df: Any      # tsfeatures 结果
    eval_df: Any         # 交叉验证评估结果
    eval_forecasters: list[str]
    fcst_df: Any        # 预测结果
    anomalies_df: Any      # 异常检测结果

    # 最终结构化输出
    output: Any
    # 重试计数（用于 validate_best_model 逻辑）
    retry_count: int
    
    # 预处理操作摘要
    preprocess_report: str | None  
# ─────────────────────────────────────────────
# 主类：chronomind
# ─────────────────────────────────────────────
class chronomind:
    """
    chronomind: An AI agent for comprehensive time series analysis.
    使用 LangGraph StateGraph 重构，保持与原版相同的对外接口。

    工作流节点：
    1. tsfeatures_node   - 计算时间序列特征
    2. cv_node           - 交叉验证模型
    3. forecast_node     - 生成预测
    4. anomaly_node      - 异常检测
    5. llm_analysis_node - LLM 综合分析并生成结构化输出
    6. validate_node     - 验证输出（是否优于 SeasonalNaive）
    """

    def __init__(
        self,
        llm: str | BaseChatModel,
        forecasters: list[Forecaster] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: LLM 标识符字符串（如 "openai:gpt-4o"）或 LangChain BaseChatModel 实例。
            forecasters: 预测模型列表，默认使用 DEFAULT_MODELS。
            **kwargs: 传递给 LLM 的额外参数。
        """
        # ── 初始化预测模型 ──────────────────────────────
        if forecasters is None:
            forecasters = DEFAULT_MODELS

        combined_forecasters = []
        for f in forecasters:
            combined_forecasters.append(f)

        forecasters = combined_forecasters
        self.forecasters = {forecaster.alias: forecaster for forecaster in forecasters}

        # 确保 SeasonalNaive 始终存在（作为基准模型）
        if "SeasonalNaive" not in self.forecasters:
            self.forecasters["SeasonalNaive"] = SeasonalNaive()

        
        # ── 初始化 LLM ──────────────────────────────────
        if isinstance(llm, str):
            # 支持 "provider:model_name" 格式，如 "openai:gpt-4o"
            self.llm = self._build_llm_from_string(llm, **kwargs)
        else:
            self.llm = llm
        
        # ── 系统提示词 ─────────────────
        self.system_prompt = f"""
    你是时间序列预测专家。你会收到序列数据（数值与时间信息），任务是选出合适模型并完成全流程分析。
    可使用以下工具：

    1. tsfeatures_tool：计算时间序列特征，辅助模型选择。
    【严格限制】只能从以下特征名中选择，禁止使用任何其他名称：
    {", ".join(TSFEATURES.keys())}

    2. cross_validation_tool：对一个或多个模型做交叉验证。
       传入模型名列表，返回各模型交叉验证结果。
       可用模型：{", ".join(self.forecasters.keys())}

    3. forecast_tool：用指定模型生成预测。
       传入模型名，返回预测值。

    4. detect_anomalies_tool：用表现最好的模型做异常检测。
       传入模型名与置信水平，返回异常检测结果。

    你必须按顺序完成四步，并依次调用上述四个工具：

    1. 特征分析（必须调用 tsfeatures_tool）：
        - 首先调用 tsfeatures_tool，选择一组关键特征即可
        - 根据特征概括序列特点，指导后续的模型选择

    2. 模型评估（必须调用 cross_validation_tool）：
        - 必须调用 cross_validation_tool，且同时评估多个模型
        - 若用户在问题中点名了某些模型，应优先纳入这些模型
        - 若用户未指定，可先试较简单模型；若明显不如Seasonal Naive:，再换更复杂模型

    3. 决定模型与预测（必须调用 forecast_tool）：
        - 以 MASE 越低越好为原则选定最终模型
        - 必须调用 forecast_tool，用选定模型出预测

    4. 异常检测（必须调用 detect_anomalies_tool）：
        - 必须调用 detect_anomalies_tool，使用表现最好的模型
        - 置信水平常用 95%；要求更严可用 99%

    默认以 MASE（平均绝对缩放误差）评估；交叉验证至少使用一个窗口。
    季节性由日期列推断。

    最终输出须为符合 ForecastAgentOutput 模式的 JSON 对象（字段名保持如下）：
    {{
        "tsfeatures_analysis": "...",
        "selected_model": "...",
        "model_details": "...",
        "model_comparison": "...",
        "is_better_than_seasonal_naive": true/false,
        "reason_for_selection": "...",
        "forecast_analysis": "...",
        "anomaly_analysis": "...",
        "user_query_response": "..." 或 null
    }}
    """

        # ── 查询 Agent 系统提示词 ────────────────────────
        self.query_system_prompt = """
    你是时间序列预测助手。当前对话基于上一轮完整分析，你可使用下列结果（以文本/上下文形式提供）：
    - fcst_df：各序列的预测值
    - eval_df：各模型的评估结果（MASE 等）
    - features_df：已提取的时间序列特征
    - anomalies_df：异常检测结果

    你还可以通过 plot_tool 生成可视化图表。

    当用户提出追问时，请结合上述数据作答，说明推理过程并引用相关数据支撑结论。
    """

        # ── 构建 LangGraph 工作流 ────────────────────────
        self._graph = self._build_forecast_graph()
        self._query_graph = self._build_query_graph()

        # ── 实例状态 ─────────────────────────────────────
        self.dataset: ExperimentDataset
        self.fcst_df: pd.DataFrame
        self.eval_df: pd.DataFrame
        self.features_df: pd.DataFrame
        self.anomalies_df: pd.DataFrame
        self.eval_forecasters: list[str]
        self._last_forecast_params: dict = {}
        self.conversation_history: list[dict] = []
        self.preprocess_report: str | None = None                                                    

    def _build_llm_from_string(self, llm_str: str, **kwargs: Any) -> BaseChatModel:
        """
        将 "provider:model_name" 格式的字符串转换为 LangChain LLM 实例。
        """
        if ":" in llm_str:
            provider, model_name = llm_str.split(":", 1)
        else:
            provider, model_name = "openai", llm_str

        provider = provider.lower()
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            inst = ChatOpenAI(model=model_name, **kwargs)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            inst = ChatAnthropic(model=model_name, **kwargs)
        elif provider in ("google", "gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            inst = ChatGoogleGenerativeAI(model=model_name, **kwargs)
        else:
            from langchain_openai import ChatOpenAI
            inst = ChatOpenAI(model=model_name, **kwargs)
        return inst

    # ─────────────────────────────────────────────
    # 工具函数（供 LangGraph 节点调用）
    # ─────────────────────────────────────────────
    # 对 dataset.df 做缺失值处理和重采样，返回处理后的 df 和操作摘要
    def _run_preprocess(
        self,
        dataset,
        fill_method: str = "linear",   # 缺失值填充方式：linear / ffill / bfill / mean
        resample_freq: str | None = None,  # 重采样目标频率，None 表示不重采样
    ) -> tuple[Any, str]:
        """
        预处理节点逻辑：
        1. 检测并填充缺失值
        2. 可选重采样到指定频率
        返回 (处理后的 dataset, 操作摘要字符串)
        """
        import numpy as np
        from copy import deepcopy

        df = dataset.df.copy()
        report_lines = []

        # ── 缺失值处理 ──────────────────────────────
        for uid, group in df.groupby("unique_id"):
            missing = group["y"].isna().sum()
            if missing > 0:
                report_lines.append(f"序列 {uid}：检测到 {missing} 个缺失值，使用 {fill_method} 填充")

        if fill_method == "linear":
            df["y"] = df.groupby("unique_id")["y"].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )
        elif fill_method == "ffill":
            df["y"] = df.groupby("unique_id")["y"].transform(lambda s: s.ffill().bfill())
        elif fill_method == "bfill":
            df["y"] = df.groupby("unique_id")["y"].transform(lambda s: s.bfill().ffill())
        elif fill_method == "mean":
            df["y"] = df.groupby("unique_id")["y"].transform(lambda s: s.fillna(s.mean()))

        # 填充后仍有 NaN（序列全为空）则填 0
        remaining = df["y"].isna().sum()
        if remaining > 0:
            df["y"] = df["y"].fillna(0)
            report_lines.append(f"仍有 {remaining} 个无法插值的缺失值，已填充为 0")

        # ── 重采样 ──────────────────────────────────
        target_freq = resample_freq or dataset.freq
        if resample_freq is not None and resample_freq != dataset.freq:
            resampled_groups = []
            for uid, group in df.groupby("unique_id"):
                group = group.set_index("ds").sort_index()
                group_resampled = group["y"].resample(target_freq).mean()
                # 重采样后可能产生新缺失值，再次线性插值
                group_resampled = group_resampled.interpolate(method="linear", limit_direction="both")
                tmp = group_resampled.reset_index()
                tmp.columns = ["ds", "y"]
                tmp["unique_id"] = uid
                resampled_groups.append(tmp)
            df = pd.concat(resampled_groups)[["unique_id", "ds", "y"]].reset_index(drop=True)
            report_lines.append(
                f"重采样：{dataset.freq} → {target_freq}，处理后共 {len(df)} 行"
            )

        if not report_lines:
            report_lines.append("数据质量良好，无缺失值，未执行重采样")

        new_dataset = deepcopy(dataset)
        new_dataset.df = df
        new_dataset.freq = target_freq

        report = "【预处理报告】\n" + "\n".join(f"- {l}" for l in report_lines)
        print(f"\n[Preprocess] {report}")
        return new_dataset, report
    def _run_tsfeatures(self, dataset: ExperimentDataset, features: list[str]) -> pd.DataFrame:
        """计算时间序列特征（Step 1）"""
        print(f"\n[Step 1] 计算时间序列特征: {features}")
        callable_features = []
        for feature in features:
            if feature not in TSFEATURES:
                raise ValueError(
                    f"特征 {feature} 不可用。可用特征：{', '.join(TSFEATURES.keys())}"
                )
            callable_features.append(TSFEATURES[feature])

        features_dfs = []
        for uid in dataset.df["unique_id"].unique():
            features_df_uid = _get_feats(
                index=uid,
                ts=dataset.df,
                features=callable_features,
                freq=dataset.seasonality,
            )
            features_dfs.append(features_df_uid)

        features_df = pd.concat(features_dfs) if features_dfs else pd.DataFrame()
        features_df = features_df.rename_axis("unique_id")
        print(f"[Step 1] 特征计算完成，共 {len(features_df.columns)} 个特征")
        return features_df

    def _run_cross_validation(
        self, dataset: ExperimentDataset, models: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """执行交叉验证（Step 2）"""
        print(f"\n[Step 2] 交叉验证模型: {models}")
        # 校验序列长度是否足够做 CV，提前给出友好提示
        min_len = dataset.h * 2 + 1
        for uid, group in dataset.df.groupby("unique_id"):
            if len(group) < min_len:
                print(
                    f"[CV] 警告：序列 {uid} 长度 {len(group)} 小于最小要求 {min_len}，"
                    f"将自动缩小 h 重试"
                )
        callable_models = []
        for str_model in models:
            if str_model not in self.forecasters:
                raise ValueError(
                    f"模型 {str_model} 不可用。可用模型：{', '.join(self.forecasters.keys())}"
                )
            callable_models.append(self.forecasters[str_model])

        def _regularize_df_for_cv(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
            """按频率补齐每条序列时间轴，避免 CV 因缺失时段对齐失败。"""
            if freq is None:
                return df
            fixed_groups = []
            work_df = df.copy()
            work_df["ds"] = pd.to_datetime(work_df["ds"])
            for uid, group in work_df.groupby("unique_id"):
                g = group.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")
                if len(g) < 2:
                    fixed_groups.append(g[["unique_id", "ds", "y"]])
                    continue
                full_ds = pd.date_range(g["ds"].min(), g["ds"].max(), freq=freq)
                g = g.set_index("ds").reindex(full_ds)
                g["unique_id"] = uid
                g["y"] = pd.to_numeric(g["y"], errors="coerce")
                g["y"] = g["y"].interpolate(method="time").ffill().bfill()
                g = g.reset_index().rename(columns={"index": "ds"})
                fixed_groups.append(g[["unique_id", "ds", "y"]])
            return pd.concat(fixed_groups, ignore_index=True)

        forecaster = chronomindForecaster(models=callable_models)
        cv_h = dataset.h
        fcst_cv = None
        cv_df = dataset.df
        regularized_once = False
        for attempt in range(3):
            try:
                fcst_cv = forecaster.cross_validation(
                    df=cv_df,
                    h=cv_h,
                    freq=dataset.freq,
                )
                break
            except Exception as e:
                err = str(e)
                if (
                    (not regularized_once)
                    and "交叉验证得到的行数少于预期" in err
                ):
                    old_rows = len(cv_df)
                    cv_df = _regularize_df_for_cv(cv_df, dataset.freq)
                    regularized_once = True
                    print(
                        f"[CV] 检测到时段不连续，已自动补齐时间轴并插值："
                        f"{old_rows} -> {len(cv_df)} 行，保持 h={cv_h} 重试"
                    )
                    continue
                if attempt < 2:
                    cv_h = max(1, cv_h // 2)
                    print(f"[CV] 交叉验证失败，缩小 h 至 {cv_h} 重试（第 {attempt+2} 次）: {e}")
                else:
                    raise
        eval_df = dataset.evaluate_forecast_df(
            forecast_df=fcst_cv,
            models=[model.alias for model in callable_models],
        )
        eval_df = eval_df.groupby(["metric"], as_index=False).mean(numeric_only=True)

        # 打印各模型 MASE 分数
        for m in models:
            if m in eval_df.columns:
                print(f"[Step 2]   {m}: MASE = {eval_df[m].iloc[0]:.4f}")

        return eval_df, models

    def _run_forecast(self, dataset: ExperimentDataset, model: str) -> pd.DataFrame:
        """生成预测（Step 3）"""
        print(f"\n[Step 3] 使用模型 '{model}' 生成预测，horizon={dataset.h}")
        callable_model = self.forecasters[model]
        forecaster = chronomindForecaster(models=[callable_model])
        fcst_df = forecaster.forecast(
            df=dataset.df,
            h=dataset.h,
            freq=dataset.freq,
        )
        print(f"[Step 3] 预测完成，生成 {len(fcst_df)} 条预测记录")
        return fcst_df

    def _run_detect_anomalies(
        self, dataset: ExperimentDataset, model: str, level: int = 95
    ) -> pd.DataFrame:
        """异常检测（Step 4）"""
        print(f"\n[Step 4] 使用模型 '{model}' 进行异常检测（置信度 {level}%）")
        callable_model = self.forecasters[model]
        anomalies_df = callable_model.detect_anomalies(
            df=dataset.df,
            freq=dataset.freq,
            level=level,
        )
        anomaly_count = anomalies_df[f"{model}-anomaly"].sum()
        total_points = len(anomalies_df)
        anomaly_rate = (anomaly_count / total_points) * 100 if total_points > 0 else 0
        print(
            f"[Step 4] 异常检测完成: 发现 {anomaly_count}/{total_points} 个异常点 "
            f"（{anomaly_rate:.1f}%）"
        )
        return anomalies_df

    def _run_ensemble_forecast(
        self,
        dataset,
        eval_df,
        eval_forecasters,
        top_k=2,
        forecast_cache: dict[str, pd.DataFrame] | None = None,
    ):
        mase_row = eval_df.iloc[0]
        candidates = {
            m: float(mase_row[m])
            for m in eval_forecasters
            if m in eval_df.columns and pd.notna(mase_row[m]) and float(mase_row[m]) > 0
        }
        if not candidates:
            raise ValueError("eval_df 中无有效 MASE 分数，无法集成")

        sorted_models = sorted(candidates, key=lambda m: candidates[m])
        top_models = sorted_models[:top_k]
        print(f"\n[Ensemble] Top-{top_k} 模型: {top_models}")

        inv_mase = {m: 1.0 / candidates[m] for m in top_models}
        total = sum(inv_mase.values())
        weights = {m: v / total for m, v in inv_mase.items()}
        print(f"[Ensemble] 归一化权重: { {m: round(w, 4) for m, w in weights.items()} }")

        # ── 直接用 eval_df 里已有的 MASE 加权平均作为 Ensemble MASE 估算，不重跑 CV ──
        ensemble_mase = sum(candidates[m] * weights[m] for m in top_models)
        print(f"[Ensemble] Ensemble MASE（加权估算）= {ensemble_mase:.4f}")

        # ── 正式预测：按 unique_id+ds 对齐后做加权，避免不同模型行数不一致 ──
        base_df = None
        weighted_cols: list[str] = []
        for model_name, w in weights.items():
            if forecast_cache is not None and model_name in forecast_cache:
                fcst = forecast_cache[model_name]
                print(f"[Ensemble] 复用缓存预测: {model_name}")
            else:
                fcst = self._run_forecast(dataset, model=model_name)
                if forecast_cache is not None:
                    forecast_cache[model_name] = fcst
            cur = fcst[["unique_id", "ds", model_name]].copy()
            w_col = f"__w_{model_name}"
            cur[w_col] = cur[model_name].astype(float) * w
            cur = cur[["unique_id", "ds", w_col]]
            weighted_cols.append(w_col)
            if base_df is None:
                base_df = cur
            else:
                base_df = base_df.merge(cur, on=["unique_id", "ds"], how="inner")

        if base_df is None or base_df.empty:
            raise ValueError("集成失败：模型预测结果无法在 unique_id/ds 上对齐。")

        base_df["Ensemble"] = base_df[weighted_cols].sum(axis=1)
        base_df = base_df[["unique_id", "ds", "Ensemble"]]
        print(f"[Ensemble] 集成预测完成，共 {len(base_df)} 条记录")
        return base_df, ensemble_mase, top_models, weights
    # ─────────────────────────────────────────────
    # LangGraph 图构建：预测工作流
    # ─────────────────────────────────────────────
    def _build_forecast_graph(self) -> Any:
        """
        构建预测工作流的 LangGraph StateGraph。

        节点顺序：
        agent_node（LLM 决策 + 工具调用循环）→ validate_node → END

        LLM 通过 tool_calls 驱动四个工具依次执行：
        tsfeatures_tool → cross_validation_tool → forecast_tool → detect_anomalies_tool
        最后 LLM 生成结构化 ForecastAgentOutput。
        """
        from langgraph.graph import StateGraph

        # 使用 with_structured_output 让 LLM 直接输出 ForecastAgentOutput
        structured_llm = self.llm.with_structured_output(ForecastAgentOutput)
        # 对 dataset.df 做缺失值填充，并将报告注入消息供 LLM 参考
        def preprocess_node(state: ForecastGraphState) -> dict:
            dataset = state["dataset"]
            df = dataset.df.copy()
            before = len(df)
            df = df.sort_values("ds").drop_duplicates(subset=["unique_id", "ds"], keep="last")
            after = len(df)
            if before != after:
                from copy import deepcopy
                new_dataset = deepcopy(dataset)
                new_dataset.df = df
                dataset = new_dataset
                print(f"\n[Preprocess] 去除重复时间戳 {before - after} 条")
            # 检查是否有缺失值，没有则跳过
            has_missing = dataset.df["y"].isna().any()
            if not has_missing:
                print("\n[Preprocess] 数据无缺失值，跳过预处理")
                return {
                    "preprocess_report": "数据质量良好，无需预处理",
                }
            print("\n[Preprocess] 检测到缺失值，开始预处理...")
            new_dataset, report = self._run_preprocess(dataset, fill_method="linear")
            return {
                "dataset": new_dataset,
                "preprocess_report": report,
                "messages": [HumanMessage(content=report)],
            }
        def agent_node(state: ForecastGraphState) -> dict:
            """
            核心 Agent 节点：LLM 驱动工具调用循环。
            LLM 依次调用四个工具，收集结果后生成结构化输出。
            """
            print("\n[Agent] LLM 开始分析，准备调用工具...")
            # LangGraph 传入的是普通 dict；TypedDict 仅用于类型标注
            dataset: ExperimentDataset = state["dataset"]
            
            # 构建工具调用上下文
            ts_text = _transform_time_series_to_text(dataset.df)
            system_msg = SystemMessage(content=self.system_prompt + "\n\n" + ts_text)
            
            # 收集所有消息（系统提示 + 历史消息）
            messages = [system_msg] + list(state["messages"])
            
            # ── 工具调用循环 ──────────────────────────────
            # 我们手动实现 ReAct 循环：LLM 输出 tool_calls → 执行工具 → 将结果追加 → 再次调用 LLM
            tool_results_context = []
            max_iterations = 10
            iteration = 0

            # 绑定工具定义到 LLM（用于 function calling）
            llm_with_tools = self.llm.bind_tools([
                {
                    "name": "tsfeatures_tool",
                    "description": "计算时间序列统计特征，用于选模",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "features": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": f"特征名列表，须从以下选取：{', '.join(TSFEATURES.keys())}"
                            }
                        },
                        "required": ["features"]
                    }
                },
                {
                    "name": "cross_validation_tool",
                    "description": "对预测模型执行交叉验证",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": f"模型名列表，须从以下选取：{', '.join(self.forecasters.keys())}"
                            }
                        },
                        "required": ["models"]
                    }
                },
                {
                    "name": "forecast_tool",
                    "description": "使用指定模型生成预测",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "预测模型名称"
                            }
                        },
                        "required": ["model"]
                    }
                },
                {
                    "name": "detect_anomalies_tool",
                    "description": "对时间序列进行异常检测",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "用于异常检测的模型名"
                            },
                            "level": {
                                "type": "integer",
                                "description": "置信水平，默认 95",
                                "default": 95
                            }
                        },
                        "required": ["model"]
                    }
                }
            ])

            current_messages = list(messages)
            forecast_cache: dict[str, pd.DataFrame] = {}
            features_df = state["features_df"]
            eval_df = state["eval_df"]
            eval_forecasters = (
                list(state["eval_forecasters"]) if state["eval_forecasters"] else []
            )
            fcst_df = state["fcst_df"]
            anomalies_df = state["anomalies_df"]

            def _resolve_runtime_model(raw_model: str | None) -> str:
                """工具执行期的最小兜底：无效模型名回退到可执行基础模型。"""
                if raw_model in self.forecasters:
                    return raw_model  # type: ignore[return-value]

                # 先从已有预测列反推（最贴近当前上下文）
                if fcst_df is not None and not fcst_df.empty:
                    for m in self.forecasters.keys():
                        if m in fcst_df.columns:
                            print(f"[Agent] 无效模型名 '{raw_model}'，改用已预测模型 '{m}'")
                            return m

                # 再从评估结果选最优基础模型
                if eval_df is not None and not eval_df.empty:
                    try:
                        if "metric" in eval_df.columns:
                            mase_rows = eval_df[
                                eval_df["metric"].astype(str).str.contains("MASE", case=False)
                            ]
                            target_row = mase_rows.iloc[0] if len(mase_rows) > 0 else eval_df.iloc[0]
                        else:
                            target_row = eval_df.iloc[0]
                        base_cols = [
                            c for c in eval_df.columns
                            if c != "metric" and c in self.forecasters and pd.notna(target_row.get(c))
                        ]
                        if base_cols:
                            best_base = target_row[base_cols].astype(float).idxmin()
                            print(f"[Agent] 无效模型名 '{raw_model}'，改用评估最优模型 '{best_base}'")
                            return best_base
                    except Exception:
                        pass

                fallback = (
                    "SeasonalNaive"
                    if "SeasonalNaive" in self.forecasters
                    else next(iter(self.forecasters.keys()))
                )
                print(f"[Agent] 无效模型名 '{raw_model}'，回退到 '{fallback}'")
                return fallback

            while iteration < max_iterations:
                iteration += 1
                print(f"\n[Agent] 第 {iteration} 轮 LLM 调用...")
                
                response = llm_with_tools.invoke(current_messages)
                current_messages.append(response)
                tc = getattr(response, "tool_calls", None) or []
                content_preview = getattr(response, "content", "") or ""
                if len(content_preview) > 300:
                    content_preview = content_preview[:300] + "…"
                
                # 检查是否有工具调用
                if not hasattr(response, "tool_calls") or not response.tool_calls:
                    print("[Agent] LLM 未发出工具调用，准备检查是否需要补全工具结果")

                    # 如果关键阶段输出仍缺失，则由程序自动补齐，避免 LLM 过早停止导致后续表为空
                    if (
                        features_df is None
                        or eval_df is None
                        or fcst_df is None
                        or anomalies_df is None
                    ):
                        print(
                            "[Agent] 检测到关键输出缺失："
                            f"features_df={'缺失' if features_df is None else '已就绪'}, "
                            f"eval_df={'缺失' if eval_df is None else '已就绪'}, "
                            f"fcst_df={'缺失' if fcst_df is None else '已就绪'}, "
                            f"anomalies_df={'缺失' if anomalies_df is None else '已就绪'}。开始自动补全。"
                        )

                        # 1) features：使用全部可用特征（保持与 tsfeatures_tool 的语义一致）
                        if features_df is None:
                            try:
                                features_df = self._run_tsfeatures(
                                    dataset, features=list(TSFEATURES.keys())
                                )
                            except Exception as e:
                                print(f"[Agent] 自动补全 tsfeatures 失败: {e}")

                        # 2) eval：对所有已注册 forecasters 做交叉验证
                        if eval_df is None:
                            try:
                                eval_df, eval_forecasters = self._run_cross_validation(
                                    dataset, models=list(self.forecasters.keys())
                                )
                            except Exception as e:
                                print(f"[Agent] 自动补全 cross_validation 失败: {e}")

                        # 3) forecast：优先从交叉验证结果中选取 MASE 最小的模型；失败则回退 SeasonalNaive
                        model_for_anomalies: str | None = None
                        if fcst_df is None:
                            fallback_model = (
                                "SeasonalNaive"
                                if "SeasonalNaive" in self.forecasters
                                else next(iter(self.forecasters.keys()))
                            )
                            best_model = fallback_model
                            model_for_anomalies = best_model
                            try:
                                if eval_df is not None:
                                    # 交叉验证结果一般含 metric 行，此处优先选 MASE 行
                                    if "metric" in eval_df.columns and len(eval_df) > 0:
                                        mase_rows = eval_df[
                                            eval_df["metric"].astype(str).str.contains("MASE", case=False)
                                        ]
                                        target_row = (
                                            mase_rows.iloc[0]
                                            if len(mase_rows) > 0
                                            else eval_df.iloc[0]
                                        )
                                    else:
                                        target_row = eval_df.iloc[0]

                                    candidate_cols = [
                                        m
                                        for m in (eval_forecasters or list(self.forecasters.keys()))
                                        if m in eval_df.columns
                                    ]
                                    candidate_vals = (
                                        target_row[candidate_cols]
                                        .dropna()
                                        .astype(float)
                                    )
                                    if len(candidate_vals) > 0:
                                        best_model = candidate_vals.idxmin()
                                        model_for_anomalies = best_model
                            except Exception as e:
                                print(f"[Agent] 从 eval_df 选取最佳模型失败: {e}")

                        try:
                            if best_model in forecast_cache:
                                fcst_df = forecast_cache[best_model]
                                print(f"[Agent] 复用缓存预测: {best_model}")
                            else:
                                fcst_df = self._run_forecast(dataset, model=best_model)
                                forecast_cache[best_model] = fcst_df
                            if eval_df is not None and len(eval_forecasters) >= 2:
                                try:
                                    ensemble_df, ensemble_mase, ensemble_top_models, ensemble_weights = self._run_ensemble_forecast(
                                        dataset, eval_df, eval_forecasters, top_k=3, forecast_cache=forecast_cache
                                    )
                                    fcst_df = fcst_df.merge(
                                        ensemble_df[["unique_id", "ds", "Ensemble"]],
                                        on=["unique_id", "ds"], how="left",
                                    )
                                    eval_df["Ensemble"] = ensemble_mase
                                    if "Ensemble" not in eval_forecasters:
                                        eval_forecasters.append("Ensemble")
                                    self._ensemble_composition = {
                                        m: round(w, 4) for m, w in ensemble_weights.items()
                                    }
                                except Exception as e:
                                    print(f"[Ensemble] 自动补全集成失败，跳过: {e}")
                        except Exception as e:
                            print(f"[Agent] 自动补全 forecast 失败: {e}")

                        # 4) anomalies：对 forecast 使用同一模型
                        if anomalies_df is None and fcst_df is not None:
                            try:
                                if model_for_anomalies is None:
                                    # fcst_df 已存在但 anomalies_df 缺失：从 forecast 列名反推出模型 alias
                                    model_for_anomalies = next(
                                        (
                                            m
                                            for m in self.forecasters.keys()
                                            if m in fcst_df.columns
                                        ),
                                        None,
                                    )
                                if model_for_anomalies is None:
                                    model_for_anomalies = (
                                        "SeasonalNaive"
                                        if "SeasonalNaive" in self.forecasters
                                        else next(iter(self.forecasters.keys()))
                                    )
                                anomalies_df = self._run_detect_anomalies(
                                    dataset,
                                    model=model_for_anomalies,
                                    level=95,
                                )
                            except Exception as e:
                                print(f"[Agent] 自动补全 detect_anomalies 失败: {e}")

                    print("[Agent] 结束工具调用循环")
                    break

                # 执行每个工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    print(f"[Agent] 执行工具: {tool_name}，参数: {tool_args}")

                    try:
                        if tool_name == "tsfeatures_tool":
                            features_df = self._run_tsfeatures(
                                dataset, tool_args["features"]
                            )
                            result_text = _transform_features_to_text(features_df)

                        elif tool_name == "cross_validation_tool":
                            eval_df, eval_forecasters = self._run_cross_validation(
                                dataset, tool_args["models"]
                            )
                            result_text = _transform_eval_to_text(eval_df, eval_forecasters)

                        elif tool_name == "forecast_tool":
                            model_name = _resolve_runtime_model(tool_args.get("model"))
                            if model_name in forecast_cache:
                                fcst_df = forecast_cache[model_name]
                                print(f"[Agent] 复用缓存预测: {model_name}")
                            else:
                                fcst_df = self._run_forecast(dataset, model_name)
                                forecast_cache[model_name] = fcst_df
                            if eval_df is not None and len(eval_forecasters) >= 2:
                                try:
                                    ensemble_df, ensemble_mase, ensemble_top_models, ensemble_weights = self._run_ensemble_forecast(
                                        dataset, eval_df, eval_forecasters, top_k=3, forecast_cache=forecast_cache
                                    )
                                    fcst_df = fcst_df.merge(
                                        ensemble_df[["unique_id", "ds", "Ensemble"]],
                                        on=["unique_id", "ds"], how="left",
                                    )
                                    eval_df["Ensemble"] = ensemble_mase
                                    if "Ensemble" not in eval_forecasters:
                                        eval_forecasters.append("Ensemble")
                                    self._ensemble_composition = {
                                        m: round(w, 4) for m, w in ensemble_weights.items()
                                    }
                                except Exception as e:
                                    print(f"[Ensemble] 集成失败，跳过: {e}")
                            result_text = _transform_fcst_to_text(fcst_df)

                        elif tool_name == "detect_anomalies_tool":
                            level = tool_args.get("level", 95)
                            model_name = _resolve_runtime_model(tool_args.get("model"))
                            anomalies_df = self._run_detect_anomalies(
                                dataset, model_name, level
                            )
                            result_text = _transform_anomalies_to_text(anomalies_df)

                        else:
                            result_text = f"未知工具：{tool_name}"

                    except Exception as e:
                        result_text = f"工具执行错误：{str(e)}"
                        print(f"[Agent] 工具执行错误: {e}")

                    # 将工具结果追加到消息历史
                    current_messages.append(
                        ToolMessage(content=result_text, tool_call_id=tool_call_id)
                    )

            # ── 生成最终结构化输出 ────────────────────────
            print("\n[Agent] 所有工具调用完成，生成结构化分析报告...")
            
            # 构建包含所有工具结果的最终提示
            final_prompt = (
                "根据以上全部工具结果，给出综合分析，并以结构化 JSON 输出（符合 ForecastAgentOutput）。"
            )
            if features_df is not None:
                final_prompt += f"\n\n【特征】{_transform_features_to_text(features_df)}"
            if eval_df is not None:
                final_prompt += f"\n\n【交叉验证】{_transform_eval_to_text(eval_df, eval_forecasters)}"
            if fcst_df is not None:
                final_prompt += f"\n\n【预测】{_transform_fcst_to_text(fcst_df)}"
            if anomalies_df is not None:
                final_prompt += f"\n\n【异常】{_transform_anomalies_to_text(anomalies_df)}"
            if hasattr(self, "_ensemble_composition") and self._ensemble_composition:
                composition_str = "、".join(
                    f"{m}（权重 {w}）" for m, w in self._ensemble_composition.items()
                )
                final_prompt += f"\n\n【集成模型组成】Ensemble 由以下模型加权平均：{composition_str}"
                final_prompt += "\n若 selected_model 为 Ensemble，请将参与集成的模型名称列表填入 ensemble_models 字段。"
            current_messages.append(HumanMessage(content=final_prompt))

            # 使用结构化输出 LLM 生成最终报告
            output: ForecastAgentOutput = structured_llm.invoke(current_messages)

            print(f"[Agent] 分析完成，选定模型: {output.selected_model}")
            print(f"[Agent] 优于 SeasonalNaive: {output.is_better_than_seasonal_naive}")
            
            return {
                "messages": [AIMessage(content=f"分析完成。选定模型：{output.selected_model}")],
                "features_df": features_df,
                "eval_df": eval_df,
                "eval_forecasters": eval_forecasters,
                "fcst_df": fcst_df,
                "anomalies_df": anomalies_df,
                "output": output,
            }

        def validate_node(state: ForecastGraphState) -> dict:
            """
            验证节点：检查选定模型是否优于 SeasonalNaive。
            若不优于，增加重试计数并清空输出，触发重新分析。
            最多重试 3 次。
            """
            output: ForecastAgentOutput = state["output"]
            retry_count = state.get("retry_count", 0)
            
            if output is None:
                print("[Validate] 输出为空，跳过验证")
                return {}

            print(f"\n[Validate] 验证模型性能（第 {retry_count + 1} 次）...")
            print(f"[Validate] 选定模型: {output.selected_model}")
            print(f"[Validate] 优于 SeasonalNaive: {output.is_better_than_seasonal_naive}")

            if not output.is_better_than_seasonal_naive and retry_count < 3:
                print(
                    f"[Validate] 模型未优于 SeasonalNaive，触发重试 "
                    f"（{retry_count + 1}/3）..."
                )
                # 追加重试提示到消息，让 agent_node 重新尝试更好的模型
                retry_msg = HumanMessage(
                    content=(
                        "当前选定模型未优于季节朴素（Seasonal Naive）。"
                        "请换用不同或更复杂的模型重新尝试。"
                        f"上一轮对比摘要：{output.model_comparison}"
                    )
                )
                ret = {
                    "messages": [retry_msg],
                    "output": None,
                    "retry_count": retry_count + 1,
                }
                return ret

            if not output.is_better_than_seasonal_naive:
                print("[Validate] 已达最大重试次数，接受当前结果")
            else:
                print("[Validate] 验证通过！")

            return {}

        def should_retry(state: ForecastGraphState) -> str:
            out_none = state.get("output") is None
            rc = state.get("retry_count", 0)
            if out_none and rc > 0 and rc <= 3:
                return "retry"
            return "done"

        # ── 构建图 ────────────────────────────────────
        graph = StateGraph(ForecastGraphState)
        graph.add_node("preprocess", preprocess_node)  # 新增
        graph.add_node("agent", agent_node)
        graph.add_node("validate", validate_node)

        graph.set_entry_point("preprocess")           # 入口改为 preprocess
        graph.add_edge("preprocess", "agent")         # 新增边
        graph.add_edge("agent", "validate")
        graph.add_conditional_edges(
            "validate",
            should_retry,
            {"retry": "agent", "done": END},
        )
        print("[Graph] 预测工作流图构建完成")
        return graph.compile()

    # ─────────────────────────────────────────────
    # LangGraph 图构建：查询工作流
    # ─────────────────────────────────────────────

    def _build_query_graph(self) -> Any:
        """
        构建查询工作流的 LangGraph StateGraph。
        用于 analyze() 完成后的 follow-up 问答。
        """
        from langgraph.graph import StateGraph

        def query_agent_node(state: ForecastGraphState) -> dict:
            """
            查询 Agent 节点：基于已有分析结果回答用户问题。
            支持 plot_tool 调用。
            """
            print("\n[Query Agent] 处理用户查询...")
            dataset: ExperimentDataset = state["dataset"]
            
            # 构建上下文：将所有已有数据注入系统提示
            context_parts = [self.query_system_prompt]
            if state["features_df"] is not None:
                context_parts.append(
                    _transform_features_to_text(state["features_df"])
                )
            if state["eval_df"] is not None:
                context_parts.append(
                    _transform_eval_to_text(
                        state["eval_df"], state["eval_forecasters"]
                    )
                )
            if state["fcst_df"] is not None:
                context_parts.append(_transform_fcst_to_text(state["fcst_df"]))
            if state["anomalies_df"] is not None:
                context_parts.append(
                    _transform_anomalies_to_text(state["anomalies_df"])
                )
            if dataset is not None:
                context_parts.append(_transform_time_series_to_text(dataset.df))

            system_msg = SystemMessage(content="\n\n".join(context_parts))

            # 绑定 plot_tool
            llm_with_plot = self.llm.bind_tools([
                {
                    "name": "plot_tool",
                    "description": "为时间序列数据生成并展示图表",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plot_type": {
                                "type": "string",
                                "enum": ["forecast", "series", "anomalies", "both", "raw"],
                                "description": "图表类型：forecast=预测，series/raw=原始序列，anomalies=异常，both=预测+异常"
                            },
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "可选，要绘制的模型名列表"
                            }
                        },
                        "required": ["plot_type"]
                    }
                }
            ])

            messages = [system_msg] + list(state["messages"])
            response = llm_with_plot.invoke(messages)
            
            # 处理 plot_tool 调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                result_messages = [response]
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "plot_tool":
                        plot_result = self._execute_plot_tool(
                            dataset=dataset,
                            plot_type=tool_call["args"].get("plot_type", "forecast"),
                            models=tool_call["args"].get("models"),
                            fcst_df=state["fcst_df"],
                            anomalies_df=state["anomalies_df"],
                        )
                        result_messages.append(
                            ToolMessage(
                                content=plot_result,
                                tool_call_id=tool_call["id"]
                            )
                        )
                # 让 LLM 基于 plot 结果生成最终回复
                final_response = self.llm.invoke(
                    [system_msg] + list(state["messages"]) + result_messages
                )
                print(f"[Query Agent] 回答完成（含图表）")
                ans = getattr(final_response, "content", "") or ""
                return {"messages": [final_response]}

            print(f"[Query Agent] 回答完成")
            ans2 = getattr(response, "content", "") or ""
            return {"messages": [response]}

        graph = StateGraph(ForecastGraphState)
        graph.add_node("query_agent", query_agent_node)
        graph.set_entry_point("query_agent")
        graph.add_edge("query_agent", END)

        print("[Graph] 查询工作流图构建完成")
        return graph.compile()

    def _execute_plot_tool(
        self,
        dataset: ExperimentDataset,
        plot_type: str,
        models: list[str] | None,
        fcst_df: pd.DataFrame | None,
        anomalies_df: pd.DataFrame | None,
    ) -> str:
        """执行绘图工具"""
        try:
            import os
            import subprocess
            import sys

            import matplotlib
            import matplotlib.pyplot as plt

            from chronomind.models.utils.base_forecaster import Forecaster as FcstPlotter

            in_tmux = bool(os.environ.get("TMUX"))
            has_display = bool(os.environ.get("DISPLAY"))

            has_terminal_viewer = False
            terminal_viewers = ["imgcat", "catimg", "timg", "chafa"]
            for viewer in terminal_viewers:
                try:
                    if subprocess.run(["which", viewer], capture_output=True).returncode == 0:
                        has_terminal_viewer = True
                        break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

            if in_tmux or has_terminal_viewer:
                matplotlib.use("Agg")
                save_and_display = True
            elif not has_display:
                matplotlib.use("Agg")
                save_and_display = True
            else:
                try:
                    matplotlib.use("TkAgg")
                    save_and_display = False
                except ImportError:
                    try:
                        matplotlib.use("Qt5Agg")
                        save_and_display = False
                    except ImportError:
                        matplotlib.use("Agg")
                        save_and_display = True

            def try_display_plot(plot_file: str) -> str:
                for viewer, cmd in [
                    ("imgcat", [plot_file]),
                    ("catimg", [plot_file]),
                    ("timg", [plot_file]),
                    ("chafa", [plot_file]),
                ]:
                    try:
                        if subprocess.run(["which", viewer], capture_output=True).returncode == 0:
                            subprocess.run([viewer] + cmd, check=True)
                            return f"图表已保存为「{plot_file}」并已用 {viewer} 显示"
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                try:
                    if sys.platform == "darwin":
                        subprocess.run(["open", plot_file], check=True, capture_output=True)
                        return f"图表已保存为「{plot_file}」并已用系统查看器打开"
                    elif sys.platform.startswith("linux"):
                        subprocess.run(["xdg-open", plot_file], check=True, capture_output=True)
                        return f"图表已保存为「{plot_file}」并已用系统查看器打开"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(plot_file)}")
                    return f"图表已保存为「{plot_file}」并已在浏览器中打开"
                except Exception:
                    pass
                return f"图表已保存为「{plot_file}」。"

            print(f"[Plot] 生成 {plot_type} 图表...")

            if plot_type in ("series", "raw"):
                fig = FcstPlotter.plot(
                    df=dataset.df, forecasts_df=None, engine="matplotlib", max_ids=10
                )
                if save_and_display:
                    plot_file = "chronomind_series.png"
                    (fig or plt).savefig(plot_file, dpi=300, bbox_inches="tight")
                    plt.close(fig if fig else plt.gcf())
                    return try_display_plot(plot_file)
                else:
                    plt.show()
                    return "已生成并显示原始时间序列图。"

            elif plot_type == "anomalies" and anomalies_df is not None:
                fig = FcstPlotter.plot(
                    df=dataset.df,
                    forecasts_df=anomalies_df,
                    plot_anomalies=True,
                    engine="matplotlib",
                    max_ids=5,
                )
                if save_and_display:
                    plot_file = "chronomind_anomalies.png"
                    (fig or plt).savefig(plot_file, dpi=300, bbox_inches="tight")
                    plt.close(fig if fig else plt.gcf())
                    return try_display_plot(plot_file)
                else:
                    plt.show()
                    return "已生成并显示异常检测图。"

            elif plot_type == "forecast" and fcst_df is not None:
                if models is None:
                    models = [
                        col for col in fcst_df.columns
                        if col not in ["unique_id", "ds"] and "-" not in col
                    ]
                fig = FcstPlotter.plot(
                    df=dataset.df,
                    forecasts_df=fcst_df,
                    models=models,
                    engine="matplotlib",
                    max_ids=5,
                )
                if save_and_display:
                    plot_file = "chronomind_forecast.png"
                    (fig or plt).savefig(plot_file, dpi=300, bbox_inches="tight")
                    plt.close(fig if fig else plt.gcf())
                    return f"{try_display_plot(plot_file)} (models: {', '.join(models)})"
                else:
                    plt.show()
                    return f"已为模型 {', '.join(models)} 生成预测图。"

            elif plot_type == "both" and fcst_df is not None and anomalies_df is not None:
                if models is None:
                    models = [
                        col for col in fcst_df.columns
                        if col not in ["unique_id", "ds"] and "-" not in col
                    ]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                FcstPlotter.plot(
                    df=dataset.df, forecasts_df=fcst_df, models=models,
                    engine="matplotlib", max_ids=3, ax=ax1,
                )
                ax1.set_title("预测")
                FcstPlotter.plot(
                    df=dataset.df, forecasts_df=anomalies_df,
                    plot_anomalies=True, engine="matplotlib", max_ids=3, ax=ax2,
                )
                ax2.set_title("异常检测")
                plt.tight_layout()
                if save_and_display:
                    plot_file = "chronomind_combined.png"
                    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    return try_display_plot(plot_file)
                else:
                    plt.show()
                    return "已生成并显示预测与异常组合图。"

            else:
                available = []
                if fcst_df is not None:
                    available.append("forecasts")
                if anomalies_df is not None:
                    available.append("anomalies")
                return (
                    f"无法绘制「{plot_type}」。"
                    f"当前可用数据：{', '.join(available) if available else '无'}。"
                    "可尝试 plot_type 为 series、forecast、anomalies 或 both。"
                )

        except Exception as e:
            err = f"生成图表时出错：{str(e)}"
            return err

    # ─────────────────────────────────────────────
    # 内部辅助方法
    # ─────────────────────────────────────────────

    def _get_maybe_rerun_agent(self, query: str) -> tuple[Any, str]:
        """构建决策 Agent，判断是否需要重新运行分析"""
        context_parts = []
        if hasattr(self, "conversation_history") and self.conversation_history:
            recent_context = self.conversation_history[-3:]
            context_parts.append("最近对话上下文：")
            for msg in recent_context:
                context_parts.append(f"用户：{msg.get('user', '')}")
                context_parts.append(f"助手：{msg.get('assistant', '')}")

        analysis_context = []
        if hasattr(self, "eval_df") and self.eval_df is not None:
            models_used = [col for col in self.eval_df.columns if col != "metric"]
            analysis_context.append(f"当前分析已使用模型：{', '.join(models_used)}")
        if hasattr(self, "fcst_df") and self.fcst_df is not None:
            analysis_context.append(f"当前预测结果行数（近似 horizon）：{len(self.fcst_df)}")
        if hasattr(self, "anomalies_df") and self.anomalies_df is not None:
            analysis_context.append("已执行过异常检测")

        context_text = "\n".join(context_parts) if context_parts else "无先前对话"
        analysis_text = "\n".join(analysis_context) if analysis_context else "无已完成分析状态"

        decision_prompt = f"""
    你是时间序列分析助手。判断用户当前问题是否需要重新执行完整分析工作流。

    用户当前问题："{query}"

    对话上下文：
    {context_text}

    当前分析状态：
    {analysis_text}

    需要重新分析（返回 True）的典型情况：
    - 要换用不同模型或对比新模型
    - 要修改预测参数（如 horizon、频率等）
    - 明确要求重新做异常检测或全流程
    - 要载入新数据或换一种分析方式

    不需要重新分析（返回 False）的典型情况：
    - 仅询问已有结果、要解释或摘要
    - 仅要求展示或说明图表
    - 对已有结论的追问澄清

    只回答 True 或 False，不要其他文字。
    """
        return self.llm, decision_prompt

    def _maybe_rerun(self, query: str) -> bool:
        """同步判断是否需要重新运行分析"""
        if not query:
            return False
        llm, decision_prompt = self._get_maybe_rerun_agent(query)
        response = llm.invoke([
            SystemMessage(content="你是决策助手。只回答 True 或 False，不要其他内容。"),
            HumanMessage(content=decision_prompt),
        ])
        content = response.content.strip().lower()
        need = content.startswith("true")
        return need

    

    def is_queryable(self) -> bool:
        """检查是否可以进行 follow-up 查询。

        以是否已通过 :meth:`analyze` / :meth:`forecast` 载入 ``dataset`` 为准。
        中间产物（fcst_df、eval_df 等）可能为 None（例如 LLM 未调用工具），
        :meth:`query` 仍可运行并在有数据时使用对应上下文。
        """
        return getattr(self, "dataset", None) is not None

    def _maybe_raise_if_not_queryable(self):
        if not self.is_queryable():
            raise ValueError(
                "当前不可追问：请先调用 `analyze()` 或 `forecast()` 完成分析。"
            )

    def _build_conversation_context(self, current_query: str) -> str:
        """构建包含历史对话的上下文"""
        if not self.conversation_history:
            return current_query
        context_parts = ["历史对话："]
        for exchange in self.conversation_history[-5:]:
            context_parts.append(f"用户：{exchange['user']}")
            context_parts.append(f"助手：{exchange['assistant']}")
        context_parts.append(f"\n当前问题：{current_query}")
        return "\n".join(context_parts)
 

    def _sync_state_from_graph(self, final_state: dict):
        """将图的最终状态同步到实例属性（含 None，避免上一轮运行的陈旧值残留）"""
        self.features_df = final_state.get("features_df")
        self.eval_df = final_state.get("eval_df")
        self.eval_forecasters = final_state.get("eval_forecasters") or []
        self.fcst_df = final_state.get("fcst_df")
        self.anomalies_df = final_state.get("anomalies_df")
        self.preprocess_report = final_state.get("preprocess_report")  # 新增

        
    # ─────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        exog_df: pd.DataFrame | None = None,  

    ) -> "_AgentRunResult":
        """生成预测和异常分析。

        参数：
            df: 时间序列数据。可以是以下之一：
                - 包含 ["unique_id", "ds", "y"] 列的 pandas DataFrame。
                - 指向 CSV 或 Parquet 文件的路径或 URL。
            h: 预测范围（预测步数）。
            freq: Pandas 频率字符串（例如 "H" 表示小时，"D" 表示天，"MS" 表示月初）。
            seasonality: 主要季节性周期的长度。
            query: 可选的自然语言提示。

        返回：
            一个结果对象，其中 `.output` 的类型为 ForecastAgentOutput。
        """
        query_str = f"用户问题：{query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(model=self.llm)
        self.dataset = experiment_dataset_parser.parse(df, freq, h, seasonality, query)
        #new
        for forecaster in self.forecasters.values():
            if isinstance(forecaster, Prophet):
                forecaster._exog_df = exog_df  # exog_df 为 None 时自动清理上次残留
        
        initial_state = ForecastGraphState(
            messages=[HumanMessage(content=query_str or "请按系统说明完成完整分析流程。")],
            dataset=self.dataset,
            features_df=None,
            eval_df=None,
            eval_forecasters=[],
            fcst_df=None,
            anomalies_df=None,
            output=None,
            retry_count=0,
            preprocess_report=None,  

        )
        
        print("\n" + "="*60)
        print("[chronomind] 启动预测工作流（LangGraph）")
        print("="*60)

        final_state = self._graph.invoke(initial_state)
        self._sync_state_from_graph(final_state)
        self._last_forecast_params = {"h": h, "freq": freq}

        output = final_state["output"]
        result = _AgentRunResult(output=output)
        result.fcst_df = getattr(self, "fcst_df", None)
        result.eval_df = getattr(self, "eval_df", None)
        result.features_df = getattr(self, "features_df", None)
        result.anomalies_df = getattr(self, "anomalies_df", None)
        return result


    def query(self, query: str) -> "_AgentRunResult":
        """基于对话历史，就分析结果提出后续问题。
        参数：
            query: 用户的后续问题。
        返回：
            一个结果对象，其中 `.output` 的类型为 str。
        异常：
            ValueError: 如果尚未运行任何分析。
        """
        self._maybe_raise_if_not_queryable()
        # 判断是否需要重新运行分析
        if self._maybe_rerun(query):
            print(f"\n[Query] 检测到需要重新分析，重新运行工作流...")
            self.analyze(df=self.dataset.df, query=query)
        conversation_context = self._build_conversation_context(query)
        print(f"\n[Query] 处理查询: {query[:80]}...")
        initial_state = ForecastGraphState(
            messages=[HumanMessage(content=conversation_context)],
            dataset=self.dataset,
            features_df=getattr(self, "features_df", None),
            eval_df=getattr(self, "eval_df", None),
            eval_forecasters=getattr(self, "eval_forecasters", []),
            fcst_df=getattr(self, "fcst_df", None),
            anomalies_df=getattr(self, "anomalies_df", None),
            output=None,
            retry_count=0,
        )
        final_state = self._query_graph.invoke(initial_state)
        # 提取最后一条 AI 消息作为回答
        answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break
        self.conversation_history.append({"user": query, "assistant": answer})
        print(f"[Query] 回答完成")
        return _AgentRunResult(output=answer)

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []


# ─────────────────────────────────────────────
# 结果包装类
# ─────────────────────────────────────────────
class _AgentRunResult:
    def __init__(self, output: Any):
        self.output = output
        self.fcst_df: pd.DataFrame | None = None
        self.eval_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None
        self.anomalies_df: pd.DataFrame | None = None

    def __repr__(self) -> str:
        return f"_AgentRunResult(output={type(self.output).__name__})"
