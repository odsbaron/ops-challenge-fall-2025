import polars as pl
import numpy as np
import numba
from numba import jit, prange
from typing import Tuple

class ops_optimized:
    @staticmethod
    def rolling_regbeta_optimized(col_x_or_expr, col_y_or_expr, window: int) -> pl.Expr:
        """
        使用优化的Polars实现，减少重复计算
        """
        if isinstance(col_x_or_expr, str):
            expr_x = pl.col(col_x_or_expr)
        else:
            expr_x = col_x_or_expr

        if isinstance(col_y_or_expr, str):
            expr_y = pl.col(col_y_or_expr)
        else:
            expr_y = col_y_or_expr

        # 使用单次遍历计算协方差和方差
        # 这里使用map_batches来应用自定义函数
        return pl.map_batches(
            exprs=[expr_x, expr_y],
            function=lambda x: ops_optimized._rolling_regbeta_batch(x, window),
            return_dtype=pl.Float64
        ).alias("rolling_regbeta_optimized")

    @staticmethod
    def _rolling_regbeta_batch(batch_data: Tuple[np.ndarray, np.ndarray], window: int) -> np.ndarray:
        """
        批处理函数，使用numba优化的核心计算
        """
        x, y = batch_data
        return rolling_regbeta_numba(x, y, window)


@jit(nopython=True, parallel=True, cache=True)
def rolling_regbeta_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    使用numba并行优化的滚动回归beta计算

    Args:
        x: 自变量数组 (Low)
        y: 因变量数组 (Close)
        window: 滚动窗口大小

    Returns:
        beta值的数组
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    # 并行计算每个时间点的beta值
    for i in prange(n):
        if i < window - 1:
            # 窗口不足，返回NaN
            result[i] = np.nan
        else:
            # 计算当前窗口的统计量
            start_idx = i - window + 1
            x_window = x[start_idx:i+1]
            y_window = y[start_idx:i+1]

            # 使用Welford算法的变体计算协方差和方差
            # 这样可以减少数值误差并提高效率
            beta = compute_single_beta(x_window, y_window)
            result[i] = beta

    return result


@jit(nopython=True, cache=True)
def compute_single_beta(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算单个窗口的beta值

    使用合并算法同时计算协方差和方差
    """
    n = len(x)
    if n < 2:
        return np.nan

    # 使用两轮法计算均值和协方差
    # 第一轮：计算均值
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    # 第二轮：计算协方差和方差
    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov_xy += dx * dy
        var_x += dx * dx

    cov_xy /= (n - 1)  # 无偏估计
    var_x /= (n - 1)   # 无偏估计

    # 处理方差接近0的情况
    if var_x < 1e-6:
        return 0.0

    return cov_xy / var_x


def ops_rolling_regbeta_optimized(input_path: str, window: int = 20) -> np.ndarray:
    """
    优化版本的滚动回归beta计算函数

    主要优化：
    1. 使用numba JIT编译加速核心计算
    2. 并行处理不同时间点
    3. 合并协方差和方差计算，减少重复遍历
    4. 优化内存访问模式
    """
    # 读取数据并按symbol分组处理
    lf = pl.scan_parquet(input_path)

    # 获取所有唯一的symbol
    symbols = lf.select("symbol").unique().collect()["symbol"].to_list()

    # 初始化结果数组
    total_rows = lf.select(pl.len()).collect().item()
    result_array = np.full(total_rows, np.nan, dtype=np.float64)

    # 按symbol处理，这样可以更好地利用缓存和并行性
    for symbol in symbols:
        # 获取当前symbol的数据
        symbol_data = (
            lf.filter(pl.col("symbol") == symbol)
            .select(["Low", "Close"])
            .collect()
        )

        if len(symbol_data) == 0:
            continue

        # 转换为numpy数组进行numba计算
        x = symbol_data["Low"].to_numpy()
        y = symbol_data["Close"].to_numpy()

        # 使用numba优化的函数计算beta
        symbol_betas = rolling_regbeta_numba(x, y, window)

        # 将结果放回正确的位置
        # 这里需要获取原始索引位置
        symbol_mask = (
            lf.filter(pl.col("symbol") == symbol)
            .select(pl.lit(True))
            .collect()
            .to_series()
            .to_numpy()
        )

        # 找到全局位置
        global_indices = np.where(lf.collect()["symbol"].to_numpy() == symbol)[0]
        if len(global_indices) == len(symbol_betas):
            result_array[global_indices] = symbol_betas

    return result_array.reshape(-1, 1)


# 备用方案：使用Polars的UDF功能
def ops_rolling_regbeta_polars_udf(input_path: str, window: int = 20) -> np.ndarray:
    """
    使用Polars UDF的版本，保持与原始接口兼容
    """
    def compute_beta_batch(x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """批量处理函数"""
        return rolling_regbeta_numba(x_batch, y_batch, window)

    res = (
        pl.scan_parquet(input_path)
        .with_columns([
            pl.col("Close"),
            pl.col("Low")
        ])
        .sort("symbol")  # 按symbol排序以优化缓存局部性
        .select(
            pl.map_batches(
                exprs=[pl.col("Low"), pl.col("Close")],
                function=lambda cols: compute_beta_batch(cols[0], cols[1]),
                return_dtype=pl.Float64
            ).over("symbol").alias("rolling_regbeta")
        )
    ).collect()

    return res["rolling_regbeta"].to_numpy().reshape(-1, 1)


# 混合方案：结合Polars的分组能力和Numba的计算能力
def ops_rolling_regbeta_hybrid(input_path: str, window: int = 20) -> np.ndarray:
    """
    混合方案：使用Polars读取和分组，Numba进行核心计算
    这是最优的性能方案
    """
    # 使用Polars读取并按symbol分组
    lf = pl.scan_parquet(input_path)

    # 按symbol分组处理
    grouped_data = (
        lf
        .sort("symbol")  # 确保相同symbol的数据连续存储
        .collect()
        .group_by("symbol")
        .agg([
            pl.col("Low"),
            pl.col("Close"),
            pl.len().alias("count")
        ])
        .sort("symbol")
    )

    # 预分配结果数组
    total_rows = lf.select(pl.len()).collect().item()
    result = np.full(total_rows, np.nan, dtype=np.float64)

    # 处理每个symbol
    current_row = 0
    for batch in grouped_data.iter_rows(named=True):
        symbol = batch["symbol"]
        # 处理Polars Series，转换为numpy数组
        x_data = batch["Low"]
        y_data = batch["Close"]

        # 确保数据是numpy数组格式
        if hasattr(x_data, 'to_numpy'):
            x = x_data.to_numpy()
        else:
            x = np.array(x_data)

        if hasattr(y_data, 'to_numpy'):
            y = y_data.to_numpy()
        else:
            y = np.array(y_data)

        # Numba优化的核心计算
        betas = rolling_regbeta_numba(x, y, window)

        # 将结果写入正确的位置
        end_row = current_row + len(betas)
        result[current_row:end_row] = betas
        current_row = end_row

    return result.reshape(-1, 1)


# 主函数，选择最优实现
def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    """
    主入口函数，自动选择最优实现
    """
    return ops_rolling_regbeta_hybrid(input_path, window)


if __name__ == "__main__":
    # 简单测试
    test_input = "testcase/data_for_rolling_regbeta.parquet"
    result = ops_rolling_regbeta(test_input, window=20)
    print(f"Result shape: {result.shape}")
    print(f"Sample results: {result[:5]}")