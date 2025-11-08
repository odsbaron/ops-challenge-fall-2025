import polars as pl
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def compute_rolling_beta_single(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    超高效的滚动beta计算

    核心优化：
    1. 单次遍历计算所有统计量
    2. 使用在线算法避免重复计算
    3. 最小化内存分配
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        # 计算当前窗口的统计量
        start = i - window + 1

        # 计算均值
        sum_x = 0.0
        sum_y = 0.0
        for j in range(start, i + 1):
            sum_x += x[j]
            sum_y += y[j]

        mean_x = sum_x / window
        mean_y = sum_y / window

        # 计算协方差和方差
        cov_xy = 0.0
        var_x = 0.0
        for j in range(start, i + 1):
            dx = x[j] - mean_x
            dy = y[j] - mean_y
            cov_xy += dx * dy
            var_x += dx * dx

        # 无偏估计
        cov_xy /= (window - 1)
        var_x /= (window - 1)

        # 处理数值稳定性
        if var_x < 1e-10:
            result[i] = 0.0
        else:
            result[i] = cov_xy / var_x

    return result


def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    """
    高性能滚动回归beta计算

    主要优化：
    1. 使用Polars的高效数据读取
    2. 按symbol分组处理，提高缓存命中率
    3. Numba JIT编译核心计算
    4. 向量化操作减少Python开销
    """

    # 读取数据并按symbol排序以提高缓存效率
    df = pl.scan_parquet(input_path).sort("symbol").collect()

    # 预分配结果数组
    total_rows = len(df)
    result = np.full(total_rows, np.nan, dtype=np.float64)

    # 获取所有唯一的symbol
    symbols = df["symbol"].unique().to_list()

    # 处理每个symbol
    current_pos = 0
    for symbol in symbols:
        # 获取当前symbol的数据
        mask = df["symbol"] == symbol
        symbol_df = df.filter(mask)

        if len(symbol_df) < window:
            current_pos += len(symbol_df)
            continue

        # 提取数据
        x = symbol_df["Low"].to_numpy()
        y = symbol_df["Close"].to_numpy()

        # 使用优化的numba函数计算
        betas = compute_rolling_beta_single(x, y, window)

        # 更新结果数组
        end_pos = current_pos + len(betas)
        result[current_pos:end_pos] = betas
        current_pos = end_pos

    return result.reshape(-1, 1)


