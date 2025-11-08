import polars as pl
import numpy as np

# 增量更新的计算方法
def incremental_mean_var(x, n, mean, M2):
    mean_new = mean + (x - mean) / (n + 1)
    M2_new = M2 + (x - mean) * (x - mean_new)
    return mean_new, M2_new

def incremental_cov(x, y, n, mean_x, mean_y, cov_xy):
    mean_x_new = mean_x + (x - mean_x) / (n + 1)
    mean_y_new = mean_y + (y - mean_y) / (n + 1)
    cov_xy_new = cov_xy + (x - mean_x) * (y - mean_y)
    return mean_x_new, mean_y_new, cov_xy_new

def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    # 使用惰性加载
    res = (
        pl.scan_parquet(input_path)
        .select([pl.col("Close"), pl.col("Low")])  # 加载所需的列
    )

    # 先将数据收集到内存
    df = res.collect()

    # 增量更新统计量
    beta_values = []
    n = 0
    mean_x = 0
    M2_x = 0
    mean_y = 0
    M2_y = 0
    cov_xy = 0

    for row in df.iter_rows():
        x, y = row[0], row[1]

        # 更新均值和方差
        mean_x, M2_x = incremental_mean_var(x, n, mean_x, M2_x)
        mean_y, M2_y = incremental_mean_var(y, n, mean_y, M2_y)

        # 更新协方差
        mean_x, mean_y, cov_xy = incremental_cov(x, y, n, mean_x, mean_y, cov_xy)

        n += 1

        # 滚动窗口大小已满时，计算beta
        if n >= window:
            # 方差和协方差
            var_x = M2_x / n
            cov_xy = cov_xy / n

            # 计算滚动 beta
            beta = cov_xy / var_x
            beta_values.append(beta)
                
    return np.array(beta_values)


