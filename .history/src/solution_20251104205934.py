import polars as pl
import numpy as np
import os 
os.environ["POLARS_MAX_THREADS"] = "16"  # 设置最大线程数为 8

def ops_rolling_regbeta(input_path: str, window: int = 20, threshold: float = 1e-6) -> np.ndarray:
    # 使用懒加载（scan_parquet）读取数据
    df = pl.scan_parquet(input_path)
    
    # 计算滚动方差和协方差
    cov_xy = pl.rolling_cov(df["Low"], df["Close"], window_size=window, ddof=1, min_samples=2)
    var_x = df["Low"].rolling_var(window_size=window, ddof=1, min_samples=2)
    
    # 过滤掉 var_x 小于 threshold 的数据
    # 只保留 var_x 大于阈值的行
    df_filtered = df.filter(var_x > threshold)
    
    # 计算 beta (在过滤后的数据上)
    result = df_filtered.select(
        (cov_xy / var_x).alias("rolling_regbeta")
    )
    
    # 收集结果并转为 NumPy 数组
    return result.collect().to_numpy()