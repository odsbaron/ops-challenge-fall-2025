import polars as pl
import numpy as np
import sys
import os
from joblib import Parallel, delayed

# 添加当前目录到路径，以便导入 optimized_ops_rolling
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimized_ops_rolling import RollingCompute



def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    
    df = pl.scan_parquet(input_path).collect()
    df = df.with_columns([
        pl.col("Close").cast(pl.Float64),
        pl.col("Low").cast(pl.Float64),
        pl.col("symbol").cast(pl.Categorical)
    ]).with_row_index("_original_index")

    partitions = df.partition_by("symbol", maintain_order=True, as_dict=False)
    
    # 1. 预分配最终结果数组 (与优化 1 相同)
    output_beta = np.full(len(df), np.nan, dtype=np.float64)

    # 2. 定义一个处理单个分区的辅助函数
    def process_partition(partition):
        low_data = partition["Low"].to_numpy()
        close_data = partition["Close"].to_numpy()
        indices = partition["_original_index"].to_numpy()

        cov_xy = RollingCompute.cov(
            low_data, close_data,
            window=window, ddof=1, min_periods=2,
            method='fast', return_series=False
        )
        var_x = RollingCompute.var(
            low_data, window=window, ddof=1, min_periods=2,
            method='fast', return_series=False
        )
        
        beta = np.where(var_x < 1*1e-6, 0.0, cov_xy / var_x)
        
        # 3. 直接在辅助函数中“就地”修改 output_beta 数组
        # 这是线程安全的，因为每个分区的 indices 互不重叠！
        output_beta[indices] = beta

    # 4. 使用 joblib 并行执行
    # n_jobs=-1 使用所有核心
    # backend='threading' 利用 Numba 的 GIL 释放特性
    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_partition)(partition) for partition in partitions
    )

    return output_beta.reshape(-1, 1)