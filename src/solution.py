
import time
import pandas as pd
import numpy as np 
import polars as pl
from numba import jit, prange
from typing import Union

# 1. Numba 优化
@jit(nopython=True, fastmath=True, nogil=True)
def _rolling_rank_numba(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float32)
    
    # 预分配窗口数组避免重复切片
    window_buffer = np.empty(window, dtype=values.dtype)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx
        
        if window_size < 2:  # 窗口太小没有排名意义
            result[i] = 1.0 if window_size == 1 else np.nan
            continue
            
        # 批量复制到预分配缓冲区
        for j in range(window_size):
            window_buffer[j] = values[start_idx + j]
        
        current_val = values[i]
        count_less_equal = 0
        count_total = 0
        
        # 单次遍历计数
        for j in range(window_size):
            val = window_buffer[j]
            if not np.isnan(val):  # 处理NaN值
                count_total += 1
                if val <= current_val:
                    count_less_equal += 1
        
        if count_total > 0:
            result[i] = count_less_equal / count_total
        else:
            result[i] = np.nan
            
    return result
# 3. Polars 算子保持不变
class ops:
    @staticmethod
    def rolling_rank(col_or_expr: Union[str, pl.Expr], window: int) -> pl.Expr:
        if isinstance(col_or_expr, str):
            expr = pl.col(col_or_expr)
        else:
            expr = col_or_expr

        return expr.map_batches(
            # 4. 调用优化后的单线程版本
            lambda s: pl.Series(
                _rolling_rank_numba(s.to_numpy(), window=window)
            ),
            return_dtype=pl.Float32 
        )


# --- 3. 完整的执行函数（优化版） ---
def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    try:
        # 1. 使用 scan_parquet (惰性扫描)
        lazy_df = pl.scan_parquet(input_path) 

    except FileNotFoundError:
        print(f"错误：未找到文件 {input_path}。")
        return np.array([[]], dtype=np.float32)

    # 2. Polars 会分析整个查询计划，
    res = lazy_df.select(
        ops.rolling_rank("Close", window).over("symbol").alias("RollingRank")
    ).collect() # .collect() 触发执行
    
    # 3. 返回结果的 NumPy 数组
    result = res["RollingRank"].to_numpy().astype(np.float32)[:, None]
    
    return result