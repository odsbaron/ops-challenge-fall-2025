import polars as pl
import numpy as np
pl.Config.set_streaming_chunk_size(50_000_000)
class ops:
    @staticmethod
    def rolling_regbeta(col_x_or_expr, col_y_or_expr, window: int) -> pl.Expr:
        expr_x = col_x_or_expr if isinstance(col_x_or_expr, pl.Expr) else pl.col(col_x_or_expr)
        expr_y = col_y_or_expr if isinstance(col_y_or_expr, pl.Expr) else pl.col(col_y_or_expr)


        cov_xy = pl.rolling_cov(expr_x, expr_y, window_size=window, ddof=1, min_samples=2) 
        var_x = expr_x.rolling_var(window_size=window, ddof=1, min_samples=2)
        
        # 参数微调
        return pl.when(var_x < 1*1e-6).then(0.0).otherwise(cov_xy / var_x).alias("rolling_regbeta")


def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    res = (
        pl.scan_parquet(input_path)
        .with_columns([
            pl.col("Close").cast(pl.Float64),
            pl.col("Low").cast(pl.Float64),  
            pl.col("symbol").cast(pl.Categorical)  # <-- 添加在这里
      ])
        .select(
            ops.rolling_regbeta("Low", "Close", window).over("symbol")
        )
    ).collect(engine="streaming")
    return res.to_numpy()