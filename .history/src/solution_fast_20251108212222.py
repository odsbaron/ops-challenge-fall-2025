import polars as pl
import numpy as np

class ops:
    @staticmethod
    def rolling_regbeta(col_x_or_expr, col_y_or_expr, window: int) -> pl.Expr:
        # Convert strings to expressions if needed
        expr_x = pl.col(col_x_or_expr) if isinstance(col_x_or_expr, str) else col_x_or_expr
        expr_y = pl.col(col_y_or_expr) if isinstance(col_y_or_expr, str) else col_y_or_expr
        
        # Compute rolling covariance and variance
        cov_xy = pl.rolling_cov(expr_x, expr_y, window_size=window, ddof=1, min_samples=2)
        var_x = expr_x.rolling_var(window_size=window, ddof=1, min_samples=2)
        
        # Return rolling beta with small variance handling
        return pl.when(var_x < 1.02 * 1e-6).then(0.0).otherwise(cov_xy / var_x).alias("rolling_regbeta")


def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    # Efficiently read parquet and calculate rolling regression betas
    res = (
        pl.scan_parquet(input_path)
        .select([
            "symbol", 
            "Close", 
            "Low"
        ])
        .with_columns([
            pl.col("symbol").cast(pl.Categorical)  # Optimize symbol column with categorical type
        ])
        .select(
            ops.rolling_regbeta("Low", "Close", window).over("symbol")  # Apply rolling regression beta over symbols
        )
    ).collect(engine="streaming")  # Ensure lazy evaluation
    return res.to_numpy()  # Convert to NumPy array for downstream analysis
