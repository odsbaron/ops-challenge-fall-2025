import polars as pl
import numpy as np
import gc

class Ops:
    @staticmethod
    def rolling_regbeta(col_x_or_expr, col_y_or_expr, window: int) -> pl.Expr:
        """
        Calculate the rolling regression beta between two columns with a given window size.

        :param col_x_or_expr: Column or expression representing the independent variable (X).
        :param col_y_or_expr: Column or expression representing the dependent variable (Y).
        :param window: The rolling window size.
        :return: Rolling regression beta as a Polars expression.
        """
        expr_x = pl.col(col_x_or_expr) if isinstance(col_x_or_expr, str) else col_x_or_expr
        expr_y = pl.col(col_y_or_expr) if isinstance(col_y_or_expr, str) else col_y_or_expr

        cov_xy = pl.rolling_cov(expr_x, expr_y, window_size=window, ddof=1, min_samples=2)
        var_x = expr_x.rolling_var(window_size=window, ddof=1, min_samples=2)

        return pl.when(var_x < 1e-6).then(0.0).otherwise(cov_xy / var_x).alias("rolling_regbeta")


def ops_rolling_regbeta(input_path: str, window: int = 20, batch_size: int = 1000000) -> np.ndarray:
    """
    Perform rolling regression beta calculation for the given dataset, processed in batches.

    :param input_path: Path to the input parquet file.
    :param window: The rolling window size (default is 20).
    :param batch_size: The batch size for processing the data in chunks (default is 1000000).
    :return: Numpy array of rolling regression betas.
    """
    try:
        df_lazy = pl.scan_parquet(input_path).select([pl.col("Low").cast(pl.Float32), pl.col("Close").cast(pl.Float32)])
        
        result_list = []

        # Process in batches
        for batch in df_lazy.chunked(batch_size):
            batch_result = (
                batch
                .with_columns([Ops.rolling_regbeta("Low", "Close", window).over("symbol")])
                .groupby("symbol")  # Group by symbol to parallelize processing
                .agg([Ops.rolling_regbeta("Low", "Close", window).alias("rolling_regbeta")])
                .collect()  # Perform actual computation and bring the result into memory
            )
            result_list.append(batch_result)

            # Explicitly call garbage collection after processing each batch
            gc.collect()

        # Concatenate results of all batches
        final_result = np.concatenate([r.to_numpy() for r in result_list], axis=0)

        return final_result

    except Exception as e:
        print(f"Error processing the file: {e}")
        return np.array([])

