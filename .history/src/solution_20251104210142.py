def ops_rolling_regbeta(input_path: str, window: int = 20, threshold: float = 1e-6) -> np.ndarray:
    # 使用懒加载（scan_parquet）读取数据
    df = pl.scan_parquet(input_path)
    
    # 计算滚动方差和协方差
    cov_xy = pl.rolling_cov(df["Low"], df["Close"], window_size=window, ddof=1, min_samples=2)
    var_x = df["Low"].rolling_var(window_size=window, ddof=1, min_samples=2)
    
    # 在 lazy frame 上创建新的表达式：计算 var_x 并过滤
    filtered_df = df.select([
        pl.col("Low"),
        pl.col("Close"),
        cov_xy.alias("cov_xy"),
        var_x.alias("var_x")
    ]).filter(var_x > threshold)  # 过滤掉 var_x 小于阈值的行

    # 计算滚动回归 beta (在过滤后的数据上)
    result = filtered_df.select(
        (cov_xy / var_x).alias("rolling_regbeta")
    )
    
    # 收集结果并转为 NumPy 数组
    return result.collect().to_numpy()