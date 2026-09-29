"""Top-level worker for spawn tests (Windows spawn needs importable functions)."""


def blas_report():
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_info
    return threadpool_info()
