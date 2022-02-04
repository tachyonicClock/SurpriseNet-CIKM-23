

def ho_not(func):
    """Higher order not. For lambdas and the such"""
    def _ho_not(*args, **kwargs):
        return not func(*args, **kwargs)

