import typing

T = typing.TypeVar("T")
def index_where(predicate: typing.Callable[[T], bool], seq: typing.Sequence[T]):
    """Return the index where the predicate returns true"""
    for i, x in enumerate(seq):
        if predicate(x):
            yield i
