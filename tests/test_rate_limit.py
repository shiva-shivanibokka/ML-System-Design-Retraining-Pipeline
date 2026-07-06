"""T12: per-client sliding-window rate limiter."""
from serving.rate_limit import SlidingWindowLimiter


def test_sliding_window_blocks_over_limit_and_slides():
    lim = SlidingWindowLimiter(max_requests=2, window_seconds=10)
    assert lim.allow("a", now=0.0) is True
    assert lim.allow("a", now=1.0) is True
    assert lim.allow("a", now=2.0) is False   # 3rd hit inside the window
    # a different client is unaffected
    assert lim.allow("b", now=2.0) is True
    # once the window slides past the first hit, "a" is allowed again
    assert lim.allow("a", now=11.0) is True
