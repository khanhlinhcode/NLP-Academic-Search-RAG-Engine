from nlp_academic_search.api.security import InMemoryRateLimiter


def test_rate_limiter_is_bounded_per_subject_and_bucket():
    limiter = InMemoryRateLimiter()
    subject = limiter.subject("secret", "127.0.0.1")
    assert limiter.allow(subject, "ask", 2, now=100.0)
    assert limiter.allow(subject, "ask", 2, now=101.0)
    assert not limiter.allow(subject, "ask", 2, now=102.0)
    assert limiter.allow(subject, "search", 2, now=102.0)
    assert limiter.allow(subject, "ask", 2, now=161.1)
