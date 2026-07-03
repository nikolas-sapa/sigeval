"""Copy-paste example: a real pytest file gating an LLM feature with sigeval.

Run:  python -m pytest examples/ -q
Uses a stub model so it runs offline. Swap `fake_llm` for your real client.
"""

import random

from sigeval import assert_eval, make_judge

# --- your model call (replace this) ------------------------------------------
_rng = random.Random(0)
def fake_llm(_prompt: str) -> str:
    # pretends to answer on-topic 90% of the time
    return "Your refund will be processed in 5 days." if _rng.random() < 0.9 \
        else "I like turtles."


def complete(prompt: str) -> str:
    # a judge model would go here; stubbed to grade the fake_llm output
    return "PASS" if "refund" in prompt else "FAIL"


# --- the eval ----------------------------------------------------------------
def test_refund_answer_stays_on_topic():
    judge = make_judge(complete, criterion="answer is about the refund")

    def scorer(question):
        answer = fake_llm(question)
        return judge(answer)

    # green ONLY if the true on-topic rate is significantly above 80%
    assert_eval("refund_on_topic", scorer,
                sample="how long for my refund?",
                n_samples=30, threshold=0.8)
