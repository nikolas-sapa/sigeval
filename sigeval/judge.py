"""LLM-as-judge scorer helper — provider-agnostic.

Most evals score with a judge model. sigeval doesn't ship a provider client
(no lock-in); you pass a `complete` callable — any function that takes a prompt
string and returns the model's text. Works with OpenAI, Anthropic, Ollama, a
local vLLM, or a stub in tests.

The scorer this builds returns a bool per sample, which is exactly what the
sigeval runner consumes — so judge noise flows through the same statistical
machinery as everything else.
"""

DEFAULT_TEMPLATE = (
    "You are a strict evaluator. Given the CRITERION and the OUTPUT, answer with "
    "a single word: PASS if the output meets the criterion, FAIL otherwise.\n\n"
    "CRITERION: {criterion}\n\nOUTPUT:\n{output}\n\nVerdict:"
)


def make_judge(complete, criterion, template=DEFAULT_TEMPLATE):
    """Build a scorer(output) -> bool from a judge model.

    complete:  callable(prompt: str) -> str   (your model call)
    criterion: the pass condition, in plain language
    """
    def scorer(output):
        prompt = template.format(criterion=criterion, output=output)
        reply = complete(prompt).strip().upper()
        # ponytail: judges sometimes pad the verdict ("PASS." / "PASS - because")
        return reply.startswith("PASS")
    return scorer
