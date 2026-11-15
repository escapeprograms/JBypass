"""Turning a question into chat-formatted model input text, `(lm, question) -> str`.
Use the same formatting in training (inside a `to_text`) and eval so both see the same
template."""

from model.loading import LanguageModel


def chat_prompt(lm: LanguageModel, question: str) -> str:
    """The question as one user turn in the tokenizer's own chat template, ending with the
    assistant prefix. Model quirks (e.g. Qwen3's enable_thinking) come from the profile."""
    return lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        **lm.profile.chat_template_kwargs,
    )
