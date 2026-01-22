# rag/prompts/utils.py

from typing import Any

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None


def chat_prompt_from_template(template: str) -> Any:
    """
    Returns a ChatPromptTemplate if LangChain is available,
    otherwise returns a lightweight object with `.format(**kwargs)`.
    """

    if ChatPromptTemplate is not None:
        return ChatPromptTemplate.from_template(template)

    class _FakePrompt:
        def format(self, **kwargs):
            return template.format(**kwargs)

    return _FakePrompt()
