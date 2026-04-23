from genai_research.prompting import PromptStrategy, build_messages, build_user_content


def test_single_with_context():
    t = build_user_content("What is X?", context="X is 1.", strategy=PromptStrategy.SINGLE)
    assert "X is 1" in t
    assert "What is X?" in t
    assert t == "X is 1.\n\nWhat is X?\n"


def test_double_query_contains_two_questions():
    t = build_user_content(
        "Capital of France?",
        context="France is in Europe.",
        strategy=PromptStrategy.DOUBLE_QUERY,
    )
    assert t.count("Capital of France?") >= 2


def test_double_query_no_context_is_query_repeated():
    t = build_user_content("What is 2+2?", context=None, strategy=PromptStrategy.DOUBLE_QUERY)
    assert t == "Question: What is 2+2?\nQuestion: What is 2+2?\n"


def test_sandwich_rag_context_query_repeated_shape():
    t = build_user_content(
        "Q?",
        context="passage-unique-xyz",
        strategy=PromptStrategy.SANDWICH,
    )
    assert t == "passage-unique-xyz\n\nQ?\n\npassage-unique-xyz\n\nQ?\n"


def test_build_messages_roles():
    m = build_messages("Hi", strategy=PromptStrategy.SINGLE, system="Be brief.")
    assert m[0]["role"] == "system"
    assert m[1]["role"] == "user"
