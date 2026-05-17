from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a retrieved document is relevant to a user question.\n"
            "If the document mentions keywords or semantic concepts related to the question, grade it as relevant.\n"
            "Give a binary score 'yes' or 'no'. The goal is to filter out clearly unrelated retrievals, not to be strict.",
        ),
        (
            "human",
            "Retrieved document:\n\n{document}\n\nUser question:\n\n{question}",
        ),
    ]
)

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful technical documentation assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context is insufficient, say you don't have enough information.\n"
            "Be concise and accurate. Use code blocks for code.\n"
            "Use the chat history only to resolve pronouns and follow-up references; "
            "do not invent facts from it.",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        (
            "human",
            "Context:\n\n{context}\n\nQuestion: {question}",
        ),
    ]
)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader checking whether an answer is grounded in the provided context.\n"
            "Score 'yes' only if every factual claim in the answer is directly supported by the context.\n"
            "Score 'no' if the answer adds facts not present in the context, contradicts the context,\n"
            "or makes up details. Give a binary 'yes' or 'no'.",
        ),
        (
            "human",
            "Context:\n\n{context}\n\nAnswer:\n\n{generation}",
        ),
    ]
)

STRICT_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT technical documentation assistant. A previous answer attempt was found to\n"
            "introduce facts not in the context. Re-answer using ONLY the context below.\n"
            "Quote or paraphrase tightly. If something is not explicitly in the context, say so plainly.\n"
            "Do not infer beyond the text. Be concise.",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        (
            "human",
            "Context:\n\n{context}\n\nQuestion: {question}",
        ),
    ]
)
