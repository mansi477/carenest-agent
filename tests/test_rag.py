from app.rag import Retriever, _window, load_chunks, tokenize


def test_stemming_connects_word_forms():
    assert tokenize("rescheduling") == tokenize("reschedule")
    assert tokenize("vaccinated") == tokenize("vaccination")


def test_every_chunk_keeps_its_heading_and_size_limit():
    chunks = load_chunks(max_chars=300, overlap=50)
    assert chunks
    for c in chunks:
        assert c.text.startswith(c.heading)
        assert len(c.text) <= 300 + len(c.heading) + 2


def test_window_splits_long_text_with_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    pieces = _window(text, max_chars=120, overlap=40)
    assert len(pieces) > 1
    assert all(len(p) <= 120 for p in pieces)
    assert any(pieces[i].split(". ")[-1] in pieces[i + 1] for i in range(len(pieces) - 1))


def test_answerable_questions_find_the_right_file():
    r = Retriever(load_chunks())
    cases = {
        "How many times can I reschedule an appointment?": "appointments.md",
        "Do I need to fast before a blood sugar test?": "diagnostics.md",
        "How long should I wait after getting vaccinated?": "vaccination.md",
        "Is there a delivery charge for medicines?": "medicine_delivery.md",
        "Money got debited but my booking failed": "billing_privacy.md",
    }
    for question, source in cases.items():
        hits = r.search(question)
        assert hits and hits[0].chunk.source == source, question
        assert r.is_confident(hits)


def test_out_of_scope_questions_are_not_confident():
    r = Retriever(load_chunks())
    for question in ["Do you have a branch in Singapore?", "What is the capital of France?",
                     "Do you offer dental implants?"]:
        assert not r.is_confident(r.search(question)), question
