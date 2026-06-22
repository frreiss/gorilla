"""Snippetization code used by the Tavily integration.

This fork of gorilla substitutes a Tavily search API for the benchmark's usual 
SerpAPI implementation. This file contains code that is used to shorten Tavily's
result snippets to roughly the same length as SerpAPI's.
"""

import threading

import pysbd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# One segmenter is enough; it is stateless across calls. ``char_span=True`` makes
# :meth:`segment` return spans carrying the start/end offsets of each sentence in
# the original text, which lets us slice out a verbatim snippet.
_SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)
_LOCK = threading.Lock()


def shorten_snippet(query: str, document: str, max_chars: int = 200,
              hard_max_chars: int = 500) -> str:
    """Quick and dirty snippet generation for search results.

    There are no open source implementations of this operation that are actively
    maintained, so we roll or own. This version is intended for shortening Tavily
    result summaries so that they are about as long as the summaries of the SerpAPI
    API that the BFCLv4 benchmark uses. Snippet quality is mediocre, which is ok
    because it makes things more difficult for the LLM under test.

    The algorithm used is as follows:
    Split ``document`` into sentences, then find the contiguous window of
    sentences that (a) is as long as possible without exceeding ``max_chars``
    characters and (b) among such windows, maximizes TF-IDF cosine similarity
    with ``query``. The window is returned verbatim, exactly as it appears in
    ``document``.

    To bias the result toward the target length, only *maximal* windows are
    considered -- windows that cannot be extended by another sentence on either
    side without exceeding ``max_chars``. The best-matching maximal window is
    then chosen. This fills the character budget while still preferring the
    region of the document most relevant to the query.

    A window is allowed to consist of a single sentence even when that sentence
    is longer than ``max_chars`` -- otherwise documents whose every sentence
    exceeds the limit could not be summarized at all. ``max_chars`` is therefore
    a soft target that is only exceeded by an unavoidable single sentence.

    :param query: Keyword query that the snippet should match.
    :param document: Full text of a document that matches ``query``.
    :param max_chars: Soft upper bound on the snippet length, in characters.
    :param hard_max_chars: Hard upper bound; strings longer than this length are
        truncated mid-sentence.

    :returns: A verbatim snippet of ``document``, or the empty string if
        ``document`` or ``query`` contains no usable text.
    """
    # Something below isn't thread-safe; probably the segmenter, but it could be 
    # something inside sklearn that runs with the GIL disabled. The overhead of this
    # operation is so small that we can just run everything in a critical section.
    with _LOCK:
        if not query.strip() or not document.strip():
            return ""
        if hard_max_chars < max_chars:
            raise ValueError(
                f"Hard maximum {hard_max_chars} less than soft limit {max_chars}")

        spans = _SEGMENTER.segment(document)
        sentences = [span.sent for span in spans]
        if not sentences:
            return ""

        # Fit the vectorizer on the sentences plus the query so that query terms are
        # guaranteed to be in the vocabulary. If the document and query share no
        # vocabulary at all, every similarity is zero and we fall back to the first
        # sentence below.
        vectorizer = TfidfVectorizer()
        try:
            vectorizer.fit(sentences + [query])
        except ValueError:
            # Raised when there is no vocabulary (e.g. only stop words / punctuation).
            return document[spans[0].start : spans[0].end].strip()

        query_vec = vectorizer.transform([query])

        # For each start sentence i, grow the window as far as possible without
        # exceeding max_chars; that greedy endpoint is the maximal window for i. A
        # single sentence is always kept even if it alone exceeds max_chars, so every
        # start position contributes exactly one candidate. Keeping only these
        # maximal windows biases the result toward the target length: a shorter
        # window is considered only when it starts at a sentence that no longer
        # window can.
        candidate_texts: list[str] = []
        candidate_bounds: list[tuple[int, int]] = []  # (start_offset, end_offset)
        for i in range(len(spans)):
            j = i
            while (
                j + 1 < len(spans)
                and spans[j + 1].end - spans[i].start <= max_chars
            ):
                j += 1
            start, end = spans[i].start, spans[j].end
            candidate_texts.append(document[start:end])
            candidate_bounds.append((start, end))

        candidate_vecs = vectorizer.transform(candidate_texts)
        scores = cosine_similarity(candidate_vecs, query_vec).ravel()

        # Pick the most query-similar maximal window. On ties (e.g. all-zero
        # similarity), argmax returns the earliest, which is the conventional
        # fallback to the start of the document.
        best = int(scores.argmax())
        start, end = candidate_bounds[best]

        # Enforce hard limit and add ellipsis to indicate mid-sentence breaks
        if end - start <= hard_max_chars:
            return document[start:end].strip()

        # If we get here, we went over the hard limit. Split mid-sentence and add an
        # ellipsis.
        ELLIPSIS = "[...]"
        return document[start:start + hard_max_chars - len(ELLIPSIS)].strip() + ELLIPSIS
