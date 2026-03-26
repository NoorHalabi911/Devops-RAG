import React, { useEffect, useMemo, useRef, useState } from "react";
import { askQuestion } from "./api.js";

function Message({ role, content }) {
  return (
    <div className={`msg msg--${role}`}>
      <div className="msg__bubble">{content}</div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! Ask me a DevOps question and I’ll answer using your PDF knowledge base.",
      retrieval: null,
    },
  ]);
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(5);
  const [debug, setDebug] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const listRef = useRef(null);
  const abortRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  useEffect(() => {
    // Scroll to bottom whenever messages change.
    if (!listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, loading]);

  async function onSend() {
    const question = input.trim();
    if (!question) return;
    setError(null);

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question, retrieval: null },
    ]);
    setInput("");

    try {
      const data = await askQuestion(question, {
        topK,
        debug,
        signal: controller.signal,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "",
          retrieval: data.retrieval || null,
        },
      ]);
    } catch (e) {
      const msg =
        e?.name === "AbortError"
          ? "Request cancelled."
          : e?.message || "Something went wrong while generating the answer.";
      setError(msg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry - ${msg}`,
          retrieval: null,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <div className="brand__title">OpsRecall</div>
          <div className="brand__subtitle">DevOps RAG assistant</div>
        </div>
        <div className="controls">
          <label className="control">
            <span>Retrieved chunks</span>
            <input
              type="range"
              min="3"
              max="8"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
            <span className="control__value">{topK}</span>
          </label>
          <label className="control">
            <span>Debug</span>
            <input
              type="checkbox"
              checked={debug}
              onChange={(e) => setDebug(e.target.checked)}
            />
          </label>
          <button
            className="btn btn--secondary"
            type="button"
            onClick={() =>
              setMessages([
                {
                  role: "assistant",
                  content:
                    "Hi! Ask me a DevOps question and I’ll answer using your PDF knowledge base.",
                  retrieval: null,
                },
              ])
            }
            disabled={loading}
          >
            Clear chat
          </button>
        </div>
      </div>

      <div className="chat" aria-label="chat">
        <div className="chat__list" ref={listRef}>
          {messages.map((m, idx) => (
            <div key={idx}>
              <Message role={m.role} content={m.content} />
              {m.role === "assistant" && m.retrieval && (
                <div className="retrieval">
                  <div className="retrieval__meta">
                    Retrieved {m.retrieval.n_chunks} chunks
                    {m.retrieval.best_distance != null ? (
                      <>
                        {" "}
                        • Best distance: {Number(m.retrieval.best_distance).toFixed(4)}
                      </>
                    ) : null}
                  </div>

                  {m.retrieval.best_distance != null && Number(m.retrieval.best_distance) > 1.4 ? (
                    <div className="retrieval__warning">
                      Retrieval quality looks weak. The answer may be based on poor context.
                    </div>
                  ) : null}

                  {debug && m.retrieval.chunks && (
                    <details className="retrieval__details" open={false}>
                      <summary>Retrieved source chunks</summary>
                      <div className="retrieval__chunks">
                        {m.retrieval.chunks.map((c, i) => (
                          <details key={i} className="retrieval__chunk">
                            <summary>
                              Chunk {i + 1}
                              {m.retrieval.distances && m.retrieval.distances[i] != null
                                ? ` • distance=${Number(m.retrieval.distances[i]).toFixed(4)}`
                                : ""}
                            </summary>
                            <pre>{c}</pre>
                          </details>
                        ))}
                      </div>
                      {m.retrieval.prompt ? (
                        <details className="retrieval__chunk">
                          <summary>Prompt sent to LLM</summary>
                          <pre>{m.retrieval.prompt}</pre>
                        </details>
                      ) : null}
                    </details>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="chat__composer">
          <textarea
            className="composer__input"
            value={input}
            placeholder="Ask a DevOps question (e.g. Why is my port not working?)"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (canSend) onSend();
              }
            }}
          />
          <button
            className="btn btn--primary"
            type="button"
            disabled={!canSend}
            onClick={onSend}
          >
            {loading ? "Generating..." : "Send"}
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}
      </div>
    </div>
  );
}

