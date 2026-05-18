import type { KeyboardEvent, RefObject } from "react";
import { SUGGESTED_PROMPTS } from "../constants";
import type { ChatMessage } from "../types";

type Props = {
  hasContext: boolean;
  llmWarning: string | null;
  messages: ChatMessage[];
  chatQuestion: string;
  setChatQuestion: (value: string) => void;
  sendQuestion: (question: string) => Promise<void>;
  threadRef: RefObject<HTMLDivElement>;
};

export function DecisionChatPanel({ hasContext, llmWarning, messages, chatQuestion, setChatQuestion, sendQuestion, threadRef }: Props) {
  return <section className="panel decision-support-chat-panel polished-chat-panel">
    <header className="chat-panel-header">
      <h3>AI Decision Support</h3>
    </header>

    <div className="chat-thread polished-chat-thread" ref={threadRef}>
      {!hasContext ? <p className="chat-empty-state">No forecast context is available yet.</p> : null}
      {llmWarning ? <p className="chat-empty-state">{llmWarning}</p> : null}
      {messages.map((message, index) => <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}><p>{message.content}</p></article>)}
    </div>

    <div className="suggested-prompts">
      {SUGGESTED_PROMPTS.map((prompt) => <button key={prompt} type="button" className="chip-button" onClick={() => void sendQuestion(prompt)} disabled={!hasContext}>{prompt}</button>)}
    </div>

    <div className="chat-composer">
      <textarea value={chatQuestion} onChange={(event) => setChatQuestion(event.target.value)} onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          void sendQuestion(chatQuestion);
        }
      }} placeholder="Ask a grounded question about this forecast" disabled={!hasContext} />
      <button className="primary-button" onClick={() => void sendQuestion(chatQuestion)} disabled={!hasContext || !chatQuestion.trim()}>Ask</button>
    </div>
  </section>;
}
