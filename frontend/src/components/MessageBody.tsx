import type { ReactElement } from "react";

/**
 * Renders model output strictly as a React text child.
 *
 * Deliberately no HTML parsing and no `dangerouslySetInnerHTML`: an answer that
 * contains `<script>` or `<img onerror=...>` is displayed literally and never
 * becomes an element or an event handler.
 */
export function MessageBody({ content }: { content: string }): ReactElement {
  return <div className="message-body">{content}</div>;
}
