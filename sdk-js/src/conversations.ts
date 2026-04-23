/**
 * ConversationClient — thin wrapper around the /v1/conversations/ API.
 */

import type {
  ConversationAddRequest,
  ConversationAddResponse,
} from "./types";
import { RTMemoryError } from "./client";

export class ConversationClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { ...headers };
  }

  /** Submit a conversation fragment (triggers extraction). */
  async add(req: ConversationAddRequest): Promise<ConversationAddResponse> {
    const res = await fetch(`${this.baseUrl}/v1/conversations/`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<ConversationAddResponse>;
  }

  /** End a conversation (triggers deep scan). */
  async end(conversationId: string, spaceId: string): Promise<ConversationAddResponse> {
    const res = await fetch(`${this.baseUrl}/v1/conversations/end`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({ conversation_id: conversationId, space_id: spaceId }),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<ConversationAddResponse>;
  }
}
