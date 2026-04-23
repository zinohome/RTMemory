/**
 * DocumentClient — thin wrapper around the /v1/documents/ API.
 */

import type {
  Document,
  DocumentAddRequest,
  DocumentListResponse,
} from "./types";
import { RTMemoryError } from "./client";

export class DocumentClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { ...headers };
  }

  /** Add a document by content (text or URL). */
  async add(req: DocumentAddRequest): Promise<{ id: string; status: string }> {
    const res = await fetch(`${this.baseUrl}/v1/documents/`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<{ id: string; status: string }>;
  }

  /** List documents with optional filters. */
  async list(params: {
    spaceId?: string;
    status?: string;
    sort?: string;
    order?: string;
    limit?: number;
    offset?: number;
  }): Promise<DocumentListResponse> {
    const url = new URL(`${this.baseUrl}/v1/documents/`);
    if (params.spaceId) url.searchParams.set("space_id", params.spaceId);
    if (params.status) url.searchParams.set("status", params.status);
    if (params.sort) url.searchParams.set("sort", params.sort);
    if (params.order) url.searchParams.set("order", params.order);
    if (params.limit !== undefined) url.searchParams.set("limit", String(params.limit));
    if (params.offset !== undefined) url.searchParams.set("offset", String(params.offset));
    const res = await fetch(url.toString(), {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<DocumentListResponse>;
  }

  /** Get a single document by ID. */
  async get(id: string): Promise<Document> {
    const res = await fetch(`${this.baseUrl}/v1/documents/${id}`, {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<Document>;
  }

  /** Delete a document by ID. */
  async delete(id: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/v1/documents/${id}`, {
      method: "DELETE",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<Record<string, unknown>>;
  }
}
