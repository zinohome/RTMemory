/**
 * MemoryAddClient / MemoryListClient — thin wrappers around the /v1/memories/ API.
 */

import type {
  Memory,
  MemoryAddRequest,
  MemoryAddResponse,
  MemoryForgetRequest,
  MemoryListResponse,
} from "./types";

import { RTMemoryError } from "./client";

/** Helper: build headers from base config. */
function makeHeaders(headers: Record<string, string>): Record<string, string> {
  return { ...headers };
}

// ── MemoryAddClient ───────────────────────────────────────────────

export class MemoryAddClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = makeHeaders(headers);
  }

  /** Add a new memory (triggers extraction pipeline). */
  async add(req: MemoryAddRequest): Promise<MemoryAddResponse> {
    const res = await fetch(`${this.baseUrl}/v1/memories/`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<MemoryAddResponse>;
  }

  /** Forget a memory by ID (soft delete). */
  async forget(memoryId: string, reason?: string): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {};
    if (reason) body["forget_reason"] = reason;
    const res = await fetch(`${this.baseUrl}/v1/memories/${encodeURIComponent(memoryId)}`, {
      method: "DELETE",
      headers: { ...this.headers, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<Record<string, unknown>>;
  }
}

// ── MemoryListClient ──────────────────────────────────────────────

export class MemoryListClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = makeHeaders(headers);
  }

  /** List memories with pagination and optional space filter. */
  async list(params: {
    spaceId?: string;
    limit?: number;
    offset?: number;
  }): Promise<MemoryListResponse> {
    const url = new URL(`${this.baseUrl}/v1/memories/`);
    if (params.spaceId) url.searchParams.set("space_id", params.spaceId);
    if (params.limit !== undefined) url.searchParams.set("limit", String(params.limit));
    if (params.offset !== undefined) url.searchParams.set("offset", String(params.offset));
    const res = await fetch(url.toString(), {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<MemoryListResponse>;
  }

  /** Get a single memory by ID. */
  async get(id: string): Promise<Memory> {
    const res = await fetch(`${this.baseUrl}/v1/memories/${id}`, {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<Memory>;
  }
}
