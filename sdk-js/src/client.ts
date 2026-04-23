/**
 * RTMemory JavaScript/TypeScript SDK — async-first client.
 */

import type {
  Memory,
  MemoryAddRequest,
  MemoryAddResponse,
  MemoryForgetRequest,
  MemoryListResponse,
  SearchRequest,
  SearchResponse,
  ProfileRequest,
  ProfileResponse,
  DocumentAddRequest,
  DocumentListResponse,
  ConversationAddRequest,
  ConversationAddResponse,
  SpaceCreateRequest,
  SpaceListResponse,
  GraphNeighborhood,
} from "./types";

export interface RTMemoryConfig {
  baseUrl: string;
  apiKey?: string;
  defaultSpaceId?: string;
}

export class RTMemoryClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(config: RTMemoryConfig) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.headers = { "Content-Type": "application/json" };
    if (config.apiKey) {
      this.headers["Authorization"] = `Bearer ${config.apiKey}`;
    }
  }

  // ── Health ──────────────────────────────────────────────────────────

  async health(): Promise<{ status: string }> {
    return this.get("/health");
  }

  // ── Memories ────────────────────────────────────────────────────────

  async addMemory(req: MemoryAddRequest): Promise<MemoryAddResponse> {
    return this.post("/v1/memories/", req);
  }

  async listMemories(params: {
    spaceId: string;
    limit?: number;
    offset?: number;
  }): Promise<MemoryListResponse> {
    return this.get("/v1/memories/", params);
  }

  async getMemory(id: string): Promise<Memory> {
    return this.get(`/v1/memories/${id}`);
  }

  async forgetMemory(id: string, req: MemoryForgetRequest): Promise<Memory> {
    return this.delete(`/v1/memories/${id}`, req);
  }

  // ── Search ──────────────────────────────────────────────────────────

  async search(req: SearchRequest): Promise<SearchResponse> {
    return this.post("/v1/search/", req);
  }

  // ── Profile ─────────────────────────────────────────────────────────

  async getProfile(req: ProfileRequest): Promise<ProfileResponse> {
    return this.post("/v1/profile", req);
  }

  // ── Documents ────────────────────────────────────────────────────────

  async addDocument(req: DocumentAddRequest): Promise<{ id: string; status: string }> {
    return this.post("/v1/documents/", req);
  }

  async listDocuments(params: {
    spaceId: string;
    limit?: number;
    offset?: number;
  }): Promise<DocumentListResponse> {
    return this.get("/v1/documents/", params);
  }

  // ── Conversations ────────────────────────────────────────────────────

  async addConversation(req: ConversationAddRequest): Promise<ConversationAddResponse> {
    return this.post("/v1/conversations/", req);
  }

  async endConversation(req: {
    conversationId: string;
  }): Promise<ConversationAddResponse> {
    return this.post("/v1/conversations/end", req);
  }

  // ── Spaces ──────────────────────────────────────────────────────────

  async createSpace(req: SpaceCreateRequest): Promise<{ id: string }> {
    return this.post("/v1/spaces/", req);
  }

  async listSpaces(params?: {
    limit?: number;
    offset?: number;
  }): Promise<SpaceListResponse> {
    return this.get("/v1/spaces/", params);
  }

  // ── Graph ────────────────────────────────────────────────────────────

  async getGraphNeighborhood(params: {
    entityId: string;
    maxHops?: number;
  }): Promise<GraphNeighborhood> {
    return this.post("/v1/memories/traverse", {
      entity_id: params.entityId,
      max_hops: params.maxHops ?? 3,
    });
  }

  // ── HTTP helpers ────────────────────────────────────────────────────

  private async get<T>(path: string, params?: Record<string, unknown>): Promise<T> {
    const url = new URL(this.baseUrl + path);
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
      }
    }
    const res = await fetch(url.toString(), {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as T;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.baseUrl + path, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as T;
  }

  private async delete<T>(path: string, body?: unknown): Promise<T> {
    const res = await fetch(this.baseUrl + path, {
      method: "DELETE",
      headers: this.headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as T;
  }
}

export class RTMemoryError extends Error {
  constructor(public status: number, public body: string) {
    super(`RTMemory API error ${status}: ${body.slice(0, 200)}`);
    this.name = "RTMemoryError";
  }
}