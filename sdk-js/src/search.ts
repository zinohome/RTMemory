/**
 * SearchClient — thin wrapper around the /v1/search/ API.
 */

import type { SearchRequest, SearchResponse, SearchMode } from "./types";
import { RTMemoryError } from "./client";

export class SearchClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { ...headers };
  }

  /** Execute a hybrid search across memories, documents, and graph. */
  async search(req: SearchRequest): Promise<SearchResponse> {
    const res = await fetch(`${this.baseUrl}/v1/search/`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<SearchResponse>;
  }

  /** Convenience: quick search with just a query string. */
  async quickSearch(
    q: string,
    spaceId?: string,
    mode: SearchMode = "hybrid",
    limit: number = 10,
  ): Promise<SearchResponse> {
    return this.search({
      q,
      spaceId: spaceId ?? null,
      mode,
      limit,
    });
  }
}
