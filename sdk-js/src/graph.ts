/**
 * GraphClient — thin wrapper around graph traversal API endpoints.
 */

import type { GraphNeighborhood } from "./types";
import { RTMemoryError } from "./client";

export class GraphClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { ...headers };
  }

  /** Get the neighborhood of an entity in the knowledge graph. */
  async neighborhood(params: {
    entityId: string;
    spaceId?: string;
    maxHops?: number;
  }): Promise<GraphNeighborhood> {
    const body: Record<string, unknown> = {
      entity_id: params.entityId,
      max_hops: params.maxHops ?? 3,
    };
    if (params.spaceId) body["space_id"] = params.spaceId;
    const res = await fetch(`${this.baseUrl}/v1/memories/traverse`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<GraphNeighborhood>;
  }
}