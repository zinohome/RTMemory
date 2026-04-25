/**
 * GraphClient — thin wrapper around graph visualization API endpoints.
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
    relationTypes?: string[];
    direction?: string;
  }): Promise<GraphNeighborhood> {
    const url = new URL(`${this.baseUrl}/v1/graph/neighborhood`);
    url.searchParams.set("entity_id", params.entityId);
    url.searchParams.set("max_hops", String(params.maxHops ?? 3));
    if (params.spaceId) url.searchParams.set("space_id", params.spaceId);
    if (params.direction) url.searchParams.set("direction", params.direction);
    if (params.relationTypes?.length) {
      url.searchParams.set("relation_types", params.relationTypes.join(","));
    }
    const res = await fetch(url.toString(), {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<GraphNeighborhood>;
  }
}