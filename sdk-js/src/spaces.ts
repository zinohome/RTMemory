/**
 * SpaceClient — thin wrapper around the /v1/spaces/ API.
 */

import type { Space, SpaceCreateRequest, SpaceListResponse } from "./types";
import { RTMemoryError } from "./client";

export class SpaceClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { ...headers };
  }

  /** Create a new space. */
  async create(req: SpaceCreateRequest): Promise<{ id: string }> {
    const res = await fetch(`${this.baseUrl}/v1/spaces/`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<{ id: string }>;
  }

  /** List all spaces. */
  async list(): Promise<SpaceListResponse> {
    const res = await fetch(`${this.baseUrl}/v1/spaces/`, {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<SpaceListResponse>;
  }

  /** Get space details by ID. */
  async get(id: string): Promise<Space> {
    const res = await fetch(`${this.baseUrl}/v1/spaces/${id}`, {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<Space>;
  }

  /** Delete a space by ID. */
  async delete(id: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/v1/spaces/${id}`, {
      method: "DELETE",
      headers: this.headers,
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<Record<string, unknown>>;
  }
}
