/**
 * ProfileClient — thin wrapper around the /v1/profile API.
 */

import type { ProfileRequest, ProfileResponse } from "./types";
import { RTMemoryError } from "./client";

export class ProfileClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { ...headers };
  }

  /** Get (or compute) a user profile from the knowledge graph. */
  async get(req: ProfileRequest): Promise<ProfileResponse> {
    const res = await fetch(`${this.baseUrl}/v1/profile`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new RTMemoryError(res.status, await res.text());
    return res.json() as Promise<ProfileResponse>;
  }
}
