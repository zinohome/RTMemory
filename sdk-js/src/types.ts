/**
 * Zod schemas and inferred TypeScript types for RTMemory API objects.
 */

import { z } from "zod";

// ── Enums ──────────────────────────────────────────────────────────────

export const MemoryTypeSchema = z.enum(["fact", "preference", "status", "inference"]);
export type MemoryType = z.infer<typeof MemoryTypeSchema>;

export const SearchModeSchema = z.enum(["hybrid", "memory_only", "documents_only"]);
export type SearchMode = z.infer<typeof SearchModeSchema>;

export const DocumentStatusSchema = z.enum([
  "queued", "extracting", "chunking", "embedding", "done", "failed",
]);
export type DocumentStatus = z.infer<typeof DocumentStatusSchema>;

// ── Memories ───────────────────────────────────────────────────────────

export const MemorySchema = z.object({
  id: z.string(),
  content: z.string(),
  customId: z.string().nullable().optional(),
  memoryType: MemoryTypeSchema.nullable().optional(),
  entityId: z.string().nullable().optional(),
  relationId: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
  decayRate: z.number().default(0.01),
  isForgotten: z.boolean().default(false),
  forgetReason: z.string().nullable().optional(),
  version: z.number().default(1),
  parentId: z.string().nullable().optional(),
  rootId: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
  spaceId: z.string().nullable().optional(),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
});
export type Memory = z.infer<typeof MemorySchema>;

export const MemoryAddRequestSchema = z.object({
  content: z.string(),
  spaceId: z.string(),
  userId: z.string().nullable().optional(),
  customId: z.string().nullable().optional(),
  entityContext: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).nullable().optional(),
});
export type MemoryAddRequest = z.infer<typeof MemoryAddRequestSchema>;

export const MemoryAddResponseSchema = z.object({
  id: z.string(),
  content: z.string(),
  customId: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
  entityId: z.string().nullable().optional(),
  relationIds: z.array(z.string()).default([]),
  memoryType: MemoryTypeSchema.nullable().optional(),
  createdAt: z.string().nullable().optional(),
});
export type MemoryAddResponse = z.infer<typeof MemoryAddResponseSchema>;

export const MemoryForgetRequestSchema = z.object({
  memoryId: z.string().nullable().optional(),
  contentMatch: z.string().nullable().optional(),
  reason: z.string().nullable().optional(),
});
export type MemoryForgetRequest = z.infer<typeof MemoryForgetRequestSchema>;

export const MemoryListResponseSchema = z.object({
  items: z.array(MemorySchema).default([]),
  total: z.number().default(0),
  offset: z.number().default(0),
  limit: z.number().default(20),
});
export type MemoryListResponse = z.infer<typeof MemoryListResponseSchema>;

// ── Search ─────────────────────────────────────────────────────────────

export const SearchFilterSchema = z.object({
  key: z.string(),
  value: z.unknown(),
  operator: z.string().default("eq"),
});

export type SearchFilterGroup = {
  AND?: (z.infer<typeof SearchFilterSchema> | SearchFilterGroup)[] | null;
  OR?: (z.infer<typeof SearchFilterSchema> | SearchFilterGroup)[] | null;
};

export const SearchFilterGroupSchema: z.ZodType<SearchFilterGroup> = z.object({
  AND: z.lazy(() => z.array(z.union([SearchFilterSchema, SearchFilterGroupSchema]))).nullable().optional(),
  OR: z.lazy(() => z.array(z.union([SearchFilterSchema, SearchFilterGroupSchema]))).nullable().optional(),
});

export const SearchRequestSchema = z.object({
  q: z.string(),
  spaceId: z.string().nullable().optional(),
  userId: z.string().nullable().optional(),
  mode: SearchModeSchema.default("hybrid"),
  channels: z.array(z.string()).nullable().optional(),
  limit: z.number().default(10),
  includeProfile: z.boolean().default(false),
  chunkThreshold: z.number().default(0.0),
  documentThreshold: z.number().default(0.0),
  onlyMatchingChunks: z.boolean().default(false),
  includeFullDocs: z.boolean().default(false),
  includeSummary: z.boolean().default(false),
  filters: SearchFilterGroupSchema.nullable().optional(),
  rewriteQuery: z.boolean().default(false),
  rerank: z.boolean().default(false),
});
export type SearchRequest = z.infer<typeof SearchRequestSchema>;

export const SearchResultEntitySchema = z.object({
  name: z.string(),
  type: z.string().nullable().optional(),
});

export const SearchResultDocumentSchema = z.object({
  title: z.string().nullable().optional(),
  url: z.string().nullable().optional(),
});

export const SearchResultSchema = z.object({
  type: z.string(),
  content: z.string(),
  score: z.number(),
  source: z.string(),
  entity: SearchResultEntitySchema.nullable().optional(),
  document: SearchResultDocumentSchema.nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
});
export type SearchResult = z.infer<typeof SearchResultSchema>;

export const ProfileDataSchema = z.object({
  identity: z.record(z.unknown()).default({}),
  preferences: z.record(z.unknown()).default({}),
  currentStatus: z.record(z.unknown()).default({}),
  relationships: z.record(z.unknown()).default({}),
  dynamicMemories: z.array(z.string()).default([]),
});
export type ProfileData = z.infer<typeof ProfileDataSchema>;

export const SearchResponseSchema = z.object({
  results: z.array(SearchResultSchema).default([]),
  profile: ProfileDataSchema.nullable().optional(),
  timingMs: z.number().default(0),
});
export type SearchResponse = z.infer<typeof SearchResponseSchema>;

// ── Profile ────────────────────────────────────────────────────────────

export const ProfileRequestSchema = z.object({
  entityId: z.string(),
  spaceId: z.string(),
  q: z.string().nullable().optional(),
  fresh: z.boolean().default(false),
});
export type ProfileRequest = z.infer<typeof ProfileRequestSchema>;

export const ProfileResponseSchema = z.object({
  profile: ProfileDataSchema,
  confidence: z.record(z.number()).default({}),
  searchResults: z.array(SearchResultSchema).default([]),
  computedAt: z.string().nullable().optional(),
  timingMs: z.number().default(0),
});
export type ProfileResponse = z.infer<typeof ProfileResponseSchema>;

// ── Documents ───────────────────────────────────────────────────────────

export const DocumentSchema = z.object({
  id: z.string(),
  title: z.string().nullable().optional(),
  content: z.string().nullable().optional(),
  docType: z.string().nullable().optional(),
  url: z.string().nullable().optional(),
  status: DocumentStatusSchema.default("queued"),
  summary: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
  spaceId: z.string().nullable().optional(),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
});
export type Document = z.infer<typeof DocumentSchema>;

export const DocumentAddRequestSchema = z.object({
  content: z.string(),
  spaceId: z.string(),
  title: z.string().nullable().optional(),
});
export type DocumentAddRequest = z.infer<typeof DocumentAddRequestSchema>;

export const DocumentListResponseSchema = z.object({
  items: z.array(DocumentSchema).default([]),
  total: z.number().default(0),
  offset: z.number().default(0),
  limit: z.number().default(20),
});
export type DocumentListResponse = z.infer<typeof DocumentListResponseSchema>;

// ── Conversations ──────────────────────────────────────────────────────

export const ConversationMessageSchema = z.object({
  role: z.string(),
  content: z.string(),
});
export type ConversationMessage = z.infer<typeof ConversationMessageSchema>;

export const ConversationAddRequestSchema = z.object({
  messages: z.array(ConversationMessageSchema),
  spaceId: z.string(),
  userId: z.string().nullable().optional(),
});
export type ConversationAddRequest = z.infer<typeof ConversationAddRequestSchema>;

export const ConversationAddResponseSchema = z.object({
  id: z.string(),
  memoryIds: z.array(z.string()).default([]),
  entityIds: z.array(z.string()).default([]),
  createdAt: z.string().nullable().optional(),
});
export type ConversationAddResponse = z.infer<typeof ConversationAddResponseSchema>;

// ── Spaces ──────────────────────────────────────────────────────────────

export const SpaceSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().nullable().optional(),
  orgId: z.string().nullable().optional(),
  ownerId: z.string().nullable().optional(),
  containerTag: z.string().nullable().optional(),
  isDefault: z.boolean().default(false),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
});
export type Space = z.infer<typeof SpaceSchema>;

export const SpaceCreateRequestSchema = z.object({
  name: z.string(),
  description: z.string().nullable().optional(),
});
export type SpaceCreateRequest = z.infer<typeof SpaceCreateRequestSchema>;

export const SpaceListResponseSchema = z.object({
  items: z.array(SpaceSchema).default([]),
  total: z.number().default(0),
});
export type SpaceListResponse = z.infer<typeof SpaceListResponseSchema>;

// ── Graph ───────────────────────────────────────────────────────────────

export const GraphEntitySchema = z.object({
  id: z.string(),
  name: z.string(),
  entityType: z.string().nullable().optional(),
  description: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
});

export const GraphRelationSchema = z.object({
  id: z.string(),
  sourceEntityId: z.string(),
  targetEntityId: z.string(),
  relationType: z.string(),
  value: z.string().nullable().optional(),
  validFrom: z.string().nullable().optional(),
  validTo: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
  isCurrent: z.boolean().default(true),
});

export const GraphNeighborhoodSchema = z.object({
  center: GraphEntitySchema,
  entities: z.array(GraphEntitySchema).default([]),
  relations: z.array(GraphRelationSchema).default([]),
  depth: z.number().default(1),
});
export type GraphNeighborhood = z.infer<typeof GraphNeighborhoodSchema>;