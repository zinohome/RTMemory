"""GraphEngine -- central data access layer for the temporal knowledge graph."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Entity, Memory, MemorySource, Relation
from app.schemas.graph import (
    EntityCreate,
    EntityListOut,
    EntityOut,
    EntityUpdate,
    GraphTraversalOut,
    GraphTraversalParams,
    MemoryCreate,
    MemoryForget,
    MemoryListOut,
    MemoryOut,
    MemorySourceCreate,
    MemorySourceOut,
    MemoryUpdate,
    MemoryVersionChainOut,
    RelationCreate,
    RelationListOut,
    RelationOut,
    RelationUpdate,
    TraversedRelationOut,
)


class GraphEngine:
    """Central data access layer for all knowledge graph operations.

    Receives an async SQLAlchemy session via dependency injection.
    All methods are async and operate on the database through this session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Decay rate defaults by memory type ─────────────────────

    DECAY_RATES: dict[str, float] = {
        "fact": 0.005,
        "preference": 0.005,
        "status": 0.02,
        "inference": 0.05,
    }

    # ── Entity CRUD ────────────────────────────────────────────

    async def create_entity(self, data: EntityCreate) -> EntityOut:
        """Create a new entity."""
        now = datetime.now(timezone.utc)
        entity = Entity(
            id=uuid.uuid4(),
            name=data.name,
            entity_type=data.entity_type.value,
            description=data.description,
            confidence=data.confidence,
            org_id=data.org_id,
            space_id=data.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(entity)
        await self.session.flush()
        return EntityOut.model_validate(entity)

    async def get_entity(self, entity_id: uuid.UUID) -> Optional[EntityOut]:
        """Get an entity by ID. Returns None if not found."""
        stmt = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            return None
        return EntityOut.model_validate(entity)

    async def list_entities(
        self,
        *,
        org_id: uuid.UUID,
        space_id: Optional[uuid.UUID] = None,
        entity_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> EntityListOut:
        """List entities with optional filters and pagination."""
        stmt = select(Entity).where(Entity.org_id == org_id)
        count_stmt = select(func.count()).select_from(Entity).where(Entity.org_id == org_id)

        if space_id is not None:
            stmt = stmt.where(Entity.space_id == space_id)
            count_stmt = count_stmt.where(Entity.space_id == space_id)
        if entity_type is not None:
            stmt = stmt.where(Entity.entity_type == entity_type)
            count_stmt = count_stmt.where(Entity.entity_type == entity_type)

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Entity.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        entities = result.scalars().all()

        return EntityListOut(
            items=[EntityOut.model_validate(e) for e in entities],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def update_entity(
        self, entity_id: uuid.UUID, data: EntityUpdate
    ) -> EntityOut:
        """Update an entity's mutable fields."""
        stmt = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            raise ValueError("Entity not found")

        update_data = data.model_dump(exclude_unset=True)
        if update_data:
            # Convert entity_type enum to string if present
            if "entity_type" in update_data and update_data["entity_type"] is not None:
                update_data["entity_type"] = update_data["entity_type"].value
            update_data["updated_at"] = datetime.now(timezone.utc)
            for key, value in update_data.items():
                setattr(entity, key, value)
            await self.session.flush()

        return EntityOut.model_validate(entity)

    async def delete_entity(self, entity_id: uuid.UUID) -> None:
        """Hard-delete an entity by ID."""
        stmt = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            raise ValueError("Entity not found")

        await self.session.delete(entity)
        await self.session.flush()

    # ── Relation CRUD ──────────────────────────────────────────

    async def create_relation(self, data: RelationCreate) -> RelationOut:
        """Create a new relation with temporal defaults.

        Handles contradictions: if a current relation with the same
        source_entity_id + relation_type already exists:
          - Same target => reaffirm: increment source_count on existing
          - Different target => contradiction: close old, insert new
        """
        now = datetime.now(timezone.utc)

        # Check for existing current relation with same source + type
        stmt = select(Relation).where(
            Relation.source_entity_id == data.source_entity_id,
            Relation.relation_type == data.relation_type,
            Relation.is_current == True,  # noqa: E712
            Relation.org_id == data.org_id,
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            if existing.target_entity_id == data.target_entity_id:
                # Reaffirm: same source + type + target -> increment source_count
                existing.source_count += 1
                existing.updated_at = now
                existing.confidence = max(existing.confidence, data.confidence)
                await self.session.flush()
                return RelationOut.model_validate(existing)
            else:
                # Contradiction: different target -> close old, insert new
                existing.is_current = False
                existing.valid_to = now
                existing.updated_at = now
                await self.session.flush()

        relation = Relation(
            id=uuid.uuid4(),
            source_entity_id=data.source_entity_id,
            target_entity_id=data.target_entity_id,
            relation_type=data.relation_type,
            value=data.value,
            valid_from=now,
            valid_to=None,
            confidence=data.confidence,
            is_current=True,
            source_count=1,
            org_id=data.org_id,
            space_id=data.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(relation)
        await self.session.flush()
        return RelationOut.model_validate(relation)

    async def get_relation(self, relation_id: uuid.UUID) -> Optional[RelationOut]:
        """Get a relation by ID. Returns None if not found."""
        stmt = select(Relation).where(Relation.id == relation_id)
        result = await self.session.execute(stmt)
        relation = result.scalar_one_or_none()
        if relation is None:
            return None
        return RelationOut.model_validate(relation)

    async def list_relations(
        self,
        *,
        org_id: uuid.UUID,
        space_id: Optional[uuid.UUID] = None,
        source_entity_id: Optional[uuid.UUID] = None,
        target_entity_id: Optional[uuid.UUID] = None,
        relation_type: Optional[str] = None,
        is_current: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RelationListOut:
        """List relations with optional filters and pagination."""
        stmt = select(Relation).where(Relation.org_id == org_id)
        count_stmt = select(func.count()).select_from(Relation).where(Relation.org_id == org_id)

        if space_id is not None:
            stmt = stmt.where(Relation.space_id == space_id)
            count_stmt = count_stmt.where(Relation.space_id == space_id)
        if source_entity_id is not None:
            stmt = stmt.where(Relation.source_entity_id == source_entity_id)
            count_stmt = count_stmt.where(Relation.source_entity_id == source_entity_id)
        if target_entity_id is not None:
            stmt = stmt.where(Relation.target_entity_id == target_entity_id)
            count_stmt = count_stmt.where(Relation.target_entity_id == target_entity_id)
        if relation_type is not None:
            stmt = stmt.where(Relation.relation_type == relation_type)
            count_stmt = count_stmt.where(Relation.relation_type == relation_type)
        if is_current is not None:
            stmt = stmt.where(Relation.is_current == is_current)
            count_stmt = count_stmt.where(Relation.is_current == is_current)

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Relation.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        relations = result.scalars().all()

        return RelationListOut(
            items=[RelationOut.model_validate(r) for r in relations],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def update_relation(
        self, relation_id: uuid.UUID, data: RelationUpdate
    ) -> RelationOut:
        """Update a relation's mutable fields (value, confidence, is_current)."""
        stmt = select(Relation).where(Relation.id == relation_id)
        result = await self.session.execute(stmt)
        relation = result.scalar_one_or_none()
        if relation is None:
            raise ValueError("Relation not found")

        update_data = data.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc)
            for key, value in update_data.items():
                setattr(relation, key, value)
            await self.session.flush()

        return RelationOut.model_validate(relation)

    async def delete_relation(self, relation_id: uuid.UUID) -> None:
        """Hard-delete a relation by ID."""
        stmt = select(Relation).where(Relation.id == relation_id)
        result = await self.session.execute(stmt)
        relation = result.scalar_one_or_none()
        if relation is None:
            raise ValueError("Relation not found")

        await self.session.delete(relation)
        await self.session.flush()

    # ── Graph Traversal ────────────────────────────────────────

    async def traverse_graph(self, params: GraphTraversalParams) -> GraphTraversalOut:
        """Traverse the knowledge graph starting from an entity using recursive CTE.

        Traverses up to max_hops levels, following current relations only.
        Supports direction filtering (outgoing, incoming, both) and
        relation type filtering.

        Requires PostgreSQL for recursive CTE support.
        """
        direction = params.direction
        relation_types = params.relation_types

        # Build the recursive CTE using raw SQL for PostgreSQL
        if direction == "both":
            sql = self._build_both_direction_cte(params.max_hops, relation_types)
        elif direction == "outgoing":
            sql = self._build_single_direction_cte(
                "source_entity_id", "target_entity_id", params.max_hops, relation_types
            )
        else:  # incoming
            sql = self._build_single_direction_cte(
                "target_entity_id", "source_entity_id", params.max_hops, relation_types
            )

        # Execute the CTE query
        result = await self.session.execute(
            text(sql), {"start_entity_id": params.entity_id}
        )
        rows = result.fetchall()

        # Collect unique entity IDs and relation data
        entity_ids = {params.entity_id}
        traversed_relations = []
        seen_relation_ids = set()

        for row in rows:
            rel_id, src_eid, tgt_eid, rel_type, value, valid_from, valid_to, confidence, is_current, source_count, org_id_r, space_id_r, created_at, updated_at, hop, direction_label = row

            entity_ids.add(src_eid)
            entity_ids.add(tgt_eid)

            if rel_id not in seen_relation_ids:
                seen_relation_ids.add(rel_id)
                traversed_relations.append(
                    TraversedRelationOut(
                        relation=RelationOut(
                            id=rel_id,
                            source_entity_id=src_eid,
                            target_entity_id=tgt_eid,
                            relation_type=rel_type,
                            value=value or "",
                            valid_from=valid_from,
                            valid_to=valid_to,
                            confidence=confidence,
                            is_current=is_current,
                            source_count=source_count,
                            org_id=org_id_r,
                            space_id=space_id_r,
                            created_at=created_at,
                            updated_at=updated_at,
                        ),
                        hop=hop,
                        direction=direction_label,
                    )
                )

        # Fetch all discovered entities
        entities = []
        if entity_ids:
            stmt = select(Entity).where(Entity.id.in_(entity_ids))
            ent_result = await self.session.execute(stmt)
            entities = [EntityOut.model_validate(e) for e in ent_result.scalars().all()]

        return GraphTraversalOut(
            start_entity_id=params.entity_id,
            entities=entities,
            relations=traversed_relations,
            max_hops=params.max_hops,
        )

    def _build_single_direction_cte(
        self, src_col: str, tgt_col: str, max_hops: int, relation_types: list[str] | None
    ) -> str:
        """Build recursive CTE SQL for a single traversal direction."""
        type_filter = ""
        if relation_types:
            type_list = ",".join(f"'{rt}'" for rt in relation_types)
            type_filter = f"AND r.relation_type IN ({type_list})"

        return f"""
        WITH RECURSIVE graph_traverse AS (
            SELECT
                r.id AS rel_id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                1 AS hop,
                '{src_col}' AS direction
            FROM relations r
            WHERE r.{src_col} = :start_entity_id
              AND r.is_current = true
              {type_filter}

            UNION ALL

            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                gt.hop + 1,
                '{src_col}' AS direction
            FROM relations r
            JOIN graph_traverse gt ON r.{src_col} = gt.{tgt_col}
            WHERE r.is_current = true
              AND gt.hop < {max_hops}
              {type_filter}
        )
        SELECT * FROM graph_traverse ORDER BY hop
        """

    def _build_both_direction_cte(
        self, max_hops: int, relation_types: list[str] | None
    ) -> str:
        """Build recursive CTE SQL for both traversal directions."""
        type_filter = ""
        if relation_types:
            type_list = ",".join(f"'{rt}'" for rt in relation_types)
            type_filter = f"AND r.relation_type IN ({type_list})"

        return f"""
        WITH RECURSIVE graph_traverse AS (
            SELECT
                r.id AS rel_id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                1 AS hop,
                'outgoing' AS direction
            FROM relations r
            WHERE r.source_entity_id = :start_entity_id
              AND r.is_current = true
              {type_filter}

            UNION ALL

            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                1 AS hop,
                'incoming' AS direction
            FROM relations r
            WHERE r.target_entity_id = :start_entity_id
              AND r.is_current = true
              {type_filter}

            UNION ALL

            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                gt.hop + 1,
                'outgoing' AS direction
            FROM relations r
            JOIN graph_traverse gt ON r.source_entity_id = gt.target_entity_id
            WHERE r.is_current = true
              AND gt.hop < {max_hops}
              AND gt.direction = 'outgoing'
              {type_filter}

            UNION ALL

            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                gt.hop + 1,
                'incoming' AS direction
            FROM relations r
            JOIN graph_traverse gt ON r.target_entity_id = gt.source_entity_id
            WHERE r.is_current = true
              AND gt.hop < {max_hops}
              AND gt.direction = 'incoming'
              {type_filter}
        )
        SELECT DISTINCT ON (rel_id) * FROM graph_traverse ORDER BY rel_id, hop
        """

    # ── Memory CRUD ────────────────────────────────────────────

    async def create_memory(self, data: MemoryCreate) -> MemoryOut:
        """Create a new memory. Sets version=1, root_id=self, parent_id=None."""
        now = datetime.now(timezone.utc)
        memory_id = uuid.uuid4()
        decay_rate = data.decay_rate
        if decay_rate is None:
            decay_rate = self.DECAY_RATES.get(data.memory_type.value, 0.01)

        memory = Memory(
            id=memory_id,
            content=data.content,
            custom_id=data.custom_id,
            memory_type=data.memory_type.value,
            entity_id=data.entity_id,
            relation_id=data.relation_id,
            confidence=data.confidence,
            decay_rate=decay_rate,
            is_forgotten=False,
            forget_at=None,
            forget_reason=None,
            version=1,
            parent_id=None,
            root_id=memory_id,  # First version: root is self
            metadata=data.metadata,
            org_id=data.org_id,
            space_id=data.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(memory)
        await self.session.flush()

        # Create memory-source links if document_ids provided
        if data.document_ids:
            for doc_id in data.document_ids:
                source = MemorySource(
                    memory_id=memory_id,
                    document_id=doc_id,
                    relevance_score=0.0,
                )
                self.session.add(source)
            await self.session.flush()

        return MemoryOut.model_validate(memory)

    async def get_memory(self, memory_id: uuid.UUID) -> Optional[MemoryOut]:
        """Get a memory by ID. Returns None if not found."""
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            return None
        return MemoryOut.model_validate(memory)

    async def get_memory_version_chain(
        self, memory_id: uuid.UUID
    ) -> MemoryVersionChainOut:
        """Get a memory with its full version chain.

        Given any memory ID in the chain, finds the root and returns
        all versions ordered by version number.
        """
        # First, find the memory to get its root_id
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            raise ValueError("Memory not found")

        root_id = memory.root_id

        # Fetch all versions in the chain
        chain_stmt = (
            select(Memory)
            .where(Memory.root_id == root_id)
            .order_by(Memory.version.asc())
        )
        chain_result = await self.session.execute(chain_stmt)
        versions = [MemoryOut.model_validate(m) for m in chain_result.scalars().all()]

        # Current = highest version
        current = versions[-1] if versions else MemoryOut.model_validate(memory)

        return MemoryVersionChainOut(current=current, versions=versions)

    async def list_memories(
        self,
        *,
        org_id: uuid.UUID,
        space_id: Optional[uuid.UUID] = None,
        memory_type: Optional[str] = None,
        entity_id: Optional[uuid.UUID] = None,
        include_forgotten: bool = False,
        latest_versions_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> MemoryListOut:
        """List memories with optional filters and pagination.

        By default excludes forgotten memories and returns only the
        latest version of each memory chain.
        """
        stmt = select(Memory).where(Memory.org_id == org_id)
        count_stmt = select(func.count()).select_from(Memory).where(Memory.org_id == org_id)

        if space_id is not None:
            stmt = stmt.where(Memory.space_id == space_id)
            count_stmt = count_stmt.where(Memory.space_id == space_id)
        if not include_forgotten:
            stmt = stmt.where(Memory.is_forgotten == False)  # noqa: E712
            count_stmt = count_stmt.where(Memory.is_forgotten == False)  # noqa: E712
        if memory_type is not None:
            stmt = stmt.where(Memory.memory_type == memory_type)
            count_stmt = count_stmt.where(Memory.memory_type == memory_type)
        if entity_id is not None:
            stmt = stmt.where(Memory.entity_id == entity_id)
            count_stmt = count_stmt.where(Memory.entity_id == entity_id)
        if latest_versions_only:
            # Only include the latest version: where id is NOT a parent_id of another memory
            subq = select(Memory.parent_id).where(Memory.parent_id.isnot(None))
            stmt = stmt.where(Memory.id.notin_(subq))
            count_stmt = count_stmt.where(Memory.id.notin_(subq))

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Memory.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        memories = result.scalars().all()

        return MemoryListOut(
            items=[MemoryOut.model_validate(m) for m in memories],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def update_memory(
        self, memory_id: uuid.UUID, data: MemoryUpdate
    ) -> MemoryOut:
        """Update a memory by creating a new version.

        Instead of mutating the existing row, creates a new Memory row
        with version+1, parent_id pointing to the old memory, and the
        same root_id.
        """
        now = datetime.now(timezone.utc)

        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        old_memory = result.scalar_one_or_none()
        if old_memory is None:
            raise ValueError("Memory not found")

        new_version = old_memory.version + 1
        new_id = uuid.uuid4()

        # Build new content/metadata from update data
        new_content = data.content if data.content is not None else old_memory.content
        new_confidence = data.confidence if data.confidence is not None else old_memory.confidence
        new_decay_rate = data.decay_rate if data.decay_rate is not None else old_memory.decay_rate
        new_metadata = data.metadata if data.metadata is not None else old_memory.metadata_

        new_memory = Memory(
            id=new_id,
            content=new_content,
            custom_id=old_memory.custom_id,
            memory_type=old_memory.memory_type,
            entity_id=old_memory.entity_id,
            relation_id=old_memory.relation_id,
            confidence=new_confidence,
            decay_rate=new_decay_rate,
            is_forgotten=False,
            forget_at=None,
            forget_reason=None,
            version=new_version,
            parent_id=memory_id,
            root_id=old_memory.root_id,
            metadata=new_metadata,
            org_id=old_memory.org_id,
            space_id=old_memory.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(new_memory)
        await self.session.flush()

        # Copy memory sources from old to new
        source_stmt = select(MemorySource).where(MemorySource.memory_id == memory_id)
        source_result = await self.session.execute(source_stmt)
        for source in source_result.scalars().all():
            new_source = MemorySource(
                memory_id=new_id,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                relevance_score=source.relevance_score,
            )
            self.session.add(new_source)
        await self.session.flush()

        return MemoryOut.model_validate(new_memory)

    async def forget_memory(
        self, memory_id: uuid.UUID, data: MemoryForget
    ) -> MemoryOut:
        """Soft-delete a memory by setting is_forgotten=true.

        Sets forget_reason and forget_at timestamp.
        """
        now = datetime.now(timezone.utc)

        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            raise ValueError("Memory not found")

        memory.is_forgotten = True
        memory.forget_reason = data.forget_reason
        memory.forget_at = now
        memory.updated_at = now
        await self.session.flush()

        return MemoryOut.model_validate(memory)

    # ── Memory-Document Source Tracking ────────────────────────

    async def add_memory_source(self, data: MemorySourceCreate) -> MemorySourceOut:
        """Link a memory to its source document (and optionally chunk)."""
        source = MemorySource(
            memory_id=data.memory_id,
            document_id=data.document_id,
            chunk_id=data.chunk_id,
            relevance_score=data.relevance_score,
        )
        self.session.add(source)
        await self.session.flush()
        return MemorySourceOut.model_validate(source)

    async def get_memory_sources(self, memory_id: uuid.UUID) -> list[MemorySourceOut]:
        """Get all source links for a memory."""
        stmt = select(MemorySource).where(MemorySource.memory_id == memory_id)
        result = await self.session.execute(stmt)
        return [MemorySourceOut.model_validate(s) for s in result.scalars().all()]