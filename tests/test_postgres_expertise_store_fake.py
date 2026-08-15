from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from ctxforge.core.expertise import Expertise, ExpertiseItem, ExpertiseSection
from ctxforge.storage.postgres.expertise import PostgresExpertiseStore


class _FakeDB:
    def __init__(self):
        self.expertise = {}  # id -> row dict
        self.items = {}  # (expertise_id, item_id) -> row dict


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, db: _FakeDB):
        self._db = db

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql: str, *args):
        sql_s = " ".join(sql.split())

        if sql_s.startswith("INSERT INTO expertise_items"):
            (
                item_id,
                expertise_id,
                section,
                content,
                helpful_count,
                harmful_count,
                source,
                is_active,
                embedding,
                metadata_json,
                created_at,
                updated_at,
            ) = args
            self._db.items[(expertise_id, item_id)] = {
                "item_id": item_id,
                "expertise_id": expertise_id,
                "section": section,
                "content": content,
                "helpful_count": helpful_count,
                "harmful_count": harmful_count,
                "source": source,
                "is_active": is_active,
                "embedding": embedding,
                "metadata": metadata_json,
                "created_at": created_at,
                "updated_at": updated_at,
            }
            return "INSERT 0 1"

        if sql_s.startswith("INSERT INTO expertise"):
            (
                expertise_id,
                name,
                domain,
                description,
                version,
                token_budget,
                next_item_id,
                metadata_json,
                created_at,
                updated_at,
            ) = args
            self._db.expertise[expertise_id] = {
                "expertise_id": expertise_id,
                "name": name,
                "domain": domain,
                "description": description,
                "version": version,
                "token_budget": token_budget,
                "next_item_id": next_item_id,
                "metadata": metadata_json,
                "created_at": created_at,
                "updated_at": updated_at,
            }
            return "INSERT 0 1"

        if sql_s.startswith("DELETE FROM expertise_items WHERE expertise_id"):
            expertise_id = args[0]
            for k in list(self._db.items.keys()):
                if k[0] == expertise_id:
                    del self._db.items[k]
            return "DELETE 0"

        if sql_s.startswith("UPDATE expertise SET updated_at"):
            updated_at, expertise_id = args
            if expertise_id in self._db.expertise:
                self._db.expertise[expertise_id]["updated_at"] = updated_at
            return "UPDATE 1"

        if sql_s.startswith("UPDATE expertise_items SET"):
            (
                section,
                content,
                helpful_count,
                harmful_count,
                source,
                is_active,
                embedding,
                metadata_json,
                updated_at,
                item_id,
                expertise_id,
            ) = args
            key = (expertise_id, item_id)
            if key in self._db.items:
                row = self._db.items[key]
                row.update(
                    {
                        "section": section,
                        "content": content,
                        "helpful_count": helpful_count,
                        "harmful_count": harmful_count,
                        "source": source,
                        "is_active": is_active,
                        "embedding": embedding,
                        "metadata": metadata_json,
                        "updated_at": updated_at,
                    }
                )
            return "UPDATE 1"

        if sql_s.startswith("DELETE FROM expertise_items WHERE item_id"):
            item_id, expertise_id = args
            key = (expertise_id, item_id)
            if key in self._db.items:
                del self._db.items[key]
                return "DELETE 1"
            return "DELETE 0"

        return "OK"

    async def fetchrow(self, sql: str, *args):
        sql_s = " ".join(sql.split())
        if "FROM expertise" in sql_s and "WHERE expertise_id" in sql_s:
            expertise_id = args[0]
            return self._db.expertise.get(expertise_id)

        if "FROM expertise_items" in sql_s and "WHERE item_id" in sql_s:
            item_id, expertise_id = args
            return self._db.items.get((expertise_id, item_id))
        return None

    async def fetch(self, sql: str, *args):
        sql_s = " ".join(sql.split())
        if "SELECT item_id" in sql_s and "FROM expertise_items" in sql_s and "WHERE expertise_id" in sql_s:
            expertise_id = args[0]
            rows = [v for (eid, _), v in self._db.items.items() if eid == expertise_id]
            rows.sort(key=lambda r: r["created_at"])
            return rows

        if "SELECT expertise_id FROM expertise" in sql_s:
            # domain filter or not; always return all ids for this fake
            return [{"expertise_id": eid} for eid in self._db.expertise.keys()]

        return []


class _FakeManager:
    def __init__(self, db: _FakeDB):
        self._db = db
        self.is_connected = True

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    @asynccontextmanager
    async def acquire(self):
        yield _FakeConn(self._db)


@pytest.mark.asyncio
async def test_postgres_expertise_store_save_load_roundtrip():
    db = _FakeDB()
    store = PostgresExpertiseStore(connection_manager=_FakeManager(db))

    exp = Expertise(expertise_id="exp1", name="Test", domain="d")
    exp.items.append(
        ExpertiseItem(
            item_id="it1",
            section=ExpertiseSection.STRATEGIES,
            content="Do X",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    await store.save(exp)

    loaded = await store.load("exp1")
    assert loaded is not None
    assert loaded.expertise_id == "exp1"
    assert loaded.name == "Test"
    assert len(loaded.items) == 1
    assert loaded.items[0].item_id == "it1"


@pytest.mark.asyncio
async def test_postgres_expertise_store_item_crud():
    db = _FakeDB()
    store = PostgresExpertiseStore(connection_manager=_FakeManager(db))

    exp = Expertise(expertise_id="exp2", name="E2")
    await store.save(exp)

    item = ExpertiseItem(
        item_id="it2",
        section=ExpertiseSection.HEURISTICS,
        content="Be concise",
        metadata={"a": 1},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    await store.add_item("exp2", item)
    got = await store.get_item("exp2", "it2")
    assert got is not None
    assert got.content == "Be concise"
    assert got.metadata["a"] == 1

    item.content = "Be very concise"
    await store.update_item("exp2", item)
    got2 = await store.get_item("exp2", "it2")
    assert got2 is not None
    assert got2.content == "Be very concise"

    removed = await store.remove_item("exp2", "it2")
    assert removed is True
    assert await store.get_item("exp2", "it2") is None


