"""
Tests for Vector Store implementations.

Tests the IVectorStore protocol implementations:
- PineconeStore
- ChromaDBStore
- WeaviateStore

Uses mocks to simulate the underlying vector database clients.
"""

import sys
from unittest.mock import MagicMock

import pytest

from ctxforge.vectorstores.chroma_store import (
    ChromaConfig,
    ChromaDBStore,
)
from ctxforge.vectorstores.pinecone_store import (
    PineconeConfig,
    PineconeStore,
    VectorStoreError,
)
from ctxforge.vectorstores.protocol import (
    DistanceMetric,
    IVectorStore,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
)
from ctxforge.vectorstores.weaviate_store import (
    WeaviateConfig,
    WeaviateStore,
)

# =============================================================================
# Protocol Tests
# =============================================================================

class TestVectorRecord:
    """Tests for VectorRecord dataclass."""
    
    def test_create_vector_record(self):
        """Test creating a vector record."""
        record = VectorRecord(
            id="vec_123",
            embedding=[0.1, 0.2, 0.3],
            metadata={"user_id": "user_1"},
            content="Test content",
        )
        
        assert record.id == "vec_123"
        assert record.embedding == [0.1, 0.2, 0.3]
        assert record.metadata == {"user_id": "user_1"}
        assert record.content == "Test content"
    
    def test_create_record_without_optional_fields(self):
        """Test creating record with only required fields."""
        record = VectorRecord(
            id="vec_123",
            embedding=[0.1, 0.2],
        )
        
        assert record.id == "vec_123"
        assert record.metadata == {}
        assert record.content is None
    
    def test_record_id_cannot_be_empty(self):
        """Test that ID cannot be empty."""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            VectorRecord(id="", embedding=[0.1])
    
    def test_record_embedding_cannot_be_empty(self):
        """Test that embedding cannot be empty."""
        with pytest.raises(ValueError, match="embedding cannot be empty"):
            VectorRecord(id="vec_1", embedding=[])


class TestVectorQueryResult:
    """Tests for VectorQueryResult dataclass."""
    
    def test_create_query_result(self):
        """Test creating a query result."""
        result = VectorQueryResult(
            id="vec_123",
            score=0.95,
            embedding=[0.1, 0.2],
            metadata={"type": "semantic"},
            content="Test content",
        )
        
        assert result.id == "vec_123"
        assert result.score == 0.95
        assert result.embedding == [0.1, 0.2]
        assert result.metadata == {"type": "semantic"}
        assert result.content == "Test content"


class TestQueryFilter:
    """Tests for QueryFilter dataclass."""
    
    def test_to_pinecone_eq(self):
        """Test converting equality filter to Pinecone format."""
        f = QueryFilter(field="user_id", operator="eq", value="user_1")
        assert f.to_pinecone() == {"user_id": "user_1"}
    
    def test_to_pinecone_gt(self):
        """Test converting greater than filter to Pinecone format."""
        f = QueryFilter(field="score", operator="gt", value=0.5)
        assert f.to_pinecone() == {"score": {"$gt": 0.5}}
    
    def test_to_pinecone_in(self):
        """Test converting 'in' filter to Pinecone format."""
        f = QueryFilter(field="type", operator="in", value=["a", "b"])
        assert f.to_pinecone() == {"type": {"$in": ["a", "b"]}}
    
    def test_to_chroma_eq(self):
        """Test converting equality filter to ChromaDB format."""
        f = QueryFilter(field="user_id", operator="eq", value="user_1")
        assert f.to_chroma() == {"user_id": {"$eq": "user_1"}}
    
    def test_to_chroma_contains(self):
        """Test converting contains filter to ChromaDB format."""
        f = QueryFilter(field="content", operator="contains", value="test")
        assert f.to_chroma() == {"content": {"$contains": "test"}}


# =============================================================================
# PineconeStore Tests
# =============================================================================

class TestPineconeConfig:
    """Tests for PineconeConfig."""
    
    def test_create_config(self):
        """Test creating Pinecone config."""
        config = PineconeConfig(
            api_key="test-key",
            index_name="test-index",
            dimension=1536,
            namespace="test-ns",
        )
        
        assert config.api_key == "test-key"
        assert config.index_name == "test-index"
        assert config.dimension == 1536
        assert config.namespace == "test-ns"
    
    def test_default_values(self):
        """Test default config values."""
        config = PineconeConfig(
            api_key="test-key",
            index_name="test-index",
        )
        
        assert config.dimension == 1536
        assert config.metric == DistanceMetric.COSINE
        assert config.batch_size == 100
        assert config.serverless_cloud == "aws"


class TestPineconeStore:
    """Tests for PineconeStore implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return PineconeConfig(
            api_key="test-key",
            index_name="test-index",
            dimension=3,
            namespace="test",
        )
    
    @pytest.fixture
    def mock_pinecone_module(self):
        """Create mock Pinecone module and inject into sys.modules."""
        # Create mock module
        mock_module = MagicMock()
        
        # Mock index info
        mock_index_info = MagicMock()
        mock_index_info.name = "test-index"
        
        # Mock client class
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = [mock_index_info]
        
        # Mock index
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        
        mock_module.Pinecone.return_value = mock_client
        mock_module.ServerlessSpec = MagicMock()
        mock_module.PodSpec = MagicMock()
        
        # Inject mock
        sys.modules['pinecone'] = mock_module
        
        yield mock_module, mock_client, mock_index
        
        # Cleanup
        del sys.modules['pinecone']
    
    @pytest.mark.asyncio
    async def test_initialize(self, config, mock_pinecone_module):
        """Test store initialization."""
        _, mock_client, _ = mock_pinecone_module
        
        store = PineconeStore(config)
        await store.initialize()
        
        assert store._initialized
        mock_client.Index.assert_called_once_with("test-index")
    
    @pytest.mark.asyncio
    async def test_properties(self, config, mock_pinecone_module):
        """Test store properties."""
        store = PineconeStore(config)
        await store.initialize()
        
        assert store.name == "pinecone:test-index"
        assert store.dimension == 3
        assert store.metric == DistanceMetric.COSINE
    
    @pytest.mark.asyncio
    async def test_upsert(self, config, mock_pinecone_module):
        """Test upserting vectors."""
        _, _, mock_index = mock_pinecone_module
        
        # Mock upsert response
        mock_upsert_response = MagicMock()
        mock_upsert_response.upserted_count = 2
        mock_index.upsert.return_value = mock_upsert_response
        
        store = PineconeStore(config)
        await store.initialize()
        
        vectors = [
            VectorRecord(id="v1", embedding=[0.1, 0.2, 0.3], metadata={"user": "1"}),
            VectorRecord(id="v2", embedding=[0.4, 0.5, 0.6], metadata={"user": "2"}),
        ]
        
        count = await store.upsert(vectors)
        
        assert count == 2
        mock_index.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query(self, config, mock_pinecone_module):
        """Test querying vectors."""
        _, _, mock_index = mock_pinecone_module
        
        # Mock query response
        mock_match = MagicMock()
        mock_match.id = "v1"
        mock_match.score = 0.95
        mock_match.values = None
        mock_match.metadata = {"user": "1"}
        
        mock_result = MagicMock()
        mock_result.matches = [mock_match]
        mock_index.query.return_value = mock_result
        
        store = PineconeStore(config)
        await store.initialize()
        
        results = await store.query(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )
        
        assert len(results) == 1
        assert results[0].id == "v1"
        assert results[0].score == 0.95
        assert results[0].metadata == {"user": "1"}
    
    @pytest.mark.asyncio
    async def test_query_with_filters(self, config, mock_pinecone_module):
        """Test querying with metadata filters."""
        _, _, mock_index = mock_pinecone_module
        
        mock_result = MagicMock()
        mock_result.matches = []
        mock_index.query.return_value = mock_result
        
        store = PineconeStore(config)
        await store.initialize()
        
        filters = [QueryFilter(field="user_id", operator="eq", value="user_1")]
        
        await store.query(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            filters=filters,
        )
        
        # Verify filter was passed
        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == {"user_id": "user_1"}
    
    @pytest.mark.asyncio
    async def test_fetch(self, config, mock_pinecone_module):
        """Test fetching vectors by ID."""
        _, _, mock_index = mock_pinecone_module
        
        # Mock fetch response
        mock_vector = MagicMock()
        mock_vector.values = [0.1, 0.2, 0.3]
        mock_vector.metadata = {"user": "1"}
        
        mock_result = MagicMock()
        mock_result.vectors = {"v1": mock_vector}
        mock_index.fetch.return_value = mock_result
        
        store = PineconeStore(config)
        await store.initialize()
        
        records = await store.fetch(["v1"])
        
        assert "v1" in records
        assert records["v1"].embedding == [0.1, 0.2, 0.3]
        assert records["v1"].metadata == {"user": "1"}
    
    @pytest.mark.asyncio
    async def test_delete(self, config, mock_pinecone_module):
        """Test deleting vectors."""
        _, _, mock_index = mock_pinecone_module
        
        store = PineconeStore(config)
        await store.initialize()
        
        count = await store.delete(["v1", "v2"])
        
        assert count == 2
        mock_index.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_count(self, config, mock_pinecone_module):
        """Test counting vectors."""
        _, _, mock_index = mock_pinecone_module
        
        # Mock stats response
        mock_ns_stats = MagicMock()
        mock_ns_stats.vector_count = 100
        
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 100
        mock_stats.namespaces = {"test": mock_ns_stats}
        mock_index.describe_index_stats.return_value = mock_stats
        
        store = PineconeStore(config)
        await store.initialize()
        
        count = await store.count(namespace="test")
        
        assert count == 100
    
    @pytest.mark.asyncio
    async def test_list_namespaces(self, config, mock_pinecone_module):
        """Test listing namespaces."""
        _, _, mock_index = mock_pinecone_module
        
        # Mock stats response
        mock_stats = MagicMock()
        mock_stats.namespaces = {"ns1": MagicMock(), "ns2": MagicMock()}
        mock_index.describe_index_stats.return_value = mock_stats
        
        store = PineconeStore(config)
        await store.initialize()
        
        namespaces = await store.list_namespaces()
        
        assert "ns1" in namespaces
        assert "ns2" in namespaces
    
    @pytest.mark.asyncio
    async def test_not_initialized_error(self, config):
        """Test error when not initialized."""
        store = PineconeStore(config)
        
        with pytest.raises(VectorStoreError, match="not initialized"):
            await store.query([0.1, 0.2, 0.3])
    
    @pytest.mark.asyncio
    async def test_close(self, config, mock_pinecone_module):
        """Test closing connection."""
        store = PineconeStore(config)
        await store.initialize()
        
        assert store._initialized
        
        await store.close()
        
        assert not store._initialized
        assert store._index is None


# =============================================================================
# ChromaDBStore Tests
# =============================================================================

class TestChromaConfig:
    """Tests for ChromaConfig."""
    
    def test_create_config(self):
        """Test creating ChromaDB config."""
        config = ChromaConfig(
            collection_name="memories",
            persist_directory="/tmp/chroma",
            dimension=1536,
        )
        
        assert config.collection_name == "memories"
        assert config.persist_directory == "/tmp/chroma"
        assert config.dimension == 1536
    
    def test_default_values(self):
        """Test default config values."""
        config = ChromaConfig()
        
        assert config.collection_name == "memories"
        assert config.persist_directory is None
        assert config.host is None
        assert config.create_collection_if_missing


class TestChromaDBStore:
    """Tests for ChromaDBStore implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return ChromaConfig(
            collection_name="test_collection",
            dimension=3,
        )
    
    @pytest.fixture
    def mock_chromadb_module(self):
        """Create mock ChromaDB module and inject into sys.modules."""
        # Create mock module
        mock_module = MagicMock()
        
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        
        # Mock client
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        
        mock_module.Client.return_value = mock_client
        mock_module.PersistentClient.return_value = mock_client
        mock_module.HttpClient.return_value = mock_client
        
        # Mock Settings
        mock_settings = MagicMock()
        mock_module.config.Settings = mock_settings
        
        # Inject mock
        sys.modules['chromadb'] = mock_module
        sys.modules['chromadb.config'] = mock_module.config
        
        yield mock_module, mock_client, mock_collection
        
        # Cleanup
        del sys.modules['chromadb']
        del sys.modules['chromadb.config']
    
    @pytest.mark.asyncio
    async def test_initialize(self, config, mock_chromadb_module):
        """Test store initialization."""
        _, mock_client, mock_collection = mock_chromadb_module
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        assert store._initialized
        mock_client.get_or_create_collection.assert_called()
    
    @pytest.mark.asyncio
    async def test_properties(self, config, mock_chromadb_module):
        """Test store properties."""
        store = ChromaDBStore(config)
        await store.initialize()
        
        assert store.name == "chromadb:test_collection"
        assert store.dimension == 3
        assert store.metric == DistanceMetric.COSINE
    
    @pytest.mark.asyncio
    async def test_upsert(self, config, mock_chromadb_module):
        """Test upserting vectors."""
        _, _, mock_collection = mock_chromadb_module
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        vectors = [
            VectorRecord(id="v1", embedding=[0.1, 0.2, 0.3], metadata={"user": "1"}),
            VectorRecord(id="v2", embedding=[0.4, 0.5, 0.6], content="test"),
        ]
        
        count = await store.upsert(vectors)
        
        assert count == 2
        mock_collection.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query(self, config, mock_chromadb_module):
        """Test querying vectors."""
        _, _, mock_collection = mock_chromadb_module
        
        # Mock query response
        mock_collection.query.return_value = {
            "ids": [["v1", "v2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"user": "1"}, {"user": "2"}]],
            "documents": [["doc1", "doc2"]],
            "embeddings": None,
        }
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        results = await store.query(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )
        
        assert len(results) == 2
        assert results[0].id == "v1"
        assert results[0].metadata == {"user": "1"}
        assert results[0].content == "doc1"
    
    @pytest.mark.asyncio
    async def test_query_score_conversion(self, config, mock_chromadb_module):
        """Test that distance is converted to similarity score."""
        _, _, mock_collection = mock_chromadb_module
        
        # Mock query response - distance of 0 means identical
        mock_collection.query.return_value = {
            "ids": [["v1"]],
            "distances": [[0.0]],
            "metadatas": [[{}]],
            "documents": [[None]],
        }
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        results = await store.query(embedding=[0.1, 0.2, 0.3])
        
        # For cosine, distance 0 should give score 1.0
        assert results[0].score == 1.0
    
    @pytest.mark.asyncio
    async def test_fetch(self, config, mock_chromadb_module):
        """Test fetching vectors by ID."""
        _, _, mock_collection = mock_chromadb_module
        
        # Mock get response
        mock_collection.get.return_value = {
            "ids": ["v1"],
            "embeddings": [[0.1, 0.2, 0.3]],
            "metadatas": [{"user": "1"}],
            "documents": ["test doc"],
        }
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        records = await store.fetch(["v1"])
        
        assert "v1" in records
        assert records["v1"].embedding == [0.1, 0.2, 0.3]
        assert records["v1"].content == "test doc"
    
    @pytest.mark.asyncio
    async def test_delete(self, config, mock_chromadb_module):
        """Test deleting vectors."""
        _, _, mock_collection = mock_chromadb_module
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        count = await store.delete(["v1", "v2"])
        
        assert count == 2
        mock_collection.delete.assert_called_once_with(ids=["v1", "v2"])
    
    @pytest.mark.asyncio
    async def test_count(self, config, mock_chromadb_module):
        """Test counting vectors."""
        _, _, mock_collection = mock_chromadb_module
        mock_collection.count.return_value = 42
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        count = await store.count()
        
        assert count == 42
    
    @pytest.mark.asyncio
    async def test_namespace_creates_separate_collection(self, config, mock_chromadb_module):
        """Test that namespaces create separate collections."""
        _, mock_client, _ = mock_chromadb_module
        
        store = ChromaDBStore(config)
        await store.initialize()
        
        # Access namespace
        await store._get_collection("user_123")
        
        # Should create collection with namespace suffix
        calls = mock_client.get_or_create_collection.call_args_list
        collection_names = [call[1]["name"] for call in calls]
        
        assert "test_collection_user_123" in collection_names


# =============================================================================
# WeaviateStore Tests
# =============================================================================

class TestWeaviateConfig:
    """Tests for WeaviateConfig."""
    
    def test_create_config(self):
        """Test creating Weaviate config."""
        config = WeaviateConfig(
            url="http://localhost:8080",
            class_name="Memory",
            dimension=1536,
        )
        
        assert config.url == "http://localhost:8080"
        assert config.class_name == "Memory"
        assert config.dimension == 1536
    
    def test_default_values(self):
        """Test default config values."""
        config = WeaviateConfig()
        
        assert config.url == "http://localhost:8080"
        assert config.class_name == "Memory"
        assert config.grpc_port == 50051
        assert config.enable_hybrid_search


class TestWeaviateStore:
    """Tests for WeaviateStore implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return WeaviateConfig(
            url="http://localhost:8080",
            class_name="TestMemory",
            dimension=3,
        )
    
    @pytest.fixture
    def mock_weaviate_module(self):
        """Create mock Weaviate module and inject into sys.modules."""
        # Create mock module
        mock_module = MagicMock()
        
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.config.get.return_value = MagicMock(properties=[])
        
        # Mock aggregate result
        mock_agg_result = MagicMock()
        mock_agg_result.total_count = 0
        mock_collection.aggregate.over_all.return_value = mock_agg_result
        
        # Mock collections manager
        mock_collections = MagicMock()
        mock_collections.exists.return_value = True
        mock_collections.get.return_value = mock_collection
        mock_collections.list_all.return_value = {}
        
        # Mock client
        mock_client = MagicMock()
        mock_client.collections = mock_collections
        
        mock_module.connect_to_local.return_value = mock_client
        mock_module.connect_to_weaviate_cloud.return_value = mock_client
        
        # Mock classes.init
        mock_classes_init = MagicMock()
        mock_classes_init.Auth = MagicMock()
        mock_classes_init.Auth.api_key.return_value = MagicMock()
        mock_module.classes = MagicMock()
        mock_module.classes.init = mock_classes_init
        
        # Mock classes.config
        mock_classes_config = MagicMock()
        mock_classes_config.Configure = MagicMock()
        mock_classes_config.Property = MagicMock()
        mock_classes_config.DataType = MagicMock()
        mock_module.classes.config = mock_classes_config
        
        # Mock classes.query
        mock_classes_query = MagicMock()
        mock_classes_query.MetadataQuery = MagicMock()
        mock_classes_query.Filter = MagicMock()
        mock_module.classes.query = mock_classes_query
        
        # Inject mocks
        sys.modules['weaviate'] = mock_module
        sys.modules['weaviate.classes'] = mock_module.classes
        sys.modules['weaviate.classes.init'] = mock_classes_init
        sys.modules['weaviate.classes.config'] = mock_classes_config
        sys.modules['weaviate.classes.query'] = mock_classes_query
        sys.modules['weaviate.classes.data'] = MagicMock()
        
        yield mock_module, mock_client, mock_collection
        
        # Cleanup
        del sys.modules['weaviate']
        del sys.modules['weaviate.classes']
        del sys.modules['weaviate.classes.init']
        del sys.modules['weaviate.classes.config']
        del sys.modules['weaviate.classes.query']
        del sys.modules['weaviate.classes.data']
    
    @pytest.mark.asyncio
    async def test_initialize_local(self, config, mock_weaviate_module):
        """Test store initialization with local connection."""
        mock_module, mock_client, _ = mock_weaviate_module
        
        store = WeaviateStore(config)
        await store.initialize()
        
        assert store._initialized
        mock_module.connect_to_local.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_cloud(self, mock_weaviate_module):
        """Test store initialization with cloud connection."""
        mock_module, mock_client, _ = mock_weaviate_module
        
        config = WeaviateConfig(
            url="https://test.weaviate.network",
            api_key="test-key",
            class_name="TestMemory",
        )
        
        store = WeaviateStore(config)
        await store.initialize()
        
        assert store._initialized
        mock_module.connect_to_weaviate_cloud.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_properties(self, config, mock_weaviate_module):
        """Test store properties."""
        store = WeaviateStore(config)
        await store.initialize()
        
        assert store.name == "weaviate:TestMemory"
        assert store.dimension == 3
        assert store.metric == DistanceMetric.COSINE
    
    @pytest.mark.asyncio
    async def test_query(self, config, mock_weaviate_module):
        """Test querying vectors."""
        _, _, mock_collection = mock_weaviate_module
        
        # Mock query response
        mock_obj = MagicMock()
        mock_obj.uuid = "123e4567-e89b-12d3-a456-426614174000"
        mock_obj.properties = {"content": "test", "metadata_json": "{}"}
        mock_obj.metadata = MagicMock(distance=0.1)
        mock_obj.vector = None
        
        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_result
        
        store = WeaviateStore(config)
        await store.initialize()
        
        results = await store.query(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )
        
        assert len(results) == 1
        assert results[0].content == "test"
    
    @pytest.mark.asyncio
    async def test_count(self, config, mock_weaviate_module):
        """Test counting vectors."""
        _, _, mock_collection = mock_weaviate_module
        
        mock_result = MagicMock()
        mock_result.total_count = 42
        mock_collection.aggregate.over_all.return_value = mock_result
        
        store = WeaviateStore(config)
        await store.initialize()
        
        count = await store.count()
        
        assert count == 42
    
    @pytest.mark.asyncio
    async def test_close(self, config, mock_weaviate_module):
        """Test closing connection."""
        _, mock_client, _ = mock_weaviate_module
        
        store = WeaviateStore(config)
        await store.initialize()
        
        await store.close()
        
        assert not store._initialized
        mock_client.close.assert_called_once()
    
    def test_valid_uuid_check(self, config):
        """Test UUID validation helper."""
        store = WeaviateStore(config)
        
        assert store._is_valid_uuid("123e4567-e89b-12d3-a456-426614174000")
        assert not store._is_valid_uuid("not-a-uuid")
        assert not store._is_valid_uuid("12345")


# =============================================================================
# Protocol Compliance Tests
# =============================================================================

class TestProtocolCompliance:
    """Test that all stores implement IVectorStore protocol."""
    
    def test_pinecone_implements_protocol(self):
        """Test PineconeStore implements IVectorStore."""
        config = PineconeConfig(api_key="test", index_name="test")
        store = PineconeStore(config)
        assert isinstance(store, IVectorStore)
    
    def test_chromadb_implements_protocol(self):
        """Test ChromaDBStore implements IVectorStore."""
        config = ChromaConfig()
        store = ChromaDBStore(config)
        assert isinstance(store, IVectorStore)
    
    def test_weaviate_implements_protocol(self):
        """Test WeaviateStore implements IVectorStore."""
        config = WeaviateConfig()
        store = WeaviateStore(config)
        assert isinstance(store, IVectorStore)


# =============================================================================
# Integration-Style Tests (with mocks)
# =============================================================================

class TestVectorStoreWorkflows:
    """Test common vector store workflows."""
    
    @pytest.fixture
    def mock_chromadb_for_workflow(self):
        """Create a fully mocked ChromaDB for workflow tests."""
        # Create mock module
        mock_module = MagicMock()
        
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        
        # Mock client
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        
        mock_module.Client.return_value = mock_client
        mock_module.PersistentClient.return_value = mock_client
        
        # Mock Settings
        mock_module.config = MagicMock()
        mock_module.config.Settings = MagicMock()
        
        # Inject mock
        sys.modules['chromadb'] = mock_module
        sys.modules['chromadb.config'] = mock_module.config
        
        yield mock_collection
        
        # Cleanup
        del sys.modules['chromadb']
        del sys.modules['chromadb.config']
    
    @pytest.mark.asyncio
    async def test_upsert_and_query_workflow(self, mock_chromadb_for_workflow):
        """Test typical upsert then query workflow."""
        mock_collection = mock_chromadb_for_workflow
        
        # Configure mock responses
        mock_collection.query.return_value = {
            "ids": [["v1"]],
            "distances": [[0.1]],
            "metadatas": [[{"user_id": "user_1"}]],
            "documents": [["User prefers dark mode"]],
        }
        
        config = ChromaConfig(dimension=1536)
        store = ChromaDBStore(config)
        await store.initialize()
        
        # Upsert some vectors
        vectors = [
            VectorRecord(
                id="v1",
                embedding=[0.1] * 1536,
                metadata={"user_id": "user_1", "type": "preference"},
                content="User prefers dark mode",
            ),
        ]
        await store.upsert(vectors)
        
        # Query for similar
        query_embedding = [0.1] * 1536
        results = await store.query(
            embedding=query_embedding,
            top_k=5,
            filters=[QueryFilter(field="user_id", operator="eq", value="user_1")],
        )
        
        assert len(results) == 1
        assert results[0].content == "User prefers dark mode"
    
    @pytest.mark.asyncio
    async def test_memory_lifecycle(self, mock_chromadb_for_workflow):
        """Test full lifecycle: create, query, update, delete."""
        mock_collection = mock_chromadb_for_workflow
        
        config = ChromaConfig(dimension=3)
        store = ChromaDBStore(config)
        await store.initialize()
        
        # 1. Create
        vectors = [
            VectorRecord(id="mem_1", embedding=[0.1, 0.2, 0.3], content="fact 1"),
            VectorRecord(id="mem_2", embedding=[0.4, 0.5, 0.6], content="fact 2"),
        ]
        count = await store.upsert(vectors)
        assert count == 2
        
        # 2. Query
        mock_collection.query.return_value = {
            "ids": [["mem_1"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
            "documents": [["fact 1"]],
        }
        results = await store.query([0.1, 0.2, 0.3])
        assert len(results) == 1
        
        # 3. Update (upsert with same ID)
        updated = [VectorRecord(id="mem_1", embedding=[0.2, 0.3, 0.4], content="updated fact")]
        await store.upsert(updated)
        mock_collection.upsert.assert_called()
        
        # 4. Delete
        deleted = await store.delete(["mem_1"])
        assert deleted == 1
        mock_collection.delete.assert_called_with(ids=["mem_1"])
