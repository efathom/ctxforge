"""
Component Registry for the ctxforge framework.

Provides a centralized registry for pluggable components,
enabling dynamic registration and lookup of implementations.
"""

import importlib
import warnings
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """
    Registry for pluggable components.
    
    Allows registering and looking up implementations of various
    component types (LLM providers, storage backends, retrievers, etc.)
    
    Example:
        registry = ComponentRegistry()
        
        @registry.register_llm("custom")
        class CustomLLM:
            ...
        
        llm_class = registry.get_llm("custom")
    """
    
    def __init__(self):
        self._llm_providers: Dict[str, Type] = {}
        self._embedding_providers: Dict[str, Type] = {}
        self._session_stores: Dict[str, Type] = {}
        self._memory_stores: Dict[str, Type] = {}
        self._expertise_stores: Dict[str, Type] = {}
        self._scoped_memory_stores: Dict[str, Type] = {}
        self._skill_stores: Dict[str, Type] = {}
        self._retrievers: Dict[str, Type] = {}
        self._compactors: Dict[str, Type] = {}
        self._extractors: Dict[str, Type] = {}
        self._middleware: Dict[str, Type] = {}
        self._middleware_factories: Dict[str, Any] = {}
        self._rerankers: Dict[str, Type] = {}
        self._assemblers: Dict[str, Type] = {}
    
    # ==========================================================================
    # LLM Providers
    # ==========================================================================
    
    def register_llm(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register an LLM provider."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._llm_providers[name.lower()] = cls
            return cls
        return decorator
    
    def get_llm(self, name: str) -> Optional[Type]:
        """Get a registered LLM provider by name."""
        return self._llm_providers.get(name.lower())
    
    def list_llm_providers(self) -> list:
        """List all registered LLM providers."""
        return list(self._llm_providers.keys())
    
    # ==========================================================================
    # Embedding Providers
    # ==========================================================================
    
    def register_embedding(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register an embedding provider."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._embedding_providers[name.lower()] = cls
            return cls
        return decorator
    
    def get_embedding(self, name: str) -> Optional[Type]:
        """Get a registered embedding provider by name."""
        return self._embedding_providers.get(name.lower())
    
    def list_embedding_providers(self) -> list:
        """List all registered embedding providers."""
        return list(self._embedding_providers.keys())
    
    # ==========================================================================
    # Session Stores
    # ==========================================================================
    
    def register_session_store(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a session store."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._session_stores[name.lower()] = cls
            return cls
        return decorator
    
    def get_session_store(self, name: str) -> Optional[Type]:
        """Get a registered session store by name."""
        return self._session_stores.get(name.lower())
    
    def list_session_stores(self) -> list:
        """List all registered session stores."""
        return list(self._session_stores.keys())
    
    # ==========================================================================
    # Memory Stores
    # ==========================================================================
    
    def register_memory_store(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a memory store."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._memory_stores[name.lower()] = cls
            return cls
        return decorator
    
    def get_memory_store(self, name: str) -> Optional[Type]:
        """Get a registered memory store by name."""
        return self._memory_stores.get(name.lower())
    
    def list_memory_stores(self) -> list:
        """List all registered memory stores."""
        return list(self._memory_stores.keys())

    # ==========================================================================
    # Expertise Stores
    # ==========================================================================

    def register_expertise_store(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register an expertise store."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._expertise_stores[name.lower()] = cls
            return cls
        return decorator

    def get_expertise_store(self, name: str) -> Optional[Type]:
        """Get a registered expertise store by name."""
        return self._expertise_stores.get(name.lower())

    def list_expertise_stores(self) -> list:
        """List all registered expertise stores."""
        return list(self._expertise_stores.keys())

    # ==========================================================================
    # Scoped Memory Stores
    # ==========================================================================

    def register_scoped_memory_store(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a scoped memory store."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._scoped_memory_stores[name.lower()] = cls
            return cls
        return decorator

    def get_scoped_memory_store(self, name: str) -> Optional[Type]:
        """Get a registered scoped memory store by name."""
        return self._scoped_memory_stores.get(name.lower())

    def list_scoped_memory_stores(self) -> list:
        """List all registered scoped memory stores."""
        return list(self._scoped_memory_stores.keys())

    # ==========================================================================
    # Skill Stores
    # ==========================================================================

    def register_skill_store(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a skill store."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._skill_stores[name.lower()] = cls
            return cls
        return decorator

    def get_skill_store(self, name: str) -> Optional[Type]:
        """Get a registered skill store by name."""
        return self._skill_stores.get(name.lower())

    def list_skill_stores(self) -> list:
        """List all registered skill stores."""
        return list(self._skill_stores.keys())

    # ==========================================================================
    # Retrievers
    # ==========================================================================
    
    def register_retriever(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a retriever."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._retrievers[name.lower()] = cls
            return cls
        return decorator
    
    def get_retriever(self, name: str) -> Optional[Type]:
        """Get a registered retriever by name."""
        return self._retrievers.get(name.lower())
    
    def list_retrievers(self) -> list:
        """List all registered retrievers."""
        return list(self._retrievers.keys())
    
    # ==========================================================================
    # Condensers
    # ==========================================================================

    def register_condenser(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a condenser."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._compactors[name.lower()] = cls
            return cls
        return decorator

    def get_condenser(self, name: str) -> Optional[Type]:
        """Get a registered condenser by name."""
        return self._compactors.get(name.lower())

    def list_condensers(self) -> list:
        """List all registered condensers."""
        return list(self._compactors.keys())

    # Back-compat aliases
    def register_compactor(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Deprecated alias for register_condenser()."""
        warnings.warn(
            "register_compactor() is deprecated; use register_condenser()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.register_condenser(name)

    def get_compactor(self, name: str) -> Optional[Type]:
        """Deprecated alias for get_condenser()."""
        warnings.warn(
            "get_compactor() is deprecated; use get_condenser()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_condenser(name)

    def list_compactors(self) -> list:
        """Deprecated alias for list_condensers()."""
        warnings.warn(
            "list_compactors() is deprecated; use list_condensers()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_condensers()
    
    # ==========================================================================
    # Extractors
    # ==========================================================================
    
    def register_extractor(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register an extractor."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._extractors[name.lower()] = cls
            return cls
        return decorator
    
    def get_extractor(self, name: str) -> Optional[Type]:
        """Get a registered extractor by name."""
        return self._extractors.get(name.lower())
    
    def list_extractors(self) -> list:
        """List all registered extractors."""
        return list(self._extractors.keys())
    
    def register_middleware(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register middleware."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._middleware[name.lower()] = cls
            return cls
        return decorator
    
    def get_middleware(self, name: str) -> Optional[Type]:
        """Get registered middleware by name."""
        return self._middleware.get(name.lower())

    def list_middleware(self) -> list:
        """List all registered middleware."""
        return list(self._middleware.keys())

    # ==========================================================================
    # Middleware Factories
    # ==========================================================================

    def register_middleware_factory(self, name: str):
        """
        Register a dependency-aware middleware factory.

        Factories should implement `create(config: dict, deps: EngineDeps) -> IMiddleware | None`.
        """

        def decorator(factory: Any) -> Any:
            self._middleware_factories[name.lower()] = factory
            return factory

        return decorator

    def get_middleware_factory(self, name: str) -> Optional[Any]:
        """Get registered middleware factory by name."""
        return self._middleware_factories.get(name.lower())

    def list_middleware_factories(self) -> list:
        """List all registered middleware factories."""
        return list(self._middleware_factories.keys())
    
    # ==========================================================================
    # Rerankers
    # ==========================================================================
    
    def register_reranker(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a reranker."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._rerankers[name.lower()] = cls
            return cls
        return decorator
    
    def get_reranker(self, name: str) -> Optional[Type]:
        """Get a registered reranker by name."""
        return self._rerankers.get(name.lower())
    
    def list_rerankers(self) -> list:
        """List all registered rerankers."""
        return list(self._rerankers.keys())
    
    # ==========================================================================
    # Assemblers
    # ==========================================================================
    
    def register_assembler(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register an assembler."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._assemblers[name.lower()] = cls
            return cls
        return decorator
    
    def get_assembler(self, name: str) -> Optional[Type]:
        """Get a registered assembler by name."""
        return self._assemblers.get(name.lower())
    
    def list_assemblers(self) -> list:
        """List all registered assemblers."""
        return list(self._assemblers.keys())
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================
    
    def register_component(
        self,
        component_type: str,
        name: str,
        cls: Type = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Register a component programmatically or as a decorator.
        
        Can be used in two ways:
        
        1. As a decorator:
            @registry.register_component("reranker", "my_reranker")
            class MyReranker:
                ...
        
        2. Programmatically:
            registry.register_component("reranker", "my_reranker", MyReranker)
        
        Args:
            component_type: Type of component (llm, session_store, etc.)
            name: Name to register under
            cls: The component class (optional if used as decorator)
        """
        registries = {
            "llm": self._llm_providers,
            "embedding": self._embedding_providers,
            "session_store": self._session_stores,
            "memory_store": self._memory_stores,
            "expertise_store": self._expertise_stores,
            "scoped_memory_store": self._scoped_memory_stores,
            "skill_store": self._skill_stores,
            "retriever": self._retrievers,
            "compactor": self._compactors,   # deprecated alias
            "condenser": self._compactors,
            "extractor": self._extractors,
            "middleware": self._middleware,
            "reranker": self._rerankers,
            "assembler": self._assemblers,
        }
        
        if component_type not in registries:
            raise ValueError(f"Unknown component type: {component_type}")
        
        def decorator(cls_to_register: Type[T]) -> Type[T]:
            registries[component_type][name.lower()] = cls_to_register
            return cls_to_register
        
        # If cls is provided, register immediately and return None
        if cls is not None:
            registries[component_type][name.lower()] = cls
            return None
        
        # Otherwise, return the decorator
        return decorator

    @staticmethod
    def load_class(class_path: str) -> Type:
        """
        Load a class from a string.

        Supports:
        - "pkg.mod:ClassName"
        - "pkg.mod.ClassName"
        """
        if ":" in class_path:
            module_path, class_name = class_path.split(":", 1)
        else:
            module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def register_component_class_path(
        self,
        component_type: str,
        name: str,
        class_path: str,
    ) -> None:
        """Register a component by class path."""
        cls = self.load_class(class_path)
        # register_component supports programmatic registration when cls is provided
        self.register_component(component_type, name, cls)
    
    def get_component(self, component_type: str, name: str) -> Optional[Type]:
        """
        Get a component by type and name.
        
        Args:
            component_type: Type of component
            name: Name of the component
            
        Returns:
            The component class or None if not found
        """
        getters = {
            "llm": self.get_llm,
            "embedding": self.get_embedding,
            "session_store": self.get_session_store,
            "memory_store": self.get_memory_store,
            "expertise_store": self.get_expertise_store,
            "scoped_memory_store": self.get_scoped_memory_store,
            "skill_store": self.get_skill_store,
            "retriever": self.get_retriever,
            "compactor": self.get_compactor,  # deprecated alias
            "condenser": self.get_condenser,
            "extractor": self.get_extractor,
            "middleware": self.get_middleware,
            "reranker": self.get_reranker,
            "assembler": self.get_assembler,
        }
        
        getter = getters.get(component_type)
        if getter:
            return getter(name)
        return None
    
    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._llm_providers.clear()
        self._embedding_providers.clear()
        self._session_stores.clear()
        self._memory_stores.clear()
        self._expertise_stores.clear()
        self._scoped_memory_stores.clear()
        self._skill_stores.clear()
        self._retrievers.clear()
        self._compactors.clear()
        self._extractors.clear()
        self._middleware.clear()
        self._middleware_factories.clear()
        self._rerankers.clear()
        self._assemblers.clear()


# Global registry instance
registry = ComponentRegistry()

