"""
Expertise Snapshot Service.

Provides versioning and diffing capabilities for expertise domains.
Enables tracking changes over time and identifying what knowledge
was added, removed, or modified between versions.

Use cases:
- Track expertise growth over time
- Review changes before deploying to production
- Rollback to previous states if issues arise
- Generate changelogs for knowledge bases
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ctxforge.core.expertise import Expertise, ExpertiseItem

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of changes in a diff."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class ItemChange(BaseModel):
    """A change to an individual expertise item."""
    change_type: ChangeType
    item_id: str
    section: str
    content: str
    old_content: Optional[str] = None  # For modified items
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExpertiseDiff(BaseModel):
    """
    Difference between two expertise snapshots.
    
    Provides a detailed breakdown of what changed between versions.
    """
    from_version: str
    to_version: str
    from_timestamp: Optional[datetime] = None
    to_timestamp: Optional[datetime] = None
    
    # Summary counts
    items_added: int = 0
    items_removed: int = 0
    items_modified: int = 0
    items_unchanged: int = 0
    
    # Detailed changes
    changes: List[ItemChange] = Field(default_factory=list)
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.items_added > 0 or self.items_removed > 0 or self.items_modified > 0
    
    def to_summary(self) -> str:
        """Generate a human-readable summary of changes."""
        if not self.has_changes:
            return f"No changes between {self.from_version} and {self.to_version}"
        
        parts = []
        if self.items_added > 0:
            parts.append(f"+{self.items_added} added")
        if self.items_removed > 0:
            parts.append(f"-{self.items_removed} removed")
        if self.items_modified > 0:
            parts.append(f"~{self.items_modified} modified")
        
        return f"Changes from {self.from_version} to {self.to_version}: {', '.join(parts)}"
    
    def to_changelog(self) -> str:
        """Generate a changelog-style output."""
        lines = [
            f"## Changelog: {self.from_version} → {self.to_version}",
            "",
        ]
        
        if self.to_timestamp:
            lines.append(f"**Date**: {self.to_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append("")
        
        # Group changes by section
        by_section: Dict[str, List[ItemChange]] = {}
        for change in self.changes:
            if change.change_type != ChangeType.UNCHANGED:
                if change.section not in by_section:
                    by_section[change.section] = []
                by_section[change.section].append(change)
        
        for section, changes in by_section.items():
            lines.append(f"### {section}")
            for change in changes:
                prefix = {
                    ChangeType.ADDED: "+",
                    ChangeType.REMOVED: "-",
                    ChangeType.MODIFIED: "~",
                }.get(change.change_type, "")
                
                content_preview = change.content[:100]
                if len(change.content) > 100:
                    content_preview += "..."
                
                lines.append(f"- {prefix} {content_preview}")
            lines.append("")
        
        return "\n".join(lines)


class ExpertiseSnapshot(BaseModel):
    """
    A point-in-time snapshot of an expertise domain.
    
    Captures the complete state of expertise items for versioning.
    """
    snapshot_id: str = Field(default_factory=lambda: "")
    expertise_id: str
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    description: str = ""
    
    # Serialized items
    items: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Content hash for quick comparison
    content_hash: str = ""
    
    @classmethod
    def from_expertise(
        cls,
        expertise: Expertise,
        version: str,
        created_by: Optional[str] = None,
        description: str = "",
    ) -> "ExpertiseSnapshot":
        """
        Create a snapshot from an Expertise object.
        
        Args:
            expertise: The expertise to snapshot
            version: Version identifier
            created_by: Who created this snapshot
            description: Optional description
            
        Returns:
            A new ExpertiseSnapshot
        """
        items = [item.model_dump() for item in expertise.items]
        
        # Calculate content hash
        content_str = json.dumps(items, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        snapshot_id = f"{expertise.expertise_id}:{version}:{content_hash}"
        
        return cls(
            snapshot_id=snapshot_id,
            expertise_id=expertise.expertise_id,
            version=version,
            created_by=created_by,
            description=description,
            items=items,
            content_hash=content_hash,
        )
    
    def to_expertise(self) -> Expertise:
        """
        Reconstruct an Expertise object from this snapshot.
        
        Returns:
            Expertise object with the snapshot's items
        """
        expertise = Expertise(
            expertise_id=self.expertise_id,
            name=f"Snapshot {self.version}",
            domain="restored",
        )
        for item_dict in self.items:
            item = ExpertiseItem.model_validate(item_dict)
            expertise.items.append(item)
        
        return expertise


class ExpertiseSnapshotStore:
    """
    Abstract store for expertise snapshots.
    
    Implementations should persist snapshots for version history.
    """
    
    async def save_snapshot(self, snapshot: ExpertiseSnapshot) -> None:
        """Save a snapshot."""
        raise NotImplementedError
    
    async def get_snapshot(
        self, expertise_id: str, version: str
    ) -> Optional[ExpertiseSnapshot]:
        """Get a specific snapshot by expertise ID and version."""
        raise NotImplementedError
    
    async def get_latest_snapshot(self, expertise_id: str) -> Optional[ExpertiseSnapshot]:
        """Get the most recent snapshot for an expertise."""
        raise NotImplementedError
    
    async def list_versions(self, expertise_id: str) -> List[str]:
        """List all versions for an expertise."""
        raise NotImplementedError


class InMemorySnapshotStore(ExpertiseSnapshotStore):
    """In-memory implementation for development/testing."""
    
    def __init__(self):
        self._snapshots: Dict[str, List[ExpertiseSnapshot]] = {}
    
    async def save_snapshot(self, snapshot: ExpertiseSnapshot) -> None:
        if snapshot.expertise_id not in self._snapshots:
            self._snapshots[snapshot.expertise_id] = []
        self._snapshots[snapshot.expertise_id].append(snapshot)
    
    async def get_snapshot(
        self, expertise_id: str, version: str
    ) -> Optional[ExpertiseSnapshot]:
        snapshots = self._snapshots.get(expertise_id, [])
        for s in snapshots:
            if s.version == version:
                return s
        return None
    
    async def get_latest_snapshot(self, expertise_id: str) -> Optional[ExpertiseSnapshot]:
        snapshots = self._snapshots.get(expertise_id, [])
        if not snapshots:
            return None
        return max(snapshots, key=lambda s: s.created_at)
    
    async def list_versions(self, expertise_id: str) -> List[str]:
        snapshots = self._snapshots.get(expertise_id, [])
        return [s.version for s in sorted(snapshots, key=lambda s: s.created_at)]


class FileBasedSnapshotStore(ExpertiseSnapshotStore):
    """
    File-based snapshot store.

    Stores snapshots as JSON files organized by expertise ID.
    Useful for backup, export, and version control integration.

    Directory structure:
    ```
    snapshots_dir/
    ├── expertise-abc/
    │   ├── v1.0.0.json
    │   ├── v1.1.0.json
    │   └── latest.json -> v1.1.0.json (symlink or copy)
    └── expertise-xyz/
        └── v1.0.0.json
    ```
    """

    def __init__(self, snapshots_dir: str, use_symlinks: bool = False):
        """
        Initialize the file-based store.

        Args:
            snapshots_dir: Directory to store snapshot files
            use_symlinks: If True, use symlinks for latest; else copy
        """
        self._snapshots_dir = snapshots_dir
        self._use_symlinks = use_symlinks

    def _get_expertise_dir(self, expertise_id: str) -> str:
        """Get the directory for an expertise's snapshots."""
        import os
        # Sanitize expertise_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in expertise_id)
        return os.path.join(self._snapshots_dir, safe_id)

    def _get_snapshot_path(self, expertise_id: str, version: str) -> str:
        """Get the path for a specific snapshot file."""
        import os
        safe_version = "".join(c if c.isalnum() or c in "-_." else "_" for c in version)
        return os.path.join(self._get_expertise_dir(expertise_id), f"{safe_version}.json")

    async def save_snapshot(self, snapshot: ExpertiseSnapshot) -> None:
        """Save a snapshot to a JSON file."""
        import os

        expertise_dir = self._get_expertise_dir(snapshot.expertise_id)
        os.makedirs(expertise_dir, exist_ok=True)

        snapshot_path = self._get_snapshot_path(snapshot.expertise_id, snapshot.version)

        # Serialize the snapshot
        data = snapshot.model_dump(mode="json")

        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        # Update latest pointer
        latest_path = os.path.join(expertise_dir, "latest.json")
        if self._use_symlinks:
            # Use symlink
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(snapshot_path), latest_path)
        else:
            # Copy the file
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

    async def get_snapshot(
        self, expertise_id: str, version: str
    ) -> Optional[ExpertiseSnapshot]:
        """Load a snapshot from file."""
        import os

        snapshot_path = self._get_snapshot_path(expertise_id, version)

        if not os.path.exists(snapshot_path):
            return None

        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ExpertiseSnapshot.model_validate(data)
        except Exception:
            return None

    async def get_latest_snapshot(self, expertise_id: str) -> Optional[ExpertiseSnapshot]:
        """Get the most recent snapshot."""
        import os

        latest_path = os.path.join(self._get_expertise_dir(expertise_id), "latest.json")

        if not os.path.exists(latest_path):
            return None

        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ExpertiseSnapshot.model_validate(data)
        except Exception:
            return None

    async def list_versions(self, expertise_id: str) -> List[str]:
        """List all versions for an expertise."""
        import os

        expertise_dir = self._get_expertise_dir(expertise_id)

        if not os.path.exists(expertise_dir):
            return []

        versions = []
        for filename in os.listdir(expertise_dir):
            if filename.endswith('.json') and filename != 'latest.json':
                version = filename[:-5]  # Remove .json
                versions.append(version)

        # Sort by modification time
        versions.sort(key=lambda v: os.path.getmtime(
            self._get_snapshot_path(expertise_id, v)
        ))

        return versions

    async def delete_snapshot(self, expertise_id: str, version: str) -> bool:
        """Delete a specific snapshot."""
        import os

        snapshot_path = self._get_snapshot_path(expertise_id, version)

        if not os.path.exists(snapshot_path):
            return False

        os.remove(snapshot_path)
        return True


class ExpertiseSnapshotService:
    """
    Service for managing expertise snapshots and diffs.
    
    Example usage:
    ```python
    service = ExpertiseSnapshotService(store=InMemorySnapshotStore())
    
    # Create a snapshot
    await service.create_snapshot(expertise, version="1.0.0")
    
    # Later, compare versions
    diff = await service.diff_versions(expertise_id, "1.0.0", "1.1.0")
    print(diff.to_changelog())
    ```
    """
    
    def __init__(self, store: ExpertiseSnapshotStore):
        """
        Initialize the service.
        
        Args:
            store: Store for persisting snapshots
        """
        self._store = store
    
    async def create_snapshot(
        self,
        expertise: Expertise,
        version: str,
        created_by: Optional[str] = None,
        description: str = "",
    ) -> ExpertiseSnapshot:
        """
        Create and save a snapshot of the current expertise state.
        
        Args:
            expertise: The expertise to snapshot
            version: Version identifier (e.g., "1.0.0", "2024-01-15")
            created_by: Who created this snapshot
            description: Optional description of this version
            
        Returns:
            The created snapshot
        """
        snapshot = ExpertiseSnapshot.from_expertise(
            expertise=expertise,
            version=version,
            created_by=created_by,
            description=description,
        )
        
        await self._store.save_snapshot(snapshot)
        logger.info(f"Created snapshot {snapshot.snapshot_id}")
        
        return snapshot
    
    async def get_snapshot(
        self, expertise_id: str, version: str
    ) -> Optional[ExpertiseSnapshot]:
        """Get a specific snapshot."""
        return await self._store.get_snapshot(expertise_id, version)
    
    async def get_latest(self, expertise_id: str) -> Optional[ExpertiseSnapshot]:
        """Get the latest snapshot for an expertise."""
        return await self._store.get_latest_snapshot(expertise_id)
    
    async def list_versions(self, expertise_id: str) -> List[str]:
        """List all versions for an expertise."""
        return await self._store.list_versions(expertise_id)
    
    def diff_snapshots(
        self, from_snapshot: ExpertiseSnapshot, to_snapshot: ExpertiseSnapshot
    ) -> ExpertiseDiff:
        """
        Calculate the difference between two snapshots.
        
        Args:
            from_snapshot: The earlier snapshot
            to_snapshot: The later snapshot
            
        Returns:
            ExpertiseDiff with detailed changes
        """
        # Build maps for comparison
        from_items = {item.get("item_id"): item for item in from_snapshot.items}
        to_items = {item.get("item_id"): item for item in to_snapshot.items}
        
        changes = []
        added = 0
        removed = 0
        modified = 0
        unchanged = 0
        
        # Check for added and modified items
        for item_id, item in to_items.items():
            if item_id not in from_items:
                # Added
                changes.append(ItemChange(
                    change_type=ChangeType.ADDED,
                    item_id=item_id,
                    section=item.get("section", "unknown"),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                ))
                added += 1
            else:
                # Potentially modified
                old_item = from_items[item_id]
                if item.get("content") != old_item.get("content"):
                    changes.append(ItemChange(
                        change_type=ChangeType.MODIFIED,
                        item_id=item_id,
                        section=item.get("section", "unknown"),
                        content=item.get("content", ""),
                        old_content=old_item.get("content", ""),
                        metadata=item.get("metadata", {}),
                    ))
                    modified += 1
                else:
                    unchanged += 1
        
        # Check for removed items
        for item_id, item in from_items.items():
            if item_id not in to_items:
                changes.append(ItemChange(
                    change_type=ChangeType.REMOVED,
                    item_id=item_id,
                    section=item.get("section", "unknown"),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                ))
                removed += 1
        
        return ExpertiseDiff(
            from_version=from_snapshot.version,
            to_version=to_snapshot.version,
            from_timestamp=from_snapshot.created_at,
            to_timestamp=to_snapshot.created_at,
            items_added=added,
            items_removed=removed,
            items_modified=modified,
            items_unchanged=unchanged,
            changes=changes,
        )
    
    async def diff_versions(
        self, expertise_id: str, from_version: str, to_version: str
    ) -> Optional[ExpertiseDiff]:
        """
        Calculate the difference between two versions.
        
        Args:
            expertise_id: The expertise ID
            from_version: Earlier version
            to_version: Later version
            
        Returns:
            ExpertiseDiff if both versions exist, None otherwise
        """
        from_snap = await self._store.get_snapshot(expertise_id, from_version)
        to_snap = await self._store.get_snapshot(expertise_id, to_version)
        
        if not from_snap or not to_snap:
            return None
        
        return self.diff_snapshots(from_snap, to_snap)
    
    async def diff_with_current(
        self, expertise: Expertise, version: str
    ) -> Optional[ExpertiseDiff]:
        """
        Compare current expertise state with a previous version.
        
        Args:
            expertise: Current expertise state
            version: Version to compare against
            
        Returns:
            ExpertiseDiff if version exists, None otherwise
        """
        old_snap = await self._store.get_snapshot(expertise.expertise_id, version)
        if not old_snap:
            return None
        
        # Create a temporary snapshot for current state
        current_snap = ExpertiseSnapshot.from_expertise(
            expertise=expertise,
            version="current",
        )
        
        return self.diff_snapshots(old_snap, current_snap)
