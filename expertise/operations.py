"""
Curation operations helper functions.

Provides utilities for applying curation operations to expertise,
including ID generation, formatting, and operation application.
"""

import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from ctxforge.core.expertise import (
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
)


def generate_item_id(section: ExpertiseSection, next_id: int) -> str:
    """
    Generate a unique item ID for a section.
    
    Args:
        section: The section for the item
        next_id: The next available ID number
        
    Returns:
        Item ID in format: "{section_slug}-{5-digit-id}"
        e.g., "strat-00001", "form-00042"
    """
    slug = section.to_slug()
    return f"{slug}-{next_id:05d}"


def format_item_line(item: ExpertiseItem) -> str:
    """
    Format an expertise item in ACE-style format.
    
    Args:
        item: The expertise item
        
    Returns:
        Formatted string: "[id] helpful=X harmful=Y :: content"
    """
    return (
        f"[{item.item_id}] "
        f"helpful={item.helpful_count} harmful={item.harmful_count} :: "
        f"{item.content}"
    )


def parse_item_line(line: str, section: ExpertiseSection = ExpertiseSection.CUSTOM) -> Optional[ExpertiseItem]:
    """
    Parse an ACE-style line into an ExpertiseItem.
    
    Args:
        line: String in format: "[id] helpful=X harmful=Y :: content"
        section: Section to assign to the item
        
    Returns:
        ExpertiseItem or None if parsing fails
    """
    pattern = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern, line.strip())
    
    if not match:
        return None
    
    item_id, helpful, harmful, content = match.groups()
    
    return ExpertiseItem(
        item_id=item_id,
        section=section,
        content=content.strip(),
        helpful_count=int(helpful),
        harmful_count=int(harmful),
    )


def apply_add_operation(
    expertise: Expertise,
    op: CurationOp,
) -> Optional[ExpertiseItem]:
    """
    Apply an ADD operation to expertise.
    
    Args:
        expertise: The expertise to modify
        op: The ADD operation
        
    Returns:
        The newly created item, or None if failed
    """
    if not op.content:
        return None
    
    section = op.section or ExpertiseSection.CUSTOM
    
    # Use expertise's add_item method
    item = expertise.add_item(
        section=section,
        content=op.content,
        source="curator",
    )
    
    return item


def apply_update_operation(
    expertise: Expertise,
    op: CurationOp,
) -> bool:
    """
    Apply an UPDATE operation to expertise.
    
    Args:
        expertise: The expertise to modify
        op: The UPDATE operation
        
    Returns:
        True if update was successful, False otherwise
    """
    if not op.item_ids or not op.content:
        return False
    
    item_id = op.item_ids[0]
    item = expertise.get_item(item_id)
    
    if not item:
        return False
    
    item.update_content(op.content)
    expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    return True


def apply_merge_operation(
    expertise: Expertise,
    op: CurationOp,
) -> Optional[ExpertiseItem]:
    """
    Apply a MERGE operation to expertise.
    
    Merges multiple items into one, preserving usage counts.
    
    Args:
        expertise: The expertise to modify
        op: The MERGE operation
        
    Returns:
        The merged item, or None if failed
    """
    if len(op.item_ids) < 2 or not op.content:
        return None
    
    # Collect items to merge
    items_to_merge = []
    for item_id in op.item_ids:
        item = expertise.get_item(item_id)
        if item:
            items_to_merge.append(item)
    
    if len(items_to_merge) < 2:
        return None
    
    # Sum up usage counts
    total_helpful = sum(item.helpful_count for item in items_to_merge)
    total_harmful = sum(item.harmful_count for item in items_to_merge)
    
    # Use section from first item
    section = items_to_merge[0].section
    
    # Create merged item
    merged_item = expertise.add_item(
        section=section,
        content=op.content,
        source="curator:merge",
    )
    merged_item.helpful_count = total_helpful
    merged_item.harmful_count = total_harmful
    
    # Deactivate original items
    for item in items_to_merge:
        item.deactivate()
    
    return merged_item


def apply_delete_operation(
    expertise: Expertise,
    op: CurationOp,
) -> bool:
    """
    Apply a DELETE operation to expertise.
    
    Args:
        expertise: The expertise to modify
        op: The DELETE operation
        
    Returns:
        True if deletion was successful, False otherwise
    """
    if not op.item_ids:
        return False
    
    item_id = op.item_ids[0]
    
    # Soft delete (deactivate)
    return expertise.remove_item(item_id, soft_delete=True)


def apply_curation_plan(
    expertise: Expertise,
    plan: CurationPlan,
) -> Tuple[Expertise, int, int, int, int]:
    """
    Apply a complete curation plan to expertise.
    
    Args:
        expertise: The expertise to modify
        plan: The curation plan to apply
        
    Returns:
        Tuple of (updated expertise, adds, updates, merges, deletes)
    """
    adds = 0
    updates = 0
    merges = 0
    deletes = 0
    
    for op in plan.operations:
        if op.type == CuratorOperation.ADD:
            if apply_add_operation(expertise, op):
                adds += 1
                
        elif op.type == CuratorOperation.UPDATE:
            if apply_update_operation(expertise, op):
                updates += 1
                
        elif op.type == CuratorOperation.MERGE:
            if apply_merge_operation(expertise, op):
                merges += 1
                
        elif op.type == CuratorOperation.DELETE:
            if apply_delete_operation(expertise, op):
                deletes += 1
    
    # Increment version if any changes were made
    if adds + updates + merges + deletes > 0:
        expertise.increment_version()
    
    return expertise, adds, updates, merges, deletes


def validate_operation(op: CurationOp) -> Tuple[bool, str]:
    """
    Validate a curation operation.
    
    Args:
        op: The operation to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if op.type == CuratorOperation.ADD:
        if not op.content:
            return False, "ADD operation requires content"
        if not op.section:
            return False, "ADD operation requires section"
            
    elif op.type == CuratorOperation.UPDATE:
        if not op.item_ids:
            return False, "UPDATE operation requires item_id"
        if not op.content:
            return False, "UPDATE operation requires content"
            
    elif op.type == CuratorOperation.MERGE:
        if not op.item_ids or len(op.item_ids) < 2:
            return False, "MERGE operation requires at least 2 item_ids"
        if not op.content:
            return False, "MERGE operation requires merged content"
            
    elif op.type == CuratorOperation.DELETE:
        if not op.item_ids:
            return False, "DELETE operation requires item_id"
    
    return True, ""


def format_expertise_for_prompt(expertise: Expertise) -> str:
    """
    Format expertise content for inclusion in prompts.
    
    Args:
        expertise: The expertise to format
        
    Returns:
        Formatted string with all items organized by section
    """
    return expertise.to_ace_format()


def parse_section_from_string(section_str: str) -> ExpertiseSection:
    """
    Parse a section string to ExpertiseSection enum.
    
    Args:
        section_str: Section name string
        
    Returns:
        ExpertiseSection enum value
    """
    return ExpertiseSection.from_string(section_str)

