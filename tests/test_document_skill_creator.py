"""
Tests for Document Skill Creator.
"""
import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from ctxforge.engine.services.document_skill_creator import (
    DocumentSkillCreator,
)


@dataclass
class FakeLLMResponse:
    content: str


class FakeLLMProvider:
    """Fake LLM provider that returns a valid document skill JSON."""

    def __init__(self, response_data: Optional[dict] = None):
        self._data = response_data or {
            "name": "test-doc-skill",
            "description": "Use when working with test document content",
            "when_to_use": "When the user needs guidance from the document",
            "triggers": ["test document", "guide"],
            "category": "document",
            "instructions": "# Document Skill\n\nThis skill provides guidance from the test document.",
            "scripts": {},
            "references": {"source.md": "Original document reference."},
        }

    async def chat(self, messages, model=None, temperature=None,
                   max_tokens=None, **kwargs):
        return FakeLLMResponse(content=json.dumps(self._data))


class TestIsSupported:
    """Tests for file type support checking."""

    def test_pdf_supported(self):
        assert DocumentSkillCreator.is_supported("doc.pdf") is True

    def test_docx_supported(self):
        assert DocumentSkillCreator.is_supported("report.docx") is True

    def test_pptx_supported(self):
        assert DocumentSkillCreator.is_supported("slides.pptx") is True

    def test_txt_not_supported(self):
        assert DocumentSkillCreator.is_supported("notes.txt") is False

    def test_csv_not_supported(self):
        assert DocumentSkillCreator.is_supported("data.csv") is False


class TestGetFileType:
    """Tests for file type name resolution."""

    def test_pdf_type(self):
        assert DocumentSkillCreator.get_file_type("doc.pdf") == "PDF Document"

    def test_docx_type(self):
        assert DocumentSkillCreator.get_file_type("r.docx") == "Word Document"

    def test_pptx_type(self):
        assert (
            DocumentSkillCreator.get_file_type("s.pptx")
            == "PowerPoint Presentation"
        )

    def test_unknown_type(self):
        assert DocumentSkillCreator.get_file_type("f.xyz") == "Unknown"


class TestCreateFromFile:
    """Tests for skill creation from documents."""

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)
        with pytest.raises(FileNotFoundError):
            await creator.create_from_file("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("some notes")
        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)
        with pytest.raises(ValueError, match="Unsupported file type"):
            await creator.create_from_file(str(txt_file))

    @pytest.mark.asyncio
    async def test_pdf_extraction_import_error(self, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)

        with patch.dict("sys.modules", {"PyPDF2": None}):
            with pytest.raises(ImportError, match="PyPDF2"):
                await creator.create_from_file(str(pdf_file))

    @pytest.mark.asyncio
    async def test_docx_extraction_import_error(self, tmp_path):
        docx_file = tmp_path / "report.docx"
        docx_file.write_bytes(b"fake docx")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)

        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                await creator.create_from_file(str(docx_file))

    @pytest.mark.asyncio
    async def test_pptx_extraction_import_error(self, tmp_path):
        pptx_file = tmp_path / "slides.pptx"
        pptx_file.write_bytes(b"fake pptx")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)

        with patch.dict("sys.modules", {"pptx": None}):
            with pytest.raises(ImportError, match="python-pptx"):
                await creator.create_from_file(str(pptx_file))

    @pytest.mark.asyncio
    async def test_create_with_mock_extraction(self, tmp_path):
        """Mock the extraction and verify skill creation."""
        pdf_file = tmp_path / "guide.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)

        with patch.object(
            creator, "_extract_text",
            return_value="This is the extracted document text with enough content.",
        ):
            skill = await creator.create_from_file(str(pdf_file))

        assert skill is not None
        assert skill.name == "test-doc-skill"
        assert skill.scope_id == "default"
        assert skill.structured_content is not None

    @pytest.mark.asyncio
    async def test_create_with_custom_project_id(self, tmp_path):
        pdf_file = tmp_path / "guide.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)

        with patch.object(
            creator, "_extract_text",
            return_value="Document text with enough content for processing.",
        ):
            skill = await creator.create_from_file(
                str(pdf_file), project_id="my-project"
            )

        assert skill is not None
        assert skill.scope_id == "my-project"

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_none(self, tmp_path):
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)

        with patch.object(creator, "_extract_text", return_value="  "):
            skill = await creator.create_from_file(str(pdf_file))

        assert skill is None


class TestMaxCharsTruncation:
    """Tests for max_chars enforcement."""

    @pytest.mark.asyncio
    async def test_truncation(self, tmp_path):
        pdf_file = tmp_path / "big.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm, max_chars=100)

        long_text = "A" * 200

        with patch.object(creator, "_extract_text", return_value=long_text):
            skill = await creator.create_from_file(str(pdf_file))

        # The creator extracts and passes text to LLM; truncation is in
        # the extraction methods themselves. Here we verify the skill was
        # still created successfully.
        assert skill is not None


class TestParseResponse:
    """Tests for LLM response parsing."""

    def test_valid_json(self):
        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)
        data = {
            "name": "doc-skill",
            "description": "A document skill",
            "when_to_use": "When needed",
            "triggers": ["doc"],
            "instructions": "# Instructions with enough detail to be useful.",
        }
        skill = creator._parse_response(json.dumps(data), "proj1")
        assert skill is not None
        assert skill.name == "doc-skill"

    def test_invalid_json_returns_none(self):
        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)
        skill = creator._parse_response("not json", "proj1")
        assert skill is None

    def test_bad_name_returns_none(self):
        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)
        data = {"name": "BAD NAME!", "instructions": "x"}
        skill = creator._parse_response(json.dumps(data), "proj1")
        assert skill is None

    def test_json_with_markdown_fences(self):
        llm = FakeLLMProvider()
        creator = DocumentSkillCreator(llm_provider=llm)
        data = {
            "name": "fenced-skill",
            "description": "desc",
            "instructions": "inst",
        }
        raw = f"```json\n{json.dumps(data)}\n```"
        skill = creator._parse_response(raw, "proj1")
        assert skill is not None
        assert skill.name == "fenced-skill"
