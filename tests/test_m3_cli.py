import json
from pathlib import Path

from ctxforge.cli import cmd_list_components, cmd_validate_config
from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.engine.factory import EngineFactory


def test_validate_config_requires_pinecone_api_key():
    cfg = TESTING_CONFIG.merge_with(
        {
            "storage": {
                "memory": {
                    "backend": "pinecone",
                    "extra_params": {},
                }
            }
        }
    )
    res = EngineFactory().validate_config(cfg)
    assert res.ok is False
    assert any("storage.memory.vector.extra_params.api_key" in e for e in res.errors)


def test_cli_list_components_json(capsys):
    # Ensure mock providers are registered
    import ctxforge.llm.mock_provider  # noqa: F401

    code = cmd_list_components(as_json=True)
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert "llm_providers" in payload


def test_cli_validate_config_examples_engine_config():
    example_path = Path(__file__).parent.parent / "examples" / "engine_config.yaml"
    code = cmd_validate_config(path=str(example_path), as_json=True)
    assert code == 0


