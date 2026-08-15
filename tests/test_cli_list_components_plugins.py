import json

from ctxforge.cli import main
from ctxforge.engine.registry import ComponentRegistry


def test_list_components_can_load_plugins_from_config(tmp_path, monkeypatch, capsys):
    """
    Contract test: if a config specifies plugin modules, `ctxforge list-components`
    should reflect plugin-registered components.
    """
    # Isolate this test from the global registry to avoid leaking registrations to other tests.
    monkeypatch.setattr("ctxforge.cli.registry", ComponentRegistry())

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "plugins:",
                "  modules:",
                "    - ctxforge.tests.plugin_fixtures.plugin_module",
                "",
            ]
        )
    )

    exit_code = main(["list-components", "--json", "--config", str(cfg_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    # The plugin registers a reranker named "plugin_test".
    assert "plugin_test" in data["rerankers"]

