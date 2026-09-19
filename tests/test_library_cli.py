import io
import json

from paper_rag.cli.main import main


def test_module_arguments_and_group_commands(monkeypatch, tmp_path, capsys):
    from paper_rag.library.catalog import Catalog
    from paper_rag.library.settings import LibrarySettings

    with Catalog(LibrarySettings.load(tmp_path), writable=True):
        pass
    monkeypatch.setattr(
        "sys.argv",
        ["paper-rag", "--project-root", str(tmp_path), "papers", "count", "--json"],
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["data"]["count"] == 0


def test_old_answer_command_has_no_network_path(capsys):
    assert main(["ask", "question"]) == 2
    assert (
        json.loads(capsys.readouterr().out)["warnings"][0]["code"]
        == "host_agent_required"
    )


def test_malformed_request_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{broken"))
    assert main(["search", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_doctor_on_empty_workspace(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert main(["--project-root", str(tmp_path), "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_initialized"
