#!/usr/bin/env python3
"""Add exported project chats to local Codex without loading a model or a thread.

Run with the laptop .venv Python. Default is a plan; --apply writes new legacy
rollouts, lets Codex index/read them, and assigns native projects via app-server.
Original exports, existing chats, credentials, and model settings are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import re
import shutil
import sqlite3
import subprocess
import threading
import uuid


ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT.parent / "work_context"
HOME = Path.home() / ".codex"
MANIFEST = CONTEXT / "codex_project_chat_import.json"
BACKUP = CONTEXT / "codex_import_backup" / "20261022T031322Z"
HEADER = re.compile(r"^## (User|Assistant) — ([^\n]+)\n", re.MULTILINE)
CLAUDE_TITLES = {
    "04a93a27-83a5-4bd2-9dfc-3c7df4a30ba9": "GeoFF triggers, directions and pretraining design",
    "0a94a8ab-34b1-4520-8b4b-dea7bf79f3d0": "GPT-2 pretraining runs and cluster analyses",
    "4054cb64-7e30-4a61-bd08-033f328422d4": "Pretraining status continuation",
    "373f5aef-102d-48f5-b7b8-f434425640cf": "DGX Spark compute comparison",
}


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read_messages(session):
    source = CONTEXT / "readable_chats" / session["file"]
    text = source.read_text()
    headers = list(HEADER.finditer(text))
    messages = []
    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        messages.append({"role": match[1].lower(), "timestamp": match[2],
                         "text": text[match.end():end].strip("\n")})
    counts = Counter(m["role"] for m in messages)
    assert len(messages) == session["messages"], source
    assert counts["user"] == session["user_messages"], source
    assert counts["assistant"] == session["assistant_messages"], source
    return source, messages


def plan():
    report = json.loads((CONTEXT / "chat-export-report.json").read_text())
    index = json.loads((CONTEXT / "codex-thread-index.json").read_text())
    codex = {t["thread_id"]: t for t in index["threads"] if t["kind"] == "root"}
    refresh_path = CONTEXT / "jetstream_visible_chat_refresh.json"
    refresh = json.loads(refresh_path.read_text()) if refresh_path.exists() else {}
    result = []
    for session in report["sessions"]:
        if session["kind"] != "root":
            continue
        tid = session["owner"]
        source, messages = read_messages(session)
        if session["provider"] == "codex":
            original = codex[tid]
            raw = CONTEXT / "codex_sessions" / original["rollout_relative_path"].removeprefix("sessions/")
            meta = json.loads(raw.open().readline())["payload"]
            project = Path(meta["cwd"]).name
            title = original["title"]
        else:
            project = "GeoPretrain"
            title = CLAUDE_TITLES[tid]
        assert project in ("GeoAE", "GeoPretrain")
        cutoff = max(instant(m["timestamp"]) for m in messages)
        added = []
        for m in refresh.get("threads", {}).get(tid, {}).get("messages", []):
            if instant(m["timestamp"]) > cutoff:
                added.append({k: m[k] for k in ("role", "timestamp", "text")})
        added.sort(key=lambda m: instant(m["timestamp"]))
        messages.extend(added)
        created = min([instant(session["date"])] + [instant(m["timestamp"]) for m in messages])
        target = ROOT.parent / project
        filename = f"rollout-{created.strftime('%Y-%m-%dT%H-%M-%S')}-{tid}.jsonl"
        destination = HOME / "sessions" / created.strftime("%Y/%m/%d") / filename
        display_title = f"{title} [{session['provider'].capitalize()} archive]"
        if tid == "01a1521b-38f5-7fd1-a55a-27351f0c7f39":
            display_title = "Nonlinear GeoAE experiments"
        result.append({"id": tid, "provider": session["provider"], "project": project,
                       "cwd": str(target), "title": display_title,
                       "source": str(source), "source_sha256": digest(source.read_text()),
                       "path": str(destination), "created": created.isoformat(),
                       "messages": messages, "newer_jetstream_messages": len(added)})
    assert len(result) == 8, "Expected four Codex and four Claude main conversations"
    return result


def rollout(entry):
    lines = []
    def emit(ts, kind, payload):
        lines.append(json.dumps({"timestamp": ts, "type": kind, "payload": payload}))
    emit(entry["created"], "session_meta", {
        "id": entry["id"], "timestamp": entry["created"], "cwd": entry["cwd"],
        "originator": "codex_cli_rs", "cli_version": "0.162.0-alpha.17.2",
        "source": "cli", "model_provider": "openai", "history_mode": "legacy",
    })
    groups = []
    for message in entry["messages"]:
        if not groups or message["role"] == "user":
            groups.append([])
        groups[-1].append(message)
    for number, group in enumerate(groups):
        turn_id = str(uuid.uuid5(uuid.UUID(entry["id"]), f"exported-turn-{number}"))
        emit(group[0]["timestamp"], "event_msg", {"type": "task_started", "turn_id": turn_id})
        final = max((i for i, m in enumerate(group) if m["role"] == "assistant"), default=-1)
        last = ""
        for i, m in enumerate(group):
            ts, body = m["timestamp"], m["text"]
            if m["role"] == "user":
                emit(ts, "event_msg", {"type": "user_message", "message": body,
                                      "images": [], "local_images": [], "text_elements": []})
                emit(ts, "response_item", {"type": "message", "role": "user",
                                           "content": [{"type": "input_text", "text": body}]})
            else:
                phase = "final_answer" if i == final else "commentary"
                emit(ts, "response_item", {"type": "message", "role": "assistant", "phase": phase,
                                           "content": [{"type": "output_text", "text": body}]})
                emit(ts, "event_msg", {"type": "agent_message", "message": body, "phase": phase})
                last = body
        emit(group[-1]["timestamp"], "event_msg", {
            "type": "task_complete", "turn_id": turn_id, "last_agent_message": last})
    return "\n".join(lines) + "\n"


class Server:
    def __init__(self, log):
        binary = shutil.which("codex")
        assert binary, "Codex CLI is not on PATH"
        assert Path("/usr/bin/sandbox-exec").exists(), "This importer targets macOS"
        self.log = log.open("w")
        log.chmod(0o600)
        self.process = subprocess.Popen([
            "/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
            binary, "app-server", "--stdio"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.messages = queue.Queue()
        def drain():
            for line in self.process.stdout:
                try: self.messages.put(json.loads(line))
                except ValueError: continue
            self.messages.put(None)
        threading.Thread(target=drain, daemon=True).start()
        self.counter = 0
        self.call("initialize", {"clientInfo": {"name": "project_chat_history_import", "version": "1"},
                                 "capabilities": {"experimentalApi": True}})
        self.send({"method": "initialized", "params": {}})

    def send(self, item):
        self.process.stdin.write(json.dumps(item) + "\n")
        self.process.stdin.flush()

    def call(self, method, params):
        self.counter += 1
        number = self.counter
        self.send({"id": number, "method": method, "params": params})
        while True:
            reply = self.messages.get(timeout=45)
            if reply is None:
                raise RuntimeError("Codex app-server exited; inspect the private import log")
            if reply.get("id") == number and ("result" in reply or "error" in reply):
                if "error" in reply: raise RuntimeError(f"{method}: {reply['error']}")
                return reply["result"]
            if "id" in reply and "method" in reply:
                self.send({"id": reply["id"], "error": {"code": -32601,
                           "message": "History importer does not execute tools or model turns"}})

    def threads(self, **filters):
        result, cursor = [], None
        while True:
            response = self.call("thread/list", {"limit": 100, "modelProviders": [],
                                                  **filters, "cursor": cursor})
            result.extend(response["data"])
            cursor = response.get("nextCursor")
            if cursor is None: return result

    def close(self):
        self.process.stdin.close()
        try: self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()


def visible(thread):
    result = []
    for turn in thread["turns"]:
        for item in turn["items"]:
            if item["type"] == "userMessage":
                body = "\n".join(c["text"] for c in item["content"] if c["type"] == "text")
                result.append(("user", body))
            elif item["type"] == "agentMessage":
                result.append(("assistant", item["text"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    entries = plan()
    for e in entries:
        print(f"{e['project']}: {e['title']} — {len(e['messages'])} messages, "
              f"{e['newer_jetstream_messages']} newer from Jetstream")
    if not args.apply:
        print("Plan only. Use --apply to import.")
        return
    assert (BACKUP / "state_5.sqlite").exists() and (BACKUP / "session_index.jsonl").exists(), \
        "Back up the existing Codex history before applying"
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"threads": {}}
    db = sqlite3.connect(f"file:{HOME / 'state_5.sqlite'}?mode=ro", uri=True)
    for e in entries:
        known = previous["threads"].get(e["id"])
        existing = db.execute("select id from threads where id=?", (e["id"],)).fetchone()
        if existing and not known:
            raise RuntimeError(f"Existing unrelated thread ID; refusing overwrite: {e['id']}")
        path = Path(e["path"])
        if path.exists() and not known:
            raise RuntimeError(f"Existing rollout; refusing overwrite: {path}")
    db.close()
    previous.update({"started_utc": datetime.now(timezone.utc).isoformat(), "backup": str(BACKUP),
                     "method": "Additive legacy visible-message rollouts; native app-server scan/read/name/project APIs",
                     "scope": "Eight main chats; helpers remain linked in each project index",
                     "network": "Denied by macOS sandbox for the app-server process",
                     "limitations": "Visible exported text only; tools, attachments and private reasoning excluded. Turn boundaries reconstructed; historical paths unchanged."})
    def save():
        MANIFEST.write_text(json.dumps(previous, indent=2) + "\n")
        MANIFEST.chmod(0o600)
    for e in entries:
        if e["id"] in previous["threads"]:
            continue
        path = Path(e["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as output: output.write(rollout(e))
        path.chmod(0o600)
        previous["threads"][e["id"]] = {k: v for k, v in e.items() if k != "messages"}
        previous["threads"][e["id"]]["message_count"] = len(e["messages"])
        save()
    server = Server(CONTEXT / "codex_chat_import_app_server.log")
    try:
        for name in ["GeoAE", "GeoPretrain"]:
            cwd = str(ROOT.parent / name)
            listed = server.threads(cwd=cwd)
            ids = {t["id"] for t in listed}
            selected = [e for e in entries if e["project"] == name]
            assert {e["id"] for e in selected} <= ids, f"Missing indexed chats for {name}"
            for e in selected:
                thread = server.call("thread/read", {"threadId": e["id"], "includeTurns": True})["thread"]
                expected = [(m["role"], m["text"]) for m in e["messages"]]
                actual = visible(thread)
                assert actual[:len(expected)] == expected, f"History mismatch: {e['title']}"
                assert thread["cwd"] == e["cwd"], e["title"]
                assert thread["status"]["type"] == "notLoaded", "Importer must not resume threads"
                server.call("thread/name/set", {"threadId": e["id"], "name": e["title"]})
                previous["threads"][e["id"]]["verified_visible_messages"] = len(expected)
            project = server.call("project/import", {
                "idempotencyKey": "jetstream-project-chat-import-v1-" + name,
                "name": name, "roots": [{"path": cwd}],
                "threads": sorted(ids)})["project"]
            previous.setdefault("projects", {})[name] = project
            for e in selected:
                thread = server.call("thread/read", {"threadId": e["id"]})["thread"]
                assert thread["projectId"] == project["id"]
            print(f"Verified {name}: {len(selected)} imported main chats; project {project['id']}")
            save()
    finally:
        server.close()
    previous["completed_utc"] = datetime.now(timezone.utc).isoformat()
    previous["result"] = "verified"
    save()
    print("Imported and verified all main chats. No threads resumed or model turns started.")


if __name__ == "__main__":
    main()
