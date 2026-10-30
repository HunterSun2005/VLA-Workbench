from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from vla.core.phase_annotations import (
    save_phase_annotations,
    set_episode_phase_annotation,
)


WEB_ASSET = Path(__file__).with_name("phase_annotator.html")


class PhaseAnnotationWebState:
    def __init__(
        self,
        *,
        meta: Any,
        rows: list[dict[str, Any]],
        row_positions: dict[int, int],
        camera_keys: list[str],
        document: dict[str, Any],
        output: Path,
        start_position: int,
        jump_seconds: float,
    ) -> None:
        self.meta = meta
        self.rows = rows
        self.row_positions = row_positions
        self.camera_keys = camera_keys
        self.document = document
        self.output = output
        self.start_position = start_position
        self.jump_seconds = jump_seconds
        self.lock = threading.Lock()
        self.row_by_id = {int(row["episode_index"]): row for row in rows}

    def bootstrap(self) -> dict[str, Any]:
        with self.lock:
            episodes = [
                {
                    "episode_id": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "reviewed": self.document["episodes"]
                    .get(str(int(row["episode_index"])), {})
                    .get("reviewed", False),
                }
                for row in self.rows
            ]
        return {
            "fps": float(self.meta.fps),
            "episodes": episodes,
            "camera_keys": self.camera_keys,
            "start_position": self.start_position,
            "jump_seconds": self.jump_seconds,
            "output": str(self.output),
        }

    def episode(self, episode_id: int) -> dict[str, Any]:
        row = self.row_by_id.get(episode_id)
        if row is None:
            raise KeyError(f"unknown episode {episode_id}")
        length = int(row["length"])
        with self.lock:
            stored = self.document["episodes"].get(str(episode_id), {})
            intervals = [
                [int(interval["start_frame"]), int(interval["end_frame"])]
                for interval in stored.get("intervals", [])
            ]
            reviewed = bool(stored.get("reviewed", False))
        videos = [
            {
                "camera_key": camera_key,
                "url": f"/video?episode={episode_id}&camera={self.camera_keys.index(camera_key)}",
                "from_timestamp": float(
                    row.get(f"videos/{camera_key}/from_timestamp", 0.0)
                ),
            }
            for camera_key in self.camera_keys
        ]
        return {
            "episode_id": episode_id,
            "length": length,
            "reviewed": reviewed,
            "intervals": intervals,
            "videos": videos,
        }

    def video_path(self, episode_id: int, camera_index: int) -> Path:
        row = self.row_by_id.get(episode_id)
        if row is None:
            raise KeyError(f"unknown episode {episode_id}")
        if not 0 <= camera_index < len(self.camera_keys):
            raise KeyError(f"unknown camera index {camera_index}")
        position = self.row_positions[episode_id]
        relative = self.meta.get_video_file_path(position, self.camera_keys[camera_index])
        path = (Path(self.meta.root) / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"episode video is missing: {path}")
        return path

    def save_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = int(payload["episode_id"])
        row = self.row_by_id.get(episode_id)
        if row is None:
            raise ValueError(f"unknown episode {episode_id}")
        length = int(row["length"])
        if int(payload.get("length", -1)) != length:
            raise ValueError(f"episode {episode_id} length does not match dataset")
        raw_intervals = payload.get("intervals")
        if not isinstance(raw_intervals, list):
            raise ValueError("intervals must be a list")
        intervals: list[tuple[int, int]] = []
        for interval in raw_intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("each interval must be [start_frame, end_frame]")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in interval):
                raise ValueError("interval bounds must be integers")
            intervals.append((interval[0], interval[1]))
        reviewed = payload.get("reviewed")
        if not isinstance(reviewed, bool):
            raise ValueError("reviewed must be boolean")
        with self.lock:
            set_episode_phase_annotation(
                self.document,
                episode_id=episode_id,
                length=length,
                intervals=intervals,
                reviewed=reviewed,
            )
            save_phase_annotations(self.output, self.document)
            reviewed_count = sum(
                annotation.get("reviewed") is True
                for annotation in self.document["episodes"].values()
            )
        return {"ok": True, "reviewed_count": reviewed_count}


def _parse_range(value: str | None, size: int) -> tuple[int, int, bool]:
    if value is None:
        return 0, size - 1, False
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("unsupported Range header")
    bounds = value.removeprefix("bytes=").split("-", maxsplit=1)
    try:
        if bounds[0] == "":
            suffix = int(bounds[1])
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(bounds[0])
            end = size - 1 if bounds[1] == "" else int(bounds[1])
    except ValueError as error:
        raise ValueError("invalid Range header") from error
    if start < 0 or start >= size or end < start:
        raise ValueError("Range is outside the video")
    return start, min(end, size - 1), True


def create_phase_annotation_server(
    *,
    state: PhaseAnnotationWebState,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    html = WEB_ASSET.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "EvoRLTPhaseAnnotator/1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _error(self, error: Exception, status: HTTPStatus) -> None:
            self._send_json({"error": str(error)}, status)

        def _query_int(self, query: dict[str, list[str]], name: str) -> int:
            values = query.get(name)
            if values is None or len(values) != 1:
                raise ValueError(f"query parameter {name!r} is required")
            return int(values[0])

        def _serve_video(self, query: dict[str, list[str]], *, body: bool) -> None:
            try:
                episode_id = self._query_int(query, "episode")
                camera_index = self._query_int(query, "camera")
                path = state.video_path(episode_id, camera_index)
                size = path.stat().st_size
                start, end, partial = _parse_range(self.headers.get("Range"), size)
            except (KeyError, ValueError, FileNotFoundError) as error:
                self._error(error, HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not body:
                return
            remaining = end - start + 1
            with path.open("rb") as stream:
                stream.seek(start)
                try:
                    while remaining > 0:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/video":
                self._serve_video(parse_qs(parsed.query), body=False)
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)
                return
            if parsed.path == "/api/bootstrap":
                self._send_json(state.bootstrap())
                return
            if parsed.path == "/api/episode":
                try:
                    self._send_json(state.episode(self._query_int(query, "episode")))
                except (KeyError, ValueError) as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                return
            if parsed.path == "/video":
                self._serve_video(query, body=True)
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("invalid request body length")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/episode":
                try:
                    self._send_json(state.save_episode(self._read_json()))
                except (KeyError, TypeError, ValueError) as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/quit":
                self._send_json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), Handler)


def run_phase_annotation_web(
    *,
    state: PhaseAnnotationWebState,
    host: str,
    port: int,
) -> str:
    server = create_phase_annotation_server(state=state, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    url = f"http://{display_host}:{bound_port}"
    try:
        print(f"Open the phase annotator in a browser: {url}", flush=True)
        print("Press Ctrl+C to stop the server without changing confirmed annotations.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return url
