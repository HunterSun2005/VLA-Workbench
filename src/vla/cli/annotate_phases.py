"""Interactively mark critical-phase intervals in LeRobot v3 episodes."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.core.phase_annotations import (
    PHASE_ANNOTATIONS_NAME,
    load_phase_annotations,
    new_phase_annotations,
    save_phase_annotations,
    set_episode_phase_annotation,
)


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default="local/phase-annotation",
        help="Local identifier required by LeRobot metadata loading; no Hub access is performed.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Defaults to <dataset-root>/meta/{PHASE_ANNOTATIONS_NAME}.",
    )
    parser.add_argument(
        "--camera-key",
        action="append",
        default=None,
        help="Camera feature to display. Repeat for multiple cameras; defaults to all cameras.",
    )
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--display-height", type=int, default=480)
    parser.add_argument("--jump-seconds", type=float, default=1.0)
    parser.add_argument(
        "--ui",
        choices=("auto", "web", "opencv"),
        default="auto",
        help="auto uses OpenCV HighGUI when available and otherwise starts a local web UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web UI bind address.")
    parser.add_argument("--port", type=int, default=0, help="Web UI port; 0 selects a free port.")
    return parser.parse_args()


def _episode_rows(meta: Any) -> list[dict[str, Any]]:
    rows = [dict(meta.episodes[index]) for index in range(len(meta.episodes))]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if len({int(row["episode_index"]) for row in rows}) != len(rows):
        raise ValueError("dataset contains duplicate episode indices")
    return rows


class EpisodeVideoReader:
    def __init__(
        self,
        *,
        meta: Any,
        row_position: int,
        row: dict[str, Any],
        camera_keys: list[str],
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._captures: list[tuple[str, Any, int]] = []
        for camera_key in camera_keys:
            path = Path(meta.root) / meta.get_video_file_path(row_position, camera_key)
            if not path.is_file():
                raise FileNotFoundError(f"episode video is missing: {path}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open episode video: {path}")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(meta.fps)
            from_timestamp = float(row.get(f"videos/{camera_key}/from_timestamp", 0.0))
            offset = int(round(from_timestamp * source_fps))
            self._captures.append((camera_key, capture, offset))

    def close(self) -> None:
        for _, capture, _ in self._captures:
            capture.release()

    def read(self, local_frame: int) -> list[tuple[str, np.ndarray]]:
        frames: list[tuple[str, np.ndarray]] = []
        for camera_key, capture, offset in self._captures:
            capture.set(self._cv2.CAP_PROP_POS_FRAMES, offset + local_frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode {camera_key} at episode frame {local_frame}")
            frames.append((camera_key, frame))
        return frames


def _resize_to_height(cv2: Any, frame: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _render(
    cv2: Any,
    frames: list[tuple[str, np.ndarray]],
    *,
    display_height: int,
    episode_id: int,
    episode_position: int,
    episode_count: int,
    frame_index: int,
    frame_count: int,
    fps: float,
    intervals: list[tuple[int, int]],
    pending_start: int | None,
    reviewed: bool,
    playing: bool,
) -> np.ndarray:
    footer_height = 125
    camera_height = max(120, display_height - footer_height)
    prepared = []
    for camera_key, frame in frames:
        view = _resize_to_height(cv2, frame, camera_height)
        cv2.putText(
            view,
            camera_key,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        prepared.append(view)
    camera_canvas = np.concatenate(prepared, axis=1)
    canvas_width = max(1050, camera_canvas.shape[1])
    canvas = np.zeros((camera_canvas.shape[0], canvas_width, 3), dtype=np.uint8)
    left_padding = (canvas_width - camera_canvas.shape[1]) // 2
    canvas[:, left_padding : left_padding + camera_canvas.shape[1]] = camera_canvas
    footer = np.zeros((footer_height, canvas_width, 3), dtype=np.uint8)
    critical = any(start <= frame_index < end for start, end in intervals)
    status = (
        f"episode {episode_id} ({episode_position + 1}/{episode_count})  "
        f"frame {frame_index}/{frame_count - 1}  time {frame_index / fps:.2f}s"
    )
    cv2.putText(
        footer,
        status,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    pending = "none" if pending_start is None else str(pending_start)
    cv2.putText(
        footer,
        f"phase={'CRITICAL' if critical else 'VLA'}  "
        f"{'REVIEWED' if reviewed else 'UNREVIEWED'}  "
        f"{'PLAY' if playing else 'PAUSE'}  pending_start={pending}",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "i=start  o=end  x=delete interval  r=clear  v=confirm + next  p=previous",
        (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "a/d=one frame  j/l=jump  space=play/pause  q or Esc=save + quit",
        (10, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    x0, x1, y0, y1 = 10, max(11, canvas.shape[1] - 10), 101, 118
    cv2.rectangle(footer, (x0, y0), (x1, y1), (90, 90, 90), -1)
    for start, end in intervals:
        start_x = x0 + int((x1 - x0) * start / frame_count)
        end_x = x0 + int((x1 - x0) * end / frame_count)
        cv2.rectangle(footer, (start_x, y0), (max(start_x + 1, end_x), y1), (40, 40, 220), -1)
    cursor_x = x0 + int((x1 - x0) * frame_index / max(frame_count - 1, 1))
    cv2.line(footer, (cursor_x, y0 - 3), (cursor_x, y1 + 3), (255, 255, 255), 2)
    return np.concatenate([canvas, footer], axis=0)


def _stored_episode(document: dict[str, Any], episode_id: int, length: int) -> tuple[list[tuple[int, int]], bool]:
    stored = document["episodes"].get(str(episode_id))
    if stored is None:
        return [], False
    if int(stored["length"]) != length:
        raise ValueError(
            f"episode {episode_id} length changed: annotations={stored['length']}, dataset={length}"
        )
    intervals = [
        (int(interval["start_frame"]), int(interval["end_frame"]))
        for interval in stored["intervals"]
    ]
    return intervals, bool(stored["reviewed"])


def _select_ui(requested: str, *, highgui_backend: str, graphical_session: bool) -> str:
    highgui_available = graphical_session and bool(highgui_backend)
    if requested == "auto":
        return "opencv" if highgui_available else "web"
    if requested == "opencv" and not highgui_available:
        raise RuntimeError(
            "--ui opencv requires an OpenCV HighGUI backend and a graphical session; "
            f"current backend={highgui_backend!r}. Use --ui web."
        )
    return requested


def annotate_phases(args: argparse.Namespace) -> Path:
    if args.display_height < 300:
        raise ValueError("display-height must be at least 300")
    if args.jump_seconds <= 0:
        raise ValueError("jump-seconds must be positive")
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata is missing: {info_path}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else dataset_root / "meta" / PHASE_ANNOTATIONS_NAME
    )
    info_sha256 = sha256_file(info_path)

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    rows = _episode_rows(meta)
    if not rows:
        raise ValueError("dataset contains no episodes")
    row_positions = {
        int(meta.episodes[index]["episode_index"]): index
        for index in range(len(meta.episodes))
    }
    episode_lengths = {int(row["episode_index"]): int(row["length"]) for row in rows}
    if output.exists():
        document = load_phase_annotations(
            output,
            expected_info_sha256=info_sha256,
            episode_lengths=episode_lengths,
        )
    else:
        document = new_phase_annotations(dataset_info_sha256=info_sha256, fps=float(meta.fps))

    camera_keys = list(args.camera_key or meta.video_keys)
    if not camera_keys:
        raise ValueError("dataset has no video camera features")
    unknown_cameras = sorted(set(camera_keys).difference(meta.video_keys))
    if unknown_cameras:
        raise ValueError(f"unknown camera keys: {unknown_cameras}; available={meta.video_keys}")

    if args.start_episode is not None:
        positions = [
            index for index, row in enumerate(rows)
            if int(row["episode_index"]) == args.start_episode
        ]
        if not positions:
            raise ValueError(f"unknown start episode {args.start_episode}")
        position = positions[0]
    else:
        position = next(
            (
                index for index, row in enumerate(rows)
                if document["episodes"].get(str(int(row["episode_index"])), {}).get("reviewed")
                is not True
            ),
            0,
        )

    import cv2

    log.info("phase annotations: %s", output)
    log.info(
        "controls: i/o interval, x delete, r clear, v confirm+next, p previous, "
        "a/d frame, j/l jump, space play, q save+quit"
    )
    graphical_session = bool(
        os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    highgui_backend = cv2.currentUIFramework() if hasattr(cv2, "currentUIFramework") else ""
    selected_ui = _select_ui(
        args.ui,
        highgui_backend=highgui_backend,
        graphical_session=graphical_session,
    )
    if args.ui == "auto":
        if selected_ui == "web":
            log.info(
                "OpenCV HighGUI is unavailable (backend=%r, graphical_session=%s); "
                "using browser UI",
                highgui_backend,
                graphical_session,
            )
    if selected_ui == "web":
        from vla.cli.phase_annotation_web import (
            PhaseAnnotationWebState,
            run_phase_annotation_web,
        )

        state = PhaseAnnotationWebState(
            meta=meta,
            rows=rows,
            row_positions=row_positions,
            camera_keys=camera_keys,
            document=document,
            output=output,
            start_position=position,
            jump_seconds=args.jump_seconds,
        )
        run_phase_annotation_web(state=state, host=args.host, port=args.port)
        return output

    window_name = "VLA Critical Phase Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        quit_requested = False
        while not quit_requested:
            row = rows[position]
            episode_id = int(row["episode_index"])
            frame_count = int(row["length"])
            intervals, reviewed = _stored_episode(document, episode_id, frame_count)
            current_frame = 0
            pending_start: int | None = None
            playing = False
            reader = EpisodeVideoReader(
                meta=meta,
                row_position=row_positions[episode_id],
                row=row,
                camera_keys=camera_keys,
            )
            try:
                leave_episode = False
                while not leave_episode and not quit_requested:
                    frames = reader.read(current_frame)
                    canvas = _render(
                        cv2,
                        frames,
                        display_height=args.display_height,
                        episode_id=episode_id,
                        episode_position=position,
                        episode_count=len(rows),
                        frame_index=current_frame,
                        frame_count=frame_count,
                        fps=float(meta.fps),
                        intervals=intervals,
                        pending_start=pending_start,
                        reviewed=reviewed,
                        playing=playing,
                    )
                    cv2.imshow(window_name, canvas)
                    delay_ms = max(1, int(round(1000 / float(meta.fps)))) if playing else 0
                    key = cv2.waitKeyEx(delay_ms)
                    if key < 0 and playing:
                        if current_frame >= frame_count - 1:
                            playing = False
                        else:
                            current_frame += 1
                        continue
                    character = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
                    if character == " ":
                        playing = not playing
                    elif character == "a" or key in (81, 2424832):
                        playing = False
                        current_frame = max(0, current_frame - 1)
                    elif character == "d" or key in (83, 2555904):
                        playing = False
                        current_frame = min(frame_count - 1, current_frame + 1)
                    elif character == "j":
                        playing = False
                        current_frame = max(
                            0,
                            current_frame - max(1, int(round(args.jump_seconds * float(meta.fps)))),
                        )
                    elif character == "l":
                        playing = False
                        current_frame = min(
                            frame_count - 1,
                            current_frame + max(1, int(round(args.jump_seconds * float(meta.fps)))),
                        )
                    elif character == "i":
                        pending_start = current_frame
                        reviewed = False
                    elif character == "o":
                        if pending_start is None:
                            log.warning("mark a critical start with 'i' before marking its end")
                        elif current_frame < pending_start:
                            log.warning("critical end must not precede its start")
                        else:
                            candidate = (pending_start, current_frame + 1)
                            if any(candidate[0] < end and start < candidate[1] for start, end in intervals):
                                log.warning("critical interval overlaps an existing interval")
                            else:
                                intervals.append(candidate)
                                intervals.sort()
                                pending_start = None
                                reviewed = False
                    elif character == "x":
                        remaining = [
                            interval for interval in intervals
                            if not interval[0] <= current_frame < interval[1]
                        ]
                        if len(remaining) == len(intervals):
                            log.warning("cursor is not inside a critical interval")
                        else:
                            intervals = remaining
                            reviewed = False
                    elif character == "r":
                        intervals = []
                        pending_start = None
                        reviewed = False
                    elif character == "v":
                        if pending_start is not None:
                            log.warning("close the pending critical interval with 'o' before confirming")
                            continue
                        reviewed = True
                        set_episode_phase_annotation(
                            document,
                            episode_id=episode_id,
                            length=frame_count,
                            intervals=intervals,
                            reviewed=True,
                        )
                        save_phase_annotations(output, document)
                        log.info("confirmed episode %d: %d critical intervals", episode_id, len(intervals))
                        if position < len(rows) - 1:
                            position += 1
                            leave_episode = True
                        else:
                            quit_requested = True
                    elif character == "p":
                        if pending_start is not None:
                            log.warning("discard or close the pending interval before changing episode")
                            continue
                        set_episode_phase_annotation(
                            document,
                            episode_id=episode_id,
                            length=frame_count,
                            intervals=intervals,
                            reviewed=reviewed,
                        )
                        save_phase_annotations(output, document)
                        position = max(0, position - 1)
                        leave_episode = True
                    elif character == "q" or key == 27:
                        if pending_start is not None:
                            log.warning("discarding unfinished critical interval starting at %d", pending_start)
                        set_episode_phase_annotation(
                            document,
                            episode_id=episode_id,
                            length=frame_count,
                            intervals=intervals,
                            reviewed=reviewed,
                        )
                        save_phase_annotations(output, document)
                        quit_requested = True
            finally:
                reader.close()
    finally:
        cv2.destroyAllWindows()
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = annotate_phases(parse_args())
    print(output)


if __name__ == "__main__":
    main()
