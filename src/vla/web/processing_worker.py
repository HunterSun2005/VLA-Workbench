"""Detached single export; file-based status and cooperative cancellation."""
import argparse
import json
import signal
import time
import traceback
from pathlib import Path

from .processing_service import save_json, operation_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    directory = args.root/'runtime/web/processing/jobs'/args.job
    cancelled = False
    def stop(*_):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def is_cancelled():
        return cancelled or (directory/'cancel').exists()
    last = 0.
    def progress(stage,done,total):
        nonlocal last
        now = time.monotonic()
        if now-last>.2:
            save_json(directory/'status.json',{'state':'running','stage':stage,'done':done,'total':total})
            last=now
    try:
        # Wait until the parent has atomically published this job under the shared gate.
        with operation_guard(args.root):
            pass
        from vla.datacollection.processing import export_selection
        chosen = json.loads((directory/'selection.json').read_text())
        output = export_selection(chosen['source_root'],args.output,chosen,progress=progress,cancelled=is_cancelled)
        save_json(directory/'status.json',{'state':'succeeded','stage':'转换完成','output_root':str(output)})
    except BaseException as error:
        traceback.print_exc()
        save_json(directory/'status.json',{'state':'cancelled' if isinstance(error,InterruptedError) else 'failed',
                                         'error':str(error),'stage':'已取消' if isinstance(error,InterruptedError) else '转换失败'})


if __name__=='__main__':
    main()
