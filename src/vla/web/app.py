"""Small localhost HTTP facade; all device operations live in the detached runner."""
from __future__ import annotations
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import hmac
from importlib.resources import files
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .config import SessionConfig, preflight, dataset_defaults
from .manager import Manager


def create_app(root=None, *, demo=False):
    manager = Manager(root or os.environ.get('VLA_ROOT',Path.cwd()),demo=demo)
    from .processing_service import ProcessingService
    processing = ProcessingService(manager)
    token = secrets.token_urlsafe(32)
    app = FastAPI(docs_url=None,redoc_url=None)
    app.state.manager = manager
    app.state.processing = processing
    app.state.token = token
    app.state.sockets = 0
    app.state.streams = 0
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]'])
    static = Path(str(files('vla.web').joinpath('static')))
    app.mount('/static',StaticFiles(directory=static),name='static')

    def valid_origin(headers):
        origin=headers.get('origin')
        if origin is None:
            return True
        parsed=urlsplit(origin)
        return parsed.scheme in ('http','https') and parsed.netloc == headers.get('host')

    @app.middleware('http')
    async def guard(request, call_next):
        from fastapi.responses import JSONResponse
        if not valid_origin(request.headers):
            return JSONResponse({'detail':'origin rejected'},status_code=403)
        if request.method not in ('GET','HEAD'):
            if not hmac.compare_digest(request.headers.get('x-vla-token',''),token):
                return JSONResponse({'detail':'request token rejected'},status_code=403)
            length=request.headers.get('content-length')
            limit = 524288 if request.url.path.startswith('/api/processing/') else 32768
            if length and (not length.isdigit() or int(length)>limit):
                return JSONResponse({'detail':'request too large'},status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get('/')
    async def index():
        return FileResponse(static/'index.html')

    @app.post('/api/camera-defaults')
    async def camera_defaults(request: Request):
        from dataclasses import replace
        from .config import collection_camera_serials
        try:
            config = SessionConfig.parse(await request.json())
            if config.workflow != 'collect' or config.ros:
                raise ValueError('此接口用于单臂 gamepad 采集；ROS 相机由硬件配置指定')
            config = replace(config, camera_serials=None, side_camera_serial='')
            return {'serials': await asyncio.to_thread(collection_camera_serials, config)}
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post('/api/workspace')
    async def workspace(request: Request):
        from .config import workspace_defaults
        try:
            config = SessionConfig.parse(await request.json())
            return {'bounds': await asyncio.to_thread(workspace_defaults, config)}
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get('/api/bootstrap')
    async def bootstrap():
        return {'token':token,'demo':demo,'active':await asyncio.to_thread(manager.active),
                'defaults':{'model_python':os.environ.get('VLA_PYTHON',__import__('sys').executable)},
                'camera_roles':['base','left_wrist','right_wrist','side'],
                'workspace_root':str(manager.root),'dataset_defaults':dataset_defaults()}

    async def processing_call(fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise HTTPException(422, str(error)) from error
        except (RuntimeError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    @app.get('/api/processing/datasets')
    async def datasets():
        return {'datasets': await processing_call(processing.available), 'last': processing.last()}

    @app.post('/api/processing/open')
    async def open_dataset(request: Request):
        data = await request.json()
        return await processing_call(processing.open, data.get('path') if isinstance(data, dict) else None)

    @app.get('/api/processing/datasets/{dataset_id}')
    async def dataset_summary(dataset_id: str):
        return await processing_call(processing.summary, dataset_id)

    @app.post('/api/processing/datasets/{dataset_id}/draft')
    async def update_draft(dataset_id: str, request: Request):
        return await processing_call(processing.update, dataset_id, await request.json())

    @app.get('/api/processing/datasets/{dataset_id}/episodes')
    async def episodes(dataset_id: str, offset: int = 0, limit: int = 25):
        return await processing_call(processing.episodes, dataset_id, offset, limit)

    @app.get('/api/processing/datasets/{dataset_id}/episodes/{episode}/frames/{frame}')
    async def offline_frame(dataset_id: str, episode: int, frame: int):
        return await processing_call(processing.frame, dataset_id, episode, frame)

    @app.get('/api/processing/datasets/{dataset_id}/episodes/{episode}/frames/{frame}/image/{camera}')
    async def offline_image(dataset_id: str, episode: int, frame: int, camera: str):
        from fastapi.responses import Response
        data = await processing_call(processing.jpeg, dataset_id, episode, frame, camera)
        return Response(data, media_type='image/jpeg')

    @app.get('/api/processing/job')
    async def processing_job():
        return await processing_call(processing.status)

    @app.post('/api/processing/jobs')
    async def start_processing(request: Request):
        data = await request.json()
        if not isinstance(data, dict) or set(data) != {'dataset_id', 'revision'}:
            raise HTTPException(422, 'invalid processing request')
        return await processing_call(processing.start, data['dataset_id'], data['revision'])

    @app.post('/api/processing/jobs/{job_id}/cancel')
    async def cancel_processing(job_id: str):
        return await processing_call(processing.cancel, job_id)

    from .poses import PoseStore
    poses = PoseStore(manager.root, demo=demo)

    @app.get('/api/poses')
    async def pose_records():
        return {'records': await processing_call(poses.list), 'demo': demo}

    @app.post('/api/poses/{record_id}')
    async def edit_pose(record_id: str, request: Request):
        data = await request.json()
        if not isinstance(data, dict) or data.get('operation') not in ('rename', 'delete'):
            raise HTTPException(422, 'invalid pose edit')
        if data['operation'] == 'rename':
            from .poses import note_text
            note = await processing_call(note_text, data.get('note'))
        else:
            note = None
        return await processing_call(poses.edit, record_id, note)

    @app.get('/api/hardware')
    async def hardware_profiles():
        from .hardware import profiles, inventory
        return {'profiles': await asyncio.to_thread(profiles, manager.root),
                'inventory': await asyncio.to_thread(inventory)}

    @app.post('/api/hardware')
    async def save_hardware(request: Request):
        from .hardware import save
        return await processing_call(save, manager.root, await request.json())

    @app.post('/api/preflight')
    async def check(request:Request):
        try:
            config=SessionConfig.parse(await request.json())
            return await asyncio.to_thread(preflight,config,demo=demo,root=manager.root)
        except (ValueError,TypeError) as error:
            raise HTTPException(422,str(error)) from error

    @app.post('/api/sessions',status_code=202)
    async def create(request:Request):
        try:
            return await asyncio.to_thread(manager.create,await request.json())
        except (ValueError,TypeError) as error:
            raise HTTPException(422,str(error)) from error
        except (RuntimeError,OSError) as error:
            raise HTTPException(409,str(error)) from error

    @app.post('/api/commands',status_code=202)
    async def command(request:Request):
        message=await request.json()
        if not isinstance(message,dict) or set(message)-{'session_id','command_id','revision','operation','args'}:
            raise HTTPException(422,'invalid command envelope')
        try:
            return await asyncio.to_thread(manager.rpc,'command',**message)
        except (ValueError,TypeError) as error:
            raise HTTPException(422,str(error)) from error
        except (RuntimeError,OSError) as error:
            raise HTTPException(409,str(error)) from error

    @app.get('/api/state')
    async def state():
        try:
            return await asyncio.to_thread(manager.snapshot)
        except (RuntimeError, OSError, ValueError) as error:
            return {'status': {'state': 'unavailable', 'error': str(error)}, 'samples': []}

    @app.websocket('/api/live')
    async def live(ws:WebSocket):
        protocols=ws.headers.get('sec-websocket-protocol','').split(',')
        if (not valid_origin(ws.headers) or len(protocols)!=2
                or not hmac.compare_digest(protocols[1].strip(),token) or app.state.sockets>=4):
            await ws.close(code=1008)
            return
        await ws.accept(subprotocol='vla')
        app.state.sockets+=1
        seq=-1
        session=None
        try:
            while True:
                started=asyncio.get_running_loop().time()
                try:
                    result = await asyncio.to_thread(manager.snapshot, seq)
                    new_session = result.get('session_id')
                    if new_session != session:
                        session, seq = new_session, -1
                        result = await asyncio.to_thread(manager.snapshot, seq)
                    seq = result.get('seq', seq)
                except (OSError, RuntimeError, ValueError) as error:
                    result = {'status': {'state': 'unavailable', 'error': str(error)}, 'samples': []}
                await asyncio.wait_for(ws.send_json(result),timeout=2)
                await asyncio.sleep(max(0.,.1-(asyncio.get_running_loop().time()-started)))
        except (WebSocketDisconnect,RuntimeError,OSError,asyncio.TimeoutError):
            pass
        finally:
            app.state.sockets-=1

    @app.get('/api/cameras/{role}')
    async def camera(role:str):
        if role not in ('base','left_wrist','right_wrist','side'):
            raise HTTPException(404,'unknown camera')
        if app.state.streams>=12:
            raise HTTPException(429,'too many camera viewers')
        app.state.streams+=1
        async def frames():
            previous=None
            clock = asyncio.get_running_loop()
            deadline = clock.time()
            fps = 10
            try:
                while True:
                    delivered = False
                    available = False
                    try:
                        frame=await asyncio.to_thread(manager.rpc,'camera',role=role)
                        fps = frame.get('preview_fps', 10)
                        available = bool(frame['jpeg'])
                        if frame['jpeg'] and previous!=frame['captured_at']:
                            delivered = True
                            previous=frame['captured_at']
                            data=base64.b64decode(frame['jpeg'])
                            headers = (f'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n'
                                       f'X-Captured-At: {frame["captured_at"]:.9f}\r\n'
                                       f'X-Encoded-At: {frame.get("encoded_at", frame["captured_at"]):.9f}\r\n\r\n')
                            yield headers.encode() + data + b'\r\n'
                    except (OSError,RuntimeError,ValueError):
                        pass
                    if delivered:
                        deadline += 1 / fps
                        if deadline < clock.time():
                            deadline = clock.time() + 1 / fps
                        await asyncio.sleep(max(0., deadline - clock.time()))
                    else:
                        # Do not consume a delivery slot before the next JPEG exists.
                        # Camera/encoder phases differ; a full-period retry loses frames.
                        await asyncio.sleep(1 / (4 * fps) if available else .1)
            finally:
                app.state.streams-=1
        return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame')

    return app


def main():
    p=argparse.ArgumentParser(description='VLA local operations panel; no hardware starts until a session is created.')
    p.add_argument('--demo',action='store_true')
    p.add_argument('--root',type=Path,default=Path(os.environ.get('VLA_ROOT',Path.cwd())))
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.root,demo=args.demo),host='127.0.0.1',port=args.port,access_log=False)


if __name__=='__main__':
    main()
