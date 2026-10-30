"""Synthetic panel soak; requires the web extra, httpx and psutil. Never uses hardware."""
import argparse
import httpx,json,threading,time,sys,statistics
from pathlib import Path
import psutil
from websockets.sync.client import connect
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port',type=int,default=8765)
parser.add_argument('--web-pid',type=int,required=True)
parser.add_argument('--seconds',type=float,default=600)
parser.add_argument('--output',type=Path,default=Path('runtime/web/soak.json'))
args=parser.parse_args()
base=f'http://127.0.0.1:{args.port}'
client=httpx.Client(timeout=5,trust_env=False)
boot=client.get(base+'/api/bootstrap').json()
if not boot.get('demo'):raise SystemExit('Refusing: server must be started with --demo')
if boot.get('active',{} ) and boot['active']['alive']:raise SystemExit('Stop existing demo session before soak')
token=boot['token'];headers={'X-VLA-Token':token}
data={'workflow':'deploy','arms':'dual','source':'gamepad','task':'Web simulated bimanual soak',
      'mode':'auto','model_python':sys.executable}
r=client.post(base+'/api/sessions',json=data,headers=headers);r.raise_for_status();active=r.json()
sid=active['session_id'];runner=psutil.Process(active['pid']);web=psutil.Process(args.web_pid)
for _ in range(100):
 state=client.get(base+'/api/state').json()
 if state.get('status',{}).get('state')=='standby':break
 time.sleep(.1)
else:raise RuntimeError(state)
import uuid
for op in ('enable','record'):
 state=client.get(base+'/api/state').json()
 r=client.post(base+'/api/commands',json={'session_id':sid,'revision':state['revision'],'command_id':str(uuid.uuid4()),'operation':op,'args':{}},headers=headers);r.raise_for_status();time.sleep(.1)
stop=threading.Event();counts={role:0 for role in ('base','left_wrist','right_wrist')}
def camera(role):
 try:
  with httpx.Client(timeout=None,trust_env=False) as c,c.stream('GET',base+'/api/cameras/'+role) as stream:
   for chunk in stream.iter_bytes():
    counts[role]+=chunk.count(b'Content-Type: image/jpeg')
    if stop.is_set():return
 except Exception as e:print('camera',role,str(e),flush=True)
for role in counts:threading.Thread(target=camera,args=(role,),daemon=True).start()
metrics=[];messages=0;samples=0;latencies=[];started=time.monotonic();last=started
runner.cpu_percent();web.cpu_percent()
with connect(f'ws://127.0.0.1:{args.port}/api/live',subprotocols=['vla',token],origin=base,proxy=None) as ws:
 while time.monotonic()-started<args.seconds:
  message=json.loads(ws.recv(timeout=10));messages+=1
  batch=message.get('samples',[]);samples+=len(batch)
  latencies.extend(max(0,message['now']-s['time']) for s in batch)
  now=time.monotonic()
  if now-last>=10:
   row={'elapsed_s':round(now-started,2),'runner_rss_mb':runner.memory_info().rss/2**20,
        'web_rss_mb':web.memory_info().rss/2**20,'runner_cpu_percent':runner.cpu_percent(),
        'web_cpu_percent':web.cpu_percent(),'frames':dict(counts),'seq':message.get('seq')}
   metrics.append(row);last=now
   print(json.dumps(row),flush=True)
stop.set()
result={'duration_s':time.monotonic()-started,'messages':messages,'samples':samples,'camera_frames':counts,
        'latency_p50_ms':statistics.median(latencies)*1000,'latency_p95_ms':sorted(latencies)[int(len(latencies)*.95)]*1000,
        'metrics':metrics,'session_id':sid}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2))
for op in ('discard','stop'):
 state=client.get(base+'/api/state').json()
 r=client.post(base+'/api/commands',json={'session_id':sid,'revision':state['revision'],
   'command_id':str(uuid.uuid4()),'operation':op,'args':{}},headers=headers)
 r.raise_for_status()
 time.sleep(.3)
print(f'DONE {args.output}',flush=True)
