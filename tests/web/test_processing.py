"""Processing HTTP, draft conflicts and detached job lifecycle; no hardware."""
import json
from pathlib import Path
import time

import pytest

from vla.datacollection.processing import DatasetIndex
from vla.web.manager import Manager, process_identity
from vla.web.processing_service import ProcessingService, save_json
from tests.rlt.test_dataset_processing import synthetic


def test_draft_conflicts_hardware_gate_and_preview_cache(tmp_path,monkeypatch):
    index=synthetic(tmp_path/'datasets/raw/test',episodes=1)
    manager=Manager(tmp_path);service=ProcessingService(manager)
    assert service.available()==['datasets/raw/test']
    summary=service.open('datasets/raw/test');draft=summary['draft'];did=draft['dataset_id']
    updated=service.update(did,{'revision':0,'state_fields':[index.names[0]],'episodes':[0],'output_root':'datasets/sft/custom'})
    assert updated['revision']==1
    with pytest.raises(RuntimeError,match='另一页面'):service.update(did,{'revision':0})
    assert ProcessingService(manager).summary(did)['draft']['state_fields']==[index.names[0]]
    with pytest.raises(ValueError):service.jpeg(did,0,0,'../../etc/passwd')
    for frame in range(8):service.jpeg(did,0,frame,index.cameras[0])
    assert len(service.jpeg_cache)<=24
    from vla.web.config import SessionConfig
    from dataclasses import asdict
    record={'alive':True,'demo':False,'config':asdict(SessionConfig(task='test',dataset_root=str(tmp_path/'other')))}
    monkeypatch.setattr(manager,'active',lambda:record)
    assert service.frame(did,0,0)['frame_index']==0  # Other completed dataset is previewable.
    with pytest.raises(RuntimeError,match='停止'):service.start(did,1)
    record['config']['dataset_root']=str(index.root)
    with pytest.raises(ValueError,match='正在使用'):service.frame(did,0,0)
    # The reverse direction uses the same gate without launching hardware.
    save_json(service.directory/'active.json',{'job_id':'test','pid':__import__('os').getpid(),'identity':process_identity(__import__('os').getpid())})
    monkeypatch.setattr('vla.web.manager.preflight',lambda *a,**k:{'ok':True,'dataset_root':str(tmp_path/'new')})
    with pytest.raises(RuntimeError,match='数据转换正在运行'):manager.create({'task':'test'})


def test_http_preview_and_detached_job_reattach(tmp_path):
    from fastapi.testclient import TestClient
    from vla.web.app import create_app
    index=synthetic(tmp_path/'raw',episodes=2)
    app=create_app(tmp_path,demo=True)
    with TestClient(app,base_url='http://127.0.0.1') as client:
        token=client.get('/api/bootstrap').json()['token'];headers={'X-VLA-Token':token}
        assert client.post('/api/processing/open',json={'path':'raw'}).status_code==403
        summary=client.post('/api/processing/open',json={'path':'raw'},headers=headers).json();did=summary['draft']['dataset_id']
        url=f'/api/processing/datasets/{did}'
        assert client.get(url+'/episodes').json()['total']==2
        assert client.get(url+'/episodes/0/frames/3').json()['frame_index']==3
        assert client.get(url+'/episodes/0/frames/3/image/observation.images.base').headers['content-type']=='image/jpeg'
        window=client.get(url+'/episodes/0/window?frame=3&seconds=10').json()
        assert window['start']==0 and window['stop']==8
        frame=client.get(url+'/episodes/0/frames/3').json()
        assert window['rows'][3]=={k:frame[k] for k in window['rows'][3]}
        assert client.get(url+'/episodes/0/window?frame=999').status_code==422
        assert client.get(url+'/episodes/0/window?frame=3&seconds=600').status_code==422
        assert client.get(url+'/episodes/0/frames/999').status_code==422
        assert client.get(url+'/episodes/0/frames/3/image/not-a-camera').status_code==422
        response=client.post('/api/processing/jobs',json={'dataset_id':did,'revision':0},headers=headers)
        assert response.status_code==200,response.text
        job=response.json()
        duplicate=client.post('/api/processing/jobs',json={'dataset_id':did,'revision':0},headers=headers)
        assert duplicate.status_code==409
    # Closing HTTP service must not terminate the independently owned export process.
    resumed=ProcessingService(Manager(tmp_path,demo=True))
    for _ in range(200):
        status=resumed.status()
        if not status['alive']:break
        time.sleep(.05)
    assert status['state']=='succeeded',status
    assert DatasetIndex(Path(job['output_root'])).info['total_episodes']==2
    with pytest.raises(FileExistsError):resumed.start(did,0)


def test_numeric_window_is_bounded_exact_and_keeps_missing_values(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    index=synthetic(tmp_path/'raw',episodes=1)
    length=6001
    state=np.zeros((length,len(index.names)),dtype=np.float32)
    state[:,0]=np.arange(length)
    state[3001,1]=np.nan
    table=pa.table({'frame_index':np.arange(length),'episode_index':np.zeros(length,dtype=int),
                    'timestamp':np.arange(length)/30,'observation.state':state.tolist(),
                    'action':[[0.]*len(index.features['action']['names'])]*length})
    pq.write_table(table,index.file(0),row_group_size=128)
    index.rows.loc[0,'length']=length
    window=index.window(0,3001,60)
    assert (window['start'],window['stop'])==(1201,4802)
    assert window['sampled'] and len(window['rows'])<=2001
    current=next(r for r in window['rows'] if r['frame_index']==3001)
    assert current['observation.state'][:2]==[3001.,None]
    assert window['rows'][0]['frame_index']==1201
    assert window['rows'][-1]['frame_index']==4801
    assert index.window(0,0)['start']==0
    assert index.window(0,length-1)['stop']==length
