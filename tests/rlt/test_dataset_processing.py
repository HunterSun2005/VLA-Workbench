"""Synthetic multi-view data only: no hardware, models or downloaded assets."""
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from vla.core.writer_lock import WriterLease
from vla.datacollection.processing import DatasetIndex, export_selection
from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter


def synthetic(root, width=50, episodes=4):
    from vla.datacollection.franka_robotiq_single_left.schema_31d import STATE_31D_FIELDS
    from vla.datacollection.franka_robotiq_dual.schema import STATE_FIELDS as DUAL_FIELDS
    names = {29:schema.LEGACY_STATE_FIELDS,31:STATE_31D_FIELDS,50:schema.STATE_FIELDS,100:DUAL_FIELDS}[width]
    cameras = ('observation.images.base','observation.images.left_wrist') + (('observation.images.right_wrist',) if width==100 else ())
    actions = schema.ACTION_FIELDS if width!=100 else [f'{a}_{n}' for a in ('left','right') for n in schema.ACTION_FIELDS]
    writer = V30DatasetWriter(root,30,'task-0',cameras,32,32,state_fields=names,action_fields=actions,
        extra_features={'complementary_info.phase':{'dtype':'float32','shape':[1],'names':None},
                        'complementary_info.action_valid':{'dtype':'float32','shape':[1],'names':None}},
        info_metadata={'recording_contract':{'test':'raw-contract'}},allow_multiple_tasks=True)
    for ep in range(episodes):
        length=8
        for camera_index,key in enumerate(cameras):
            path=writer.video_path(ep,key);path.parent.mkdir(parents=True,exist_ok=True)
            with av.open(str(path),'w') as out:
                stream=out.add_stream('libx264',rate=30);stream.width=stream.height=32;stream.pix_fmt='yuv420p'
                stream.options={'crf':'0','preset':'ultrafast'}
                for frame in range(length):
                    image=np.full((32,32,3),frame*20+camera_index*3+ep,np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
        states=np.arange(length*width,dtype=np.float32).reshape(length,width)/100+ep*10
        action=np.arange(length*len(actions),dtype=np.float32).reshape(length,len(actions))/100+ep
        valid=np.ones((length,1),dtype=np.float32);valid[-1]=0
        trace=root/'traces'/f'original-{ep}.jsonl';trace.parent.mkdir(exist_ok=True);trace.write_text('{"command":"test"}\n')
        image_stats={k:{**{n:np.ones((3,1,1))*ep for n in ('min','max','mean','std','q01','q10','q50','q90','q99')},'count':np.array([length])} for k in cameras}
        writer.save_episode(states,action,image_stats,task=f'task-{ep%3}',
            frame_features={'complementary_info.phase':np.ones((length,1))*ep,'complementary_info.action_valid':valid},
            extra_episode_metadata={'episode_success':'success' if ep%2 else 'failure',
                'critical_intervals':'[{"start_frame":1,"end_frame":4}]','command_trace':str(trace.relative_to(root))})
    return DatasetIndex(root)


def choose(index,fields=None,episodes=None):
    return {'version':1,'source_fingerprint':index.fingerprint,
            'state_fields':index.names if fields is None else fields,
            'episodes':list(range(len(index.rows))) if episodes is None else episodes}


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('width',[29,31,50,100])
def test_browse_and_export_keep_episode_semantics_and_source(tmp_path,width):
    source=tmp_path/'raw';index=synthetic(source,width)
    before=hashes(source)
    row=index.frame(1,3)
    assert row['episode_index']==1 and row['frame_index']==3
    import cv2
    for cam in index.cameras:
        jpeg=index.jpeg(1,3,cam)
        image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        assert image.mean()==pytest.approx(61+index.cameras.index(cam)*3,abs=3)
    # Deliberately unordered selection: export follows the source column order.
    fields=[index.names[-1], index.names[0]]
    output=export_selection(source,tmp_path/'selected',choose(index,fields,[3,1]))
    result=DatasetIndex(output)
    assert result.names==[index.names[0],index.names[-1]]
    assert len(result.rows)==2 and result.info['total_frames']==16
    assert result.frame(0,3)['observation.state']==[row['observation.state'][0],row['observation.state'][-1]]
    assert result.frame(0,3)['action']==row['action']
    assert result.frame(0,3)['complementary_info.phase']==row['complementary_info.phase']
    assert result.rows.iloc[0]['critical_intervals']==index.rows.iloc[1]['critical_intervals']
    assert result.info['processing']['episode_mapping']==[{'source_episode':1,'output_episode':0},{'source_episode':3,'output_episode':1}]
    assert 'recording_contract' not in result.info
    assert result.info['processing']['source_contract']['recording_contract']=={'test':'raw-contract'}
    tasks=pd.read_parquet(output/'meta/tasks.parquet')
    assert tasks.index.tolist()==['task-1','task-0'] and tasks.task_index.tolist()==[0,1]
    assert result.frame(0,3)['task_index']==0 and result.frame(1,3)['task_index']==1
    for new,old in enumerate((1,3)):
        for cam in index.cameras:assert result.file(new,cam).read_bytes()==index.file(old,cam).read_bytes()
        assert (output/result.rows.iloc[new]['command_trace']).read_text()=='{"command":"test"}\n'
    data=pd.concat(pd.read_parquet(result.file(ep)) for ep in range(2))
    assert data['index'].tolist()==list(range(16))
    stats=json.loads((output/'meta/stats.json').read_text())
    values=np.stack(data.action.to_numpy())[data['complementary_info.action_valid'].astype(bool)]
    np.testing.assert_allclose(stats['action']['mean'],values.mean(axis=0),rtol=1e-6)
    assert stats['action']['count']==[14] and stats['observation.state']['count']==[16]
    assert stats[index.cameras[0]]['mean'][0][0][0]==pytest.approx(2.)
    assert before==hashes(source)
    if width == 50:
        # Reprocessing a derived result must retain the original demonstration contract.
        again = export_selection(output, tmp_path/'again', choose(result, episodes=[0]))
        assert DatasetIndex(again).info['processing']['source_contract']==result.info['processing']['source_contract']



def test_single_dimension_official_loader_and_cli_selection(tmp_path):
    index=synthetic(tmp_path/'raw',episodes=2)
    value=choose(index,[index.names[0]],[1]);value['output_root']=str(tmp_path/'out')
    path=tmp_path/'selection.json';path.write_text(json.dumps(value))
    from vla.cli.extract_dataset_features import extract_dataset_features
    output=extract_dataset_features(index.root,selection_file=path)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset=LeRobotDataset('local/processing',root=output,video_backend='pyav')
    # LeRobot 0.6.1 represents numeric shape-[1] features as scalars.
    assert dataset[3]['observation.state'].numel()==1
    assert tuple(dataset.meta.features['observation.state']['shape'])==(1,)
    assert dataset[3]['task']=='task-1'
    assert dataset[3]['observation.images.base'].shape==(3,32,32)


def test_rejections_cancel_and_source_lease(tmp_path):
    index=synthetic(tmp_path/'raw',episodes=2)
    for i,chosen in enumerate((choose(index,[],[0]),choose(index,episodes=[]),
                               choose(index,episodes=[3]),{**choose(index),'source_fingerprint':'changed'})):
        with pytest.raises(ValueError):export_selection(index.root,tmp_path/f'bad{i}',chosen)
        assert not (tmp_path/f'bad{i}').exists()
    with pytest.raises(InterruptedError):
        export_selection(index.root,tmp_path/'cancelled',choose(index),cancelled=lambda:True)
    assert not (tmp_path/'cancelled').exists() and not list(tmp_path.glob('.*.processing-*'))
    with WriterLease(index.root),pytest.raises(ValueError,match='正在写入'):
        export_selection(index.root,tmp_path/'locked',choose(index))
    with pytest.raises(ValueError,match='non-nested'):
        export_selection(index.root,index.root/'child',choose(index))
    with pytest.raises(FileExistsError):export_selection(index.root,index.root,choose(index))
    def changed(stage,done,total):
        if stage=='校验输出':
            (index.root/'meta/info.json').write_text((index.root/'meta/info.json').read_text()+'\n')
    with pytest.raises(ValueError,match='已变化'):
        export_selection(index.root,tmp_path/'changed',choose(index),progress=changed)
    assert not (tmp_path/'changed').exists()
    # Undo our deliberate external mutation for the corrupt-video case.
    index=DatasetIndex(index.root)
    index.file(0,index.cameras[0]).write_bytes(b'broken video')
    index=DatasetIndex(index.root)
    with pytest.raises(Exception):export_selection(index.root,tmp_path/'broken',choose(index))
    assert not (tmp_path/'broken').exists()

