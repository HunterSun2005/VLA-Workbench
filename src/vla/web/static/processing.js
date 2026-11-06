'use strict';
(() => {
 const el=id=>document.getElementById(id), base='/api/processing';
 let info=null,draft=null,page=0,episode=null,length=0,frameNumber=0,frameData=null,job=null;
 let writes=Promise.resolve(),saveBusy=0,frameSeq=0,frameAbort=null,playing=false,playTimer=null,playSeq=0,lastDefault='',initialized=false,openVersion=0,exportStarting=false;
 let pendingFrame=null,frameLoop=null;
 const issues=new Map();
 const error=(message,key='operation')=>{if(message)issues.set(key,message);else issues.delete(key);el('processingError').textContent=[...issues.values()].join('\n');el('processingError').hidden=!issues.size;};
 const attempt=fn=>async()=>{try{error('');await fn();}catch(e){error(e.message);}};
 const endpoint=suffix=>`${base}/datasets/${draft.dataset_id}${suffix}`;
 function text(tag,value){const x=document.createElement(tag);x.textContent=value;return x;}
 function number(v){if(v==null)return '缺失';if(typeof v==='number')return Number.isInteger(v)?String(v):v.toPrecision(7);return typeof v==='object'?JSON.stringify(v):String(v);}
 function pathDefault(){return `datasets/sft/${draft.source_root.split('/').filter(Boolean).at(-1)}_${draft.state_fields.length}d`;}
 function updateDefault(force=false){if(!draft)return;const next=pathDefault(),input=el('processingOutput');if(force||!input.value||input.value===lastDefault){input.value=next;draft.output_root=next;}lastDefault=next;}
 function selectionLabel(){if(!draft)return;el('processingDimension').textContent=`${draft.state_fields.length}D`;el('processingSelection').textContent=`保留 ${draft.episodes.length} / ${info.total_episodes} 回合 · 剔除 ${info.total_episodes-draft.episodes.length} · 状态 ${draft.state_fields.length}D`;el('processingExport').disabled=!draft.state_fields.length||!draft.episodes.length||Boolean(job?.alive)||saveBusy>0||exportStarting||hardwareRunning();}
 function hardwareRunning(){return !demo&&configLocked();}
 function persist(){if(!draft)return Promise.resolve();const id=draft.dataset_id,values={state_fields:[...draft.state_fields],episodes:[...draft.episodes],output_root:el('processingOutput').value};draft.output_root=values.output_root;saveBusy++;selectionLabel();el('processingSaved').textContent='正在保存选择…';
  writes=writes.then(async()=>{if(draft.dataset_id!==id)throw Error('数据集已切换');const saved=await api(`${base}/datasets/${id}/draft`,{revision:draft.revision,action_space:draft.action_space||null,allow_derived:!!draft.allow_derived,preview_representation:draft.preview_representation||"absolute",preview_anchor_frame:draft.preview_anchor_frame||0,...values});draft.revision=saved.revision;el('processingSaved').textContent='选择已保存';}).catch(e=>{el('processingSaved').textContent='选择未保存';error(e.message);throw e;}).finally(()=>{saveBusy--;selectionLabel();});
  // Keep a handled tail; each caller still receives its own rejection.
  const result=writes;writes=result.catch(()=>{});return result;
 }
 function buildFields(){const matching=Object.entries(info.presets).find(([,names])=>names.length===draft.state_fields.length&&names.every(n=>draft.state_fields.includes(n)));const preset=el('processingPreset');preset.querySelector('[value="custom"]')?.remove();if(matching)preset.value=matching[0];else{const custom=text('option','自定义');custom.value='custom';preset.append(custom);preset.value='custom';}el('processingFields').replaceChildren();const groups=new Map();for(const d of info.fields){const key=`${d.arm==='left'?'左臂':d.arm==='right'?'右臂':'单臂'} · ${d.group} [${d.unit}]`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(d);}
  for(const [title,ds]of groups){const block=document.createElement('details');block.open=false;const summary=document.createElement('summary'),all=document.createElement('input');all.type='checkbox';all.checked=ds.every(d=>draft.state_fields.includes(d.name));all.indeterminate=!all.checked&&ds.some(d=>draft.state_fields.includes(d.name));all.onclick=e=>e.stopPropagation();all.onchange=attempt(async()=>{const selected=new Set(draft.state_fields);for(const d of ds)all.checked?selected.add(d.name):selected.delete(d.name);await fieldsChanged(info.fields.filter(d=>selected.has(d.name)).map(d=>d.name));});summary.append(all,document.createTextNode(' '+title));block.append(summary);
   for(const d of ds){const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=draft.state_fields.includes(d.name);check.onchange=attempt(async()=>{const set=new Set(draft.state_fields);check.checked?set.add(d.name):set.delete(d.name);await fieldsChanged(info.fields.filter(d=>set.has(d.name)).map(d=>d.name));});label.append(check,document.createTextNode(d.name));block.append(label);}el('processingFields').append(block);
  }
 }
 async function fieldsChanged(names){draft.state_fields=names;updateDefault();selectionLabel();buildFields();renderValues();await persist();}
 async function refreshList(){const data=await api(base+'/datasets');el('processingDatasets').replaceChildren(...data.datasets.map(v=>{const o=document.createElement('option');o.value=v;return o;}));return data;}
 async function load(summary){stopPlayback();pendingFrame=null;frameNumber=0;length=0;el('processingTimeline').value=el('processingFrameNumber').value='0';el('processingTimeline').max=el('processingFrameNumber').max='0';error('','frame');frameSeq++;frameAbort?.abort();frameStatus('');info=summary;draft=summary.draft;el('processingActionSpace').value=draft.action_space||'';el('processingDerived').checked=!!draft.allow_derived;el('processingRepresentation').value=draft.preview_representation||'absolute';el('processingAnchor').value=draft.preview_anchor_frame||0;page=0;episode=null;frameData=null;lastDefault=pathDefault();el('processingSource').value=draft.source_root;el('processingOutput').value=draft.output_root;el('processingSummary').textContent=`${summary.robot_type||'Franka'} · ${summary.total_episodes} 回合 · ${summary.total_frames} 帧 · ${summary.fps} FPS · ${summary.cameras.length} 视角`;el('processingPreset').replaceChildren(...Object.keys(info.presets).map(name=>{const o=text('option',name);o.value=name;return o;}));el('processingImages').replaceChildren();el('processingFrameTitle').textContent='选择一个回合';el('processingTime').textContent='';el('processingValues').replaceChildren();el('processingActions').replaceChildren();el('processingSaved').textContent='选择已保存';buildFields();offlinePlots.reset(info);selectionLabel();transportState();await listEpisodes();if(summary.source_changed)error('源数据已变化，原处理选择已重置为全部保留，请重新确认。');}
 async function listEpisodes(){const id=draft.dataset_id,index=page;const result=await api(endpoint(`/episodes?offset=${page*25}&limit=25`));if(id!==draft?.dataset_id||index!==page)return;el('processingEpisodes').replaceChildren();for(const e of result.episodes){const row=document.createElement('tr');row.classList.toggle('excluded',!draft.episodes.includes(e.episode));const check=document.createElement('input');check.type='checkbox';check.checked=!draft.episodes.includes(e.episode);check.setAttribute('aria-label',`剔除回合 ${e.episode}`);check.onchange=attempt(async()=>{const kept=new Set(draft.episodes);check.checked?kept.delete(e.episode):kept.add(e.episode);draft.episodes=[...kept].sort((a,b)=>a-b);row.classList.toggle('excluded',check.checked);selectionLabel();await persist();});const cell=document.createElement('td');cell.append(check);row.append(cell,text('td',String(e.episode)),text('td',`${(e.tasks||[]).join(' / ')} · ${e.outcome??'未标注'}`),text('td',`${e.length} / ${e.duration.toFixed(2)}s`));const view=document.createElement('button');view.textContent='查看';view.onclick=attempt(async()=>{stopPlayback();pendingFrame=null;frameSeq++;frameAbort?.abort();frameData=null;offlinePlots.clear();el('processingImages').replaceChildren();el('processingValues').replaceChildren();el('processingActions').replaceChildren();episode=e.episode;length=e.length;el('processingTimeline').max=el('processingFrameNumber').max=String(length-1);await requestFrame(0);});const end=document.createElement('td');end.append(view);row.append(end);el('processingEpisodes').append(row);}el('processingPage').textContent=`${page+1} / ${Math.max(1,Math.ceil(result.total/25))}`;el('processingPrevPage').disabled=page===0;el('processingNextPage').disabled=(page+1)*25>=result.total;}
 function renderValues(){if(!frameData||!info)return;el('processingValues').replaceChildren();const values=frameData['observation.state'];for(let i=0;i<info.fields.length;i++){const d=info.fields[i],chosen=draft.state_fields.includes(d.name);if(!el('processingShowAll').checked&&!chosen)continue;const row=document.createElement('tr');row.append(text('td',d.name),text('td',number(Array.isArray(values)?values[i]:i===0?values:null)),text('td',d.unit),text('td',chosen?'✓':'—'));el('processingValues').append(row);}el('processingActions').replaceChildren();const actions=frameData.action;for(let i=0;i<info.action_names.length;i++){const row=document.createElement('tr');row.append(text('td',info.action_names[i]),text('td',number(Array.isArray(actions)?actions[i]:actions)),text('td',info.action_units?.[i]||'未声明'));el('processingActions').append(row);}for(const [key,value]of Object.entries(frameData)){if(key==='observation.state'||key==='action')continue;const row=document.createElement('tr');row.append(text('td',key),text('td',number(value)));el('processingActions').append(row);}}
 function frameStatus(message){el('processingFrameStatus').textContent=message;el('processingImages').setAttribute('aria-busy',String(Boolean(message)));}
 function requestFrame(value){
  if(episode==null)return Promise.resolve();
  frameNumber=Math.max(0,Math.min(length-1,Math.trunc(Number(value)||0)));
  el('processingTimeline').value=el('processingFrameNumber').value=String(frameNumber);transportState();
  pendingFrame={value:frameNumber,episode,id:draft.dataset_id};
  if(!frameLoop)frameLoop=(async()=>{
   while(pendingFrame){const next=pendingFrame;pendingFrame=null;
    if(next.episode===episode&&next.id===draft?.dataset_id)await showFrame(next.value);
   }
  })().finally(()=>{frameLoop=null;});
  return frameLoop;
 }
 async function showFrame(value){
  if(episode==null)return;
  const requested=Math.max(0,Math.min(length-1,Math.trunc(Number(value)||0))),ep=episode,seq=++frameSeq,cameras=[...info.cameras];
  frameAbort?.abort();frameAbort=new AbortController();const signal=frameAbort.signal;
  frameStatus(`正在加载 Episode ${ep} · 帧 ${requested}${frameData?'（仍显示上一帧）':''}`);
  const url=endpoint(`/episodes/${ep}/frames/${requested}`);
  async function fetchValue(path,image=false){
   const response=await fetch(path,{signal});
   if(!response.ok){const e=await response.json();throw Error(e.detail||response.statusText);}
   if(!image)return response.json();
   const objectURL=URL.createObjectURL(await response.blob()),img=new Image();
   try{img.src=objectURL;await img.decode();return img;}
   finally{URL.revokeObjectURL(objectURL);}
  }
  try{
   const [data,context]=await Promise.all([fetchValue(url),offlinePlots.windowFor(endpoint(`/episodes/${ep}`),requested,fetchValue)]);
   if(seq!==frameSeq)return;
   const images=await Promise.allSettled(cameras.map(c=>fetchValue(url+'/image/'+encodeURIComponent(c),true)));
   const result=[{status:'fulfilled',value:data},...images];
   if(seq!==frameSeq)return;
   if(result[0].status==='rejected')throw result[0].reason;
   // Keep stable camera cards and the previous decoded frame until all new
   // images are ready. Commit images and values together in this browser task.
   for(let i=0;i<cameras.length;i++){
    let card=el('processingImages').children[i];
    if(!card){card=document.createElement('div');card.className='offline-camera';const media=document.createElement('div');media.className='offline-camera-media';card.append(text('strong',cameras[i].replace('observation.images.','')),media);el('processingImages').append(card);}
    const value=result[i+1],media=card.querySelector('.offline-camera-media');
    if(value.status==='fulfilled'){value.value.alt=`${cameras[i]} · episode ${ep} · frame ${requested}`;media.replaceChildren(value.value);}
    else media.replaceChildren(text('p','图像不可用：'+value.reason.message));
   }
   error('','frame');frameData=result[0].value;renderValues();offlinePlots.commit(context,frameData);
   el('processingFrameTitle').textContent=`Episode ${ep} · 帧 ${requested} / ${length-1}`;
   el('processingTime').textContent=`${Number(frameData.timestamp).toFixed(3)} s`;
  }catch(e){
   if(e.name!=='AbortError'&&seq===frameSeq){
    stopPlayback();frameStatus(`帧 ${requested} 加载失败${frameData?'，保留上一帧图像与数值':''}`);error(e.message,'frame');
   }
  }finally{if(seq===frameSeq){el('processingImages').setAttribute('aria-busy','false');if(!el('processingFrameStatus').textContent.includes('加载失败'))frameStatus('');}}
 }
 function transportState(){
  const unavailable=episode==null||length<1;
  el('processingPrevFrame').disabled=unavailable||frameNumber<=0;
  el('processingNextFrame').disabled=unavailable||frameNumber>=length-1;
  for(const id of ['processingPlay','processingTimeline','processingFrameNumber'])el(id).disabled=unavailable;
 }
 function stopPlayback(){playing=false;playSeq++;clearTimeout(playTimer);el('processingPlay').textContent='播放';}
 async function tick(){if(!playing)return;const generation=playSeq,started=performance.now();if(frameNumber>=length-1){stopPlayback();return;}await requestFrame(Math.min(length-1,frameNumber+Math.max(1,Math.round(info.fps/10))));if(playing&&generation===playSeq)playTimer=setTimeout(tick,Math.max(0,100-(performance.now()-started)));}
 async function poll(){try{job=await api(base+'/job');const x=el('processingJob');x.textContent=job?`${job.stage||job.state}${job.total?` · ${job.done}/${job.total}`:''}${job.error?' · '+job.error:''}${job.output_root?'\n'+job.output_root:''}`:'尚无任务';const progress=el('processingProgress');if(job?.alive&&!job.total)progress.removeAttribute('value');else progress.value=job?.state==='succeeded'?1:job?.total?(job.done||0)/job.total:0;el('processingCancel').disabled=!job?.alive;el('processingResult').hidden=job?.state!=='succeeded';selectionLabel();error('','job');}catch(e){error(e.message,'job');} }
 async function activate(){el('processingBase').textContent=`相对路径基准：${document.getElementById('datasetBase').textContent.replace('相对路径基准：','')}`;if(!initialized){initialized=true;const version=++openVersion;try{const data=await refreshList();if(version!==openVersion)return;if(data.last){el('processingSource').value=data.last.source_root;const summary=await api(`${base}/datasets/${data.last.dataset_id}`);if(version===openVersion)await load(summary);}}catch(e){error(e.message);initialized=false;}}await poll();}
 window.addEventListener('vla-page-change',e=>{if(e.detail==='processing')activate();else{stopPlayback();pendingFrame=null;frameSeq++;frameAbort?.abort();frameStatus('');}});
 window.addEventListener('vla-live-status',()=>{const running=hardwareRunning();el('processingHardware').hidden=!running;el('processingHardware').textContent=running?'硬件会话仍在运行。可预览其他已完成数据集；转换前请返回监控并停止会话。':'';selectionLabel();});
 async function openDataset(path){const version=++openVersion;await writes;const summary=await api(base+'/open',{path});if(version===openVersion)await load(summary);}
 el('processingOpen').onclick=attempt(()=>openDataset(el('processingSource').value));
 el('processingRefresh').onclick=attempt(refreshList);
 for(const id of ['processingActionSpace','processingDerived','processingRepresentation','processingAnchor'])el(id).onchange=attempt(async()=>{if(!draft)return;draft.action_space=el('processingActionSpace').value||null;draft.allow_derived=el('processingDerived').checked;draft.preview_representation=el('processingRepresentation').value;draft.preview_anchor_frame=Number(el('processingAnchor').value);await persist();if(episode!==null)await requestFrame(frameNumber);});
 el('processingPreset').onchange=attempt(async()=>{if(info&&info.presets[el('processingPreset').value])await fieldsChanged([...info.presets[el('processingPreset').value]]);});
 el('processingAllFields').onclick=attempt(async()=>{if(info)await fieldsChanged(info.fields.map(d=>d.name));});el('processingNoFields').onclick=attempt(async()=>{if(info)await fieldsChanged([]);});
 el('processingOutput').onchange=attempt(persist);el('processingDefault').onclick=attempt(async()=>{if(draft){updateDefault(true);await persist();}});
 el('processingKeepAll').onclick=attempt(async()=>{if(!draft)return;draft.episodes=Array.from({length:info.total_episodes},(_,i)=>i);selectionLabel();await persist();await listEpisodes();});el('processingExcludeAll').onclick=attempt(async()=>{if(!draft)return;draft.episodes=[];selectionLabel();await persist();await listEpisodes();});
 el('processingPrevPage').onclick=attempt(async()=>{if(page>0){page--;await listEpisodes();}});el('processingNextPage').onclick=attempt(async()=>{if(info&&(page+1)*25<info.total_episodes){page++;await listEpisodes();}});
 el('processingPrevFrame').onclick=attempt(async()=>{stopPlayback();await requestFrame(frameNumber-1);});el('processingNextFrame').onclick=attempt(async()=>{stopPlayback();await requestFrame(frameNumber+1);});
 el('processingFrameNumber').onchange=attempt(async()=>{stopPlayback();await requestFrame(el('processingFrameNumber').value);});el('processingTimeline').oninput=()=>{stopPlayback();requestFrame(el('processingTimeline').value);};
 el('processingWindow').onchange=attempt(async()=>{stopPlayback();await requestFrame(frameNumber);});
 el('processingPlay').onclick=()=>{if(playing){stopPlayback();return;}if(episode==null)return;playing=true;el('processingPlay').textContent='暂停';tick();};el('processingShowAll').onchange=renderValues;
 el('processingExport').onclick=attempt(async()=>{if(!draft||exportStarting)return;exportStarting=true;selectionLabel();try{await persist();job=await api(base+'/jobs',{dataset_id:draft.dataset_id,revision:draft.revision});await poll();}finally{exportStarting=false;selectionLabel();}});el('processingCancel').onclick=attempt(async()=>{if(job)await api(`${base}/jobs/${job.job_id}/cancel`,{});await poll();});el('processingResult').onclick=attempt(async()=>{if(job?.state==='succeeded')await openDataset(job.output_root);});
 transportState();
 setInterval(()=>{if(!el('processingPanel').hidden)poll();},1500);
})();
