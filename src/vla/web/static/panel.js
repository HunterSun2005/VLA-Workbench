'use strict';
const $=id=>document.getElementById(id), form=$('setup');
let selectedPage='collect';
let token='', demo=false, workflow='collect', active=null, current=null, socket=null, connected=false;
let lastReceive=0, pending=false, creating=false, awaitingCommandId=null, session=null, descriptors=[], rows=[], plots=[], paused=false, epoch=0;
function sessionRunning(){
 if(active?.alive&&active.session_id!==current?.session_id)return true;
 if(current?.status?.final)return false;
 return Boolean(active?.alive)||['starting','standby','active','recording','saving','fault'].includes(current?.status?.state);
}
const configLocked=()=>creating||sessionRunning();
function restoreSessionConfig(saved){
 for(const [key,value] of Object.entries(saved)){
  const el=form.elements[key];if(!el)continue;
  if(el.type==='checkbox')el.checked=Boolean(value);
  else{if(el.tagName==='SELECT'&&value&&!Array.from(el.options).some(o=>o.value===value))el.add(new Option(key==='side_camera_serial'?value:value+' · 会话配置',value));el.value=value??'';}
 }
 workflow=saved.workflow;if(selectedPage!=='processing')selectedPage=workflow;
 const key=saved.workflow==='collect'?`collect/${saved.arms}/${saved.source}/${saved.schema}`:`deploy/${saved.arms}/${saved.policy_kind}`;
 lastDatasetDefault=datasetDefaults[key]?(datasetDefaults[key]+(saved.workflow==='collect'&&saved.side_camera_serial?'_side':'')):null;
}
function lockSetup(){
 const locked=configLocked();$('start').disabled=locked||pending;$('preflight').disabled=locked;$('datasetDefault').disabled=locked;
 for(const el of form.elements)if(el.name)el.disabled=locked||workflow==='deploy'&&el.name==='action_space';
 const owner=active?.alive&&active.session_id!==current?.session_id?active.config:current?.config||active?.config;
 $('collectTab').disabled=creating||sessionRunning()&&owner?.workflow!=='collect';
 $('deployTab').disabled=creating||sessionRunning()&&owner?.workflow!=='deploy';
}
function chartPalette(){const css=getComputedStyle(document.documentElement),value=name=>css.getPropertyValue(name).trim();return {axis:value('--axis'),grid:value('--grid'),series:Array.from({length:7},(_,i)=>value(`--series-${i}`))};}
let chartTheme=chartPalette();
window.addEventListener('vla-theme-change',()=>{chartTheme=chartPalette();for(const p of plots)p.plot.redraw(true,true);});
let datasetDefaults={}, lastDatasetDefault=null;
function updateDatasetPath(force=false){const c=config(),key=c.workflow==='collect'?`collect/${c.arms}/${c.source}/${c.schema}`:`deploy/${c.arms}/${c.policy_kind}`,next=datasetDefaults[key]?(datasetDefaults[key]+(c.workflow==='collect'&&c.side_camera_serial?'_side':'')):null;if(!next)return;const input=form.elements.dataset_root;if(force||!input.value.trim()||input.value===lastDatasetDefault)input.value=next;lastDatasetDefault=next;}
$('datasetDefault').onclick=()=>updateDatasetPath(true);
let selected; try {selected=JSON.parse(localStorage.getItem('vla.signals')||'null');} catch {selected=null;}
let signalsCustomized=Array.isArray(selected);if(!signalsCustomized)selected=null;
const labels={standby:'待命',active:'已启用',recording:'运行 / 记录中',saving:'保存中',fault:'故障',stopped:'已停止',starting:'启动中',unavailable:'等待会话'};
const groups={tcp:'TCP 位置',orientation:'TCP 旋转向量',joint:'关节位置',velocity:'关节速度',torque:'关节力矩',tcp_force:'TCP 外力（基座系）',tcp_torque:'TCP 外力矩（基座系）',gripper:'夹爪',health:'控制健康',target_tcp:'TCP 目标',target_joint:'关节目标',target_linear:'指令线速度',target_angular:'指令角速度'};
const issueMessages=new Map(), seenFailedCommands=new Set();
let copyFeedbackTimer;
async function copyDiagnosticText(text){
 try{if(navigator.clipboard?.writeText){await navigator.clipboard.writeText(text);return;}}catch{}
 // Older browsers and non-secure origins still get a user-initiated fallback.
 const previous=document.activeElement,area=document.createElement('textarea');
 area.className='clipboard-fallback';area.value=text;area.readOnly=true;document.body.append(area);
 try{area.focus({preventScroll:true});area.select();if(!document.execCommand('copy'))throw new Error('Clipboard unavailable');}
 finally{area.remove();previous?.focus({preventScroll:true});}
}
$('copyErrors').onclick=async()=>{
 const text=[...issueMessages.values()].map(issue=>issue.text).join('\n\n');if(!text)return;
 const button=$('copyErrors'),feedback=$('copyErrorsStatus');clearTimeout(copyFeedbackTimer);button.disabled=true;feedback.textContent='正在复制…';
 try{await copyDiagnosticText(text);feedback.textContent='已复制';}
 catch{feedback.textContent='复制失败，请手动选择错误文本';}
 finally{button.disabled=false;copyFeedbackTimer=setTimeout(()=>{feedback.textContent='';},4000);}
};
function drawIssues(){
 const box=$('errorItems');box.replaceChildren();$('error').hidden=issueMessages.size===0;
 for(const [key,issue] of issueMessages){const row=document.createElement('div'),text=document.createElement('span');row.className='issue-row';row.dataset.issue=key;text.textContent=issue.text;row.append(text);if(issue.dismiss){const close=document.createElement('button');close.type='button';close.textContent='关闭';close.setAttribute('aria-label','关闭错误提示');close.onclick=()=>setIssue(key,'');row.append(close);}box.append(row);}
}
function setIssue(key,text,dismiss=false){
 const previous=issueMessages.get(key);if(!text){if(issueMessages.delete(key))drawIssues();return;}
 if(previous?.text===text&&previous.dismiss===dismiss)return;issueMessages.set(key,{text,dismiss});drawIssues();
}
function error(text){setIssue('local',text,true);}
function renderIssues(message){
 const s=message.status||{};setIssue('business',s.error?`会话故障：${s.error}`:'');
 const monitors=message.monitor_errors;
 const monitorText=monitors?Object.entries(monitors).filter(([,v])=>v).map(([k,v])=>`${k==='telemetry'?'遥测':k.replace('camera.','相机 ')}：${v}`).join('\n'):message.monitor_error||'';
 setIssue('monitor',monitorText?'监控异常：\n'+monitorText:'');
 for(const failed of (message.commands||[]).filter(x=>x.status==='failed')){
  if(!seenFailedCommands.has(failed.command_id)){seenFailedCommands.add(failed.command_id);setIssue('command',`${failed.operation} 执行失败：${failed.error}`,true);}
 }
 while(seenFailedCommands.size>256)seenFailedCommands.delete(seenFailedCommands.values().next().value);
 $('sessionNotice').textContent=message.connection_message||'';$('sessionNotice').hidden=!message.connection_message;
}

function clearPreflight(){const panel=$('preflightResult');panel.hidden=true;panel.replaceChildren();}
function showPreflight(result, checkedWorkflow){
 const panel=$('preflightResult'),items=[...new Set(result.errors||[])];panel.replaceChildren();panel.hidden=false;panel.dataset.result=result.ok?'passed':'failed';
 const title=document.createElement('h2');title.textContent=`${checkedWorkflow==='deploy'?'推理部署':'数据采集'} · ${result.ok?'配置检查通过':`配置检查未通过（${items.length} 项）`}`;panel.append(title);
 if(result.ok){const output=document.createElement('p');output.className='preflight-output';output.textContent=`输出目录：${result.dataset_root}`;panel.append(output);}
 else{
  const list=document.createElement('ol');for(const message of items){const item=document.createElement('li');item.textContent=message;list.append(item);}panel.append(list);
  const hint=document.createElement('p');hint.className='hint';hint.textContent=checkedWorkflow==='deploy'?'修正以上配置后重新检查。若只需遥操作采集，请切换到「数据采集」，无需填写模型或 tokenizer。':'修正以上配置后重新检查；续采需选择字段一致的数据集，新格式采集请使用新目录。';panel.append(hint);
 }
 panel.scrollIntoView({behavior:'smooth',block:'center'});
}
form.addEventListener('input',clearPreflight);form.addEventListener('change',clearPreflight);
async function api(path,data){const response=await fetch(path,{method:data?'POST':'GET',headers:data?{'Content-Type':'application/json','X-VLA-Token':token}:{},body:data?JSON.stringify(data):undefined});let result=await response.json();if(!response.ok)throw new Error(result.detail||response.statusText);return result;}
function config(){const data={workflow,mode:'vla',device:'cuda'};for(const element of form.elements){if(element.name)data[element.name]=element.type==='checkbox'?element.checked:element.value;}if(data.workflow==='deploy'){data.side_camera_serial='';data.schema='full';if(data.policy_kind==='sft')data.mode='vla';}if(data.arms==='dual'||data.source==='gello')data.schema='full';if(window.readWorkspaceBounds)data.workspace_bounds=readWorkspaceBounds(data);if(window.readCameraSerials)data.camera_serials=readCameraSerials(data);return data;}
function configureView(){const deploy=workflow==='deploy';$('sourceLabel').textContent=deploy?'人工接管输入源':'遥操作输入源';$('collectTab').classList.toggle('selected',selectedPage==='collect');$('deployTab').classList.toggle('selected',selectedPage==='deploy');$('processingTab').classList.toggle('selected',selectedPage==='processing');$('modelFields').hidden=!deploy;$('policy').hidden=!deploy;$('schemaField').hidden=deploy;$('sideCameraField').hidden=deploy;form.elements.schema.options[0].textContent=`完整状态 · ${form.elements.arms.value==='dual'?'100':'50'}D`;form.elements.action_space.disabled=deploy;$('configPathLabel').textContent=workflow==='collect'&&form.elements.source.value==='gello'?'姿态恢复 YAML（工作空间与运动参数）':'采集／部署 YAML';updateDatasetPath();lockSetup();window.dispatchEvent(new CustomEvent('vla-config-change'));}
function showPage(page){if(page!==selectedPage)clearPreflight();selectedPage=page;const processing=page==='processing';$('livePanel').hidden=processing;$('processingPanel').hidden=!processing;if(processing)stopCameras();else{workflow=page;if(session&&!current?.status?.final&&current?.status?.state!=='stopped')cameras(current?.config||active?.config);}configureView();window.dispatchEvent(new CustomEvent('vla-page-change',{detail:page}));}
$('collectTab').onclick=()=>showPage('collect');$('deployTab').onclick=()=>showPage('deploy');$('processingTab').onclick=()=>showPage('processing');form.elements.arms.onchange=configureView;form.elements.source.onchange=configureView;form.elements.policy_kind.onchange=configureView;
$('preflight').onclick=async()=>{clearPreflight();try{error('');await window.prepareHardware?.();const data=config(),result=await api('/api/preflight',data);showPreflight(result,data.workflow);}catch(e){showPreflight({ok:false,errors:[e.message]},workflow);}};
form.onsubmit=async event=>{event.preventDefault();if(creating||pending)return;creating=true;lockSetup();window.dispatchEvent(new CustomEvent('vla-live-status'));try{error('');clearPreflight();await window.prepareHardware?.();const data=config(),check=await api('/api/preflight',data);showPreflight(check,data.workflow);if(!check.ok)return;active={...await api('/api/sessions',data),alive:true};$('start').disabled=true;error('');}catch(e){error(e.message);}finally{creating=false;lockSetup();window.dispatchEvent(new CustomEvent('vla-live-status'));}};
async function command(operation,args={}){if(pending||!current?.session_id)return;pending=true;awaitingCommandId=crypto.randomUUID();freshness();error('');try{await api('/api/commands',{session_id:current.session_id,revision:current.revision,command_id:awaitingCommandId,operation,args});}catch(e){awaitingCommandId=null;pending=false;freshness();error(e.message);}}
document.querySelectorAll('[data-op]').forEach(button=>button.onclick=()=>{const op=button.dataset.op;if(op==='discard'&&!confirm('丢弃当前未保存回合？'))return;command(op,{outcome:button.dataset.outcome||'success'});});
document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=()=>command('mode',{mode:button.dataset.mode}));
function stopCameras(){window.releasePreview?.();for(const img of document.querySelectorAll('.camera img'))img.removeAttribute('src');}
function cameras(config){if(selectedPage==='processing')return;stopCameras();const roles=(config?.arms==='dual'?['base','left_wrist','right_wrist']:['base','left_wrist']).concat(config?.workflow==='collect'&&config?.side_camera_serial?['side']:[]);$('cameras').replaceChildren();for(const role of roles){const card=document.createElement('div');card.className='camera stale';card.dataset.role=role;const img=document.createElement('img');img.alt=role;img.dataset.previewSrc=`/api/cameras/${role}?session=${session}`;const title=document.createElement('span');title.className='name';title.textContent=role==='base'?'BASE · 全局':role==='left_wrist'?'LEFT · 左腕':role==='side'?'SIDE · 侧视':'RIGHT · 右腕';const age=document.createElement('span');age.className='age';age.textContent='等待相机';card.append(img,title,age);$('cameras').append(card);}window.startPreviewImages?.();}
function rebuildSignals(){if(!signalsCustomized)selected=descriptors.filter(x=>x.default).map(x=>x.key);$('signals').replaceChildren();const byGroup={};for(const d of descriptors){const group=`${d.arm}.${d.group}.${d.unit}`;(byGroup[group]??=[]).push(d);}for(const ds of Object.values(byGroup)){const block=document.createElement('div'),title=document.createElement('h3');title.textContent=`${ds[0].arm==='left'?'左臂':'右臂'} · ${groups[ds[0].group]||ds[0].group} [${ds[0].unit}]`;block.append(title);for(const d of ds){const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=selected.includes(d.key);check.onchange=()=>{signalsCustomized=true;selected=selected.filter(k=>k!==d.key);if(check.checked)selected.push(d.key);localStorage.setItem('vla.signals',JSON.stringify(selected));rebuildPlots();};label.append(check,document.createTextNode(d.label));block.append(label);}$('signals').append(block);}rebuildPlots();}
function rebuildPlots(){for(const p of plots)p.plot.destroy();plots=[];$('plots').replaceChildren();const byGroup={};for(const d of descriptors.filter(x=>selected?.includes(x.key))){const k=`${d.arm}.${d.group}.${d.unit}`;(byGroup[k]??=[]).push(d);}for(const ds of Object.values(byGroup)){const container=document.createElement('div');container.className='plot';const title=document.createElement('h3');title.textContent=`${ds[0].arm==='left'?'左臂':'右臂'} · ${groups[ds[0].group]||ds[0].group} · ${ds[0].unit} · ${ds[0].source==='target'?'目标':ds[0].source==='estimated'?'估计':'实测'}`;container.append(title);$('plots').append(container);const plot=new uPlot({width:Math.max(250,container.clientWidth-12),height:175,series:[{},...ds.map((d,i)=>({label:d.label,stroke:()=>chartTheme.series[i%chartTheme.series.length],width:1.5,spanGaps:false}))],axes:[{values:(_u,ticks)=>ticks.map(t=>new Date(t*1000).toLocaleTimeString('zh-CN',{hour12:false})),stroke:()=>chartTheme.axis,grid:{stroke:()=>chartTheme.grid},ticks:{stroke:()=>chartTheme.grid},font:'10px system-ui'},{stroke:()=>chartTheme.axis,grid:{stroke:()=>chartTheme.grid},ticks:{stroke:()=>chartTheme.grid},font:'10px system-ui',size:53}],scales:{x:{time:true}},cursor:{drag:{x:true,y:false}},legend:{show:true}},[[],...ds.map(()=>[])],container);plots.push({plot,ds,container});}$('emptyPlots').hidden=plots.length>0;draw();}
function draw(){if(selectedPage==='processing'||paused||!rows.length)return;const end=rows[rows.length-1].time,begin=end-Number($('window').value),visible=rows.filter(x=>x.time>=begin);for(const p of plots){p.plot.setData([visible.map(x=>x.time),...p.ds.map(d=>visible.map(x=>x.values[d.key]??null))],false);p.plot.setScale('x',{min:begin,max:end});}}
$('hideAll').onclick=()=>{signalsCustomized=true;selected=[];localStorage.setItem('vla.signals','[]');rebuildSignals();};$('defaults').onclick=()=>{signalsCustomized=false;selected=null;localStorage.removeItem('vla.signals');rebuildSignals();};$('freeze').onclick=()=>{paused=!paused;$('freeze').textContent=paused?'恢复跟随':'暂停图表';draw();};try{const saved=localStorage.getItem('vla.window');if(['10','60','300'].includes(saved))$('window').value=saved;}catch{}$('window').onchange=()=>{localStorage.setItem('vla.window',$('window').value);draw();};
new ResizeObserver(()=>{for(const p of plots)p.plot.setSize({width:Math.max(250,p.container.clientWidth-12),height:175});}).observe($('plots'));
function render(message){current=message;if(Object.hasOwn(message,'active'))active=message.active;lastReceive=performance.now();const s=message.status||{};
 if(message.session_id&&session!==message.session_id){issueMessages.clear();seenFailedCommands.clear();drawIssues();render.lastState=null;session=message.session_id;awaitingCommandId=null;pending=false;$('sessionConfig').open=false;rows=[];epoch=Date.now()/1000-message.now;descriptors=[];if(!s.final&&s.state!=='stopped')cameras(message.config||active?.config);else stopCameras();if(message.config){restoreSessionConfig(message.config);configureView();}}

 $('state').textContent=labels[s.state]||s.state;$('episode').textContent=s.episode??'—';$('frames').textContent=`${s.frames??'—'} / ${s.elapsed_s?.toFixed(1)??'—'}s`;$('source').textContent=({pose_return:'恢复姿态',waiting:'等待新动作',idle:'待命',human:'HUMAN',gamepad:'GAMEPAD'}[s.actual_source]||s.actual_source?.toUpperCase()||'—');
 $('modeState').textContent=s.switch_pending?'等待新策略动作':s.requested_mode?`已选择 ${s.requested_mode.toUpperCase()}`:'—';$('probability').textContent=s.probability==null?'—':s.probability.toFixed(3);$('actionContract').textContent=s.action_space?`${s.action_space.toUpperCase()} · ${s.training_representation||'absolute'} · ROS ${s.controller_state||''}`:'动作契约：等待会话';$('latency').textContent=`${s.inference_ms?.toFixed(0)??'—'} ms / ${s.rtc?'ON':'OFF'}`;
 const annotation=$('annotationStatus');annotation.hidden=(message.config||active?.config)?.workflow!=='collect';
 annotation.textContent=s.critical==null?'当前采集后端不提供 Critical 标注':`Critical：${s.critical?'ON · 正在标注':s.critical_state==='after'?'已结束（本回合不可再次开启）':'OFF'}`;
 if(s.subtask)annotation.textContent+=` · 阶段 ${s.subtask.index+1}/${s.subtask.total}：${s.subtask.name}`;
 $('armStatus').replaceChildren();for(const [arm,value] of Object.entries(s.arms||{})){const badge=document.createElement('span');badge.textContent=`${arm==='left'?'左臂':'右臂'} · ${value.source} · ${value.speed==='fast'?'Fast':value.speed==='normal'?'Normal':value.speed==='fixed'?'固定':'不可用'}${value.speed_scale==null?'':` ${value.speed_scale.toFixed(2)}×`}${value.transition?' · 过渡中':''}`;$('armStatus').append(badge);}
 const commands=message.commands||[];const ack=commands.find(x=>x.command_id===awaitingCommandId);if(ack&&ack.status!=='accepted')awaitingCommandId=null;pending=Boolean(awaitingCommandId)||commands.some(x=>x.status==='accepted');renderIssues(message);
 $('events').textContent=[...commands.map(x=>`${x.operation}: ${x.status}${x.error?' — '+x.error:''}`),...(s.events||[])].join('\n')||'暂无命令';
 if((s.state==='stopped'||s.final)&&render.lastState!=='stopped'){stopCameras();$('sessionConfig').open=true;}render.lastState=s.state;
 lockSetup();
 if(s.signals&&JSON.stringify(s.signals)!==JSON.stringify(descriptors)){descriptors=s.signals;rebuildSignals();}
 for(const sample of message.samples||[]){const time=epoch+sample.time;if(!rows.length||time>rows.at(-1).time)rows.push({time,values:sample.values});}if(rows.length>6000)rows.splice(0,rows.length-6000);draw();freshness();window.dispatchEvent(new CustomEvent('vla-live-status'));}
function operationBlock(op,s){
 if(!(s.available_commands||[]).includes(op))return '当前后端不提供此操作';
 if(op==='pause')return '';
 if(s.saving)return '正在保存，请等待完成';
 if(op==='enable')return s.state==='fault'?'请先处理故障':s.enabled?'控制已经启用':'';
 if(op==='record')return s.recording?'回合已经开始':!s.enabled?'请先启用控制':s.state==='fault'?'请先处理故障':'';
 if(['save','discard','critical','next_subtask','intervention'].includes(op)&&!s.recording)return '请先开始回合';
 if(op==='critical'&&s.critical_state==='after')return '本回合 Critical 区间已结束';
 if(op==='next_subtask'&&s.subtask?.index>=s.subtask?.total-1)return '已经是最后阶段';
 if(op==='stop'&&s.recording)return '请先保存或丢弃当前回合';
 if(op==='recover'&&s.recording)return '请先结束当前回合';
 return '';
}
function freshness(){
 const elapsed=(performance.now()-lastReceive)/1000,s=current?.status||{},isLive=connected&&elapsed<.5&&s.updated_at!=null&&current.now-s.updated_at+elapsed<.5&&!s.final;
 $('connection').textContent=connected?'已连接':'连接中断 · 自动重连';$('dot').classList.toggle('live',connected);$('freshness').textContent=isLive?'实时更新':s.final?'会话已结束':'数据过期 / 等待';
 for(const b of document.querySelectorAll('[data-op]')){
  const startupStop=connected&&elapsed<.5&&s.state==='starting'&&b.dataset.op==='stop';
  const reason=!(isLive||startupStop)?'等待实时会话连接':pending?'正在执行上一条命令':b.dataset.outcome==='failure'&&!(s.outcomes||[]).includes('failure')?'此采集后端不支持失败标签':operationBlock(b.dataset.op,s);
  b.disabled=Boolean(reason);b.title=reason|| (b.dataset.op==='pause'?'立即软件暂停；当前未保存回合可能被丢弃':'');
  if(b.dataset.op==='critical'){b.textContent=s.critical?'Critical · ON':s.critical_state==='after'?'Critical · 已结束':'Critical · OFF';b.classList.toggle('selected',Boolean(s.critical));b.setAttribute('aria-pressed',String(Boolean(s.critical)));}
 }
 for(const b of document.querySelectorAll('[data-mode]')){
  const supported=(s.modes||[]).includes(b.dataset.mode);
  b.disabled=!isLive||pending||!supported;b.title=!supported?(b.dataset.mode==='auto'?'自动模式需要兼容的分类头资源':b.dataset.mode==='rlt'?'RLT 模式需要兼容的 Actor／RLToken 资源':'启动部署会话后可用'):!isLive?'等待实时会话连接':'';
  b.classList.toggle('selected',s.requested_mode===b.dataset.mode);
 }
 for(const card of document.querySelectorAll('.camera')){if(!card.querySelector('img').getAttribute('src')){card.classList.add('stale');card.querySelector('.age').textContent='预览暂停 · 遥测继续更新';continue;}const frame=current?.cameras?.[card.dataset.role];const age=frame?current.now-frame.captured_at+elapsed:Infinity;card.classList.toggle('stale',!connected||age>.5||age<0);card.querySelector('.age').textContent=Number.isFinite(age)?`${age>.5||!connected?'画面过期':'LIVE'} · ${(age*1000).toFixed(0)} ms · 服务端预览 ${age<=.5&&connected?(frame?.fps?.toFixed(1)??'—'):'—'} FPS`:'等待相机 / 无画面';}
}
async function connect(){try{const boot=await api('/api/bootstrap');token=boot.token;demo=boot.demo;active=boot.active;datasetDefaults=boot.dataset_defaults||{};$('datasetBase').textContent=`相对路径基准：${boot.workspace_root}`;$('demo').hidden=!demo;if(!form.elements.model_python.value)form.elements.model_python.value=boot.defaults.model_python;if(active?.config&&!session)restoreSessionConfig(active.config);configureView();socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/live`,['vla',token]);socket.onopen=()=>{connected=true;setIssue('web-connection','');};socket.onmessage=event=>render(JSON.parse(event.data));socket.onclose=()=>{connected=false;freshness();setTimeout(connect,1200);};socket.onerror=()=>socket.close();}catch(e){connected=false;setIssue('web-connection',`Web 连接失败：${e.message}`);setTimeout(connect,1500);}}
configureView();setInterval(freshness,250);connect();
