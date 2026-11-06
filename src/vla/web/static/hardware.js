/* Passive discovery and editable, per-arm profiles. No device control here. */
'use strict';
(()=>{
 const panel=$('hardwareSetup'),mode=$('hardwareMode'),fields=$('hardwareFields');
 const side=$('sideCamera');
 let profiles=null,inventory={serials:[],cameras:[],warnings:[]},drafts={},shown=null,saving=false,sessionProfile=null,profileSession=null,profileSequence=0,wasRunning=false,restoredMode=null;
 const labels={LEFT_ROBOT_IP:'左臂 IP',RIGHT_ROBOT_IP:'右臂 IP',LEFT_ROBOTIQ_PORT:'左夹爪串口',RIGHT_ROBOTIQ_PORT:'右夹爪串口',LEFT_GELLO_SERIAL_ID:'左 GELLO 串口',RIGHT_GELLO_SERIAL_ID:'右 GELLO 串口',BASE_CAMERA_SERIAL:'Base 全局相机',LEFT_WRIST_CAMERA_SERIAL:'左腕相机',RIGHT_WRIST_CAMERA_SERIAL:'右腕相机',GELLO_CONFIG:'GELLO 校准 YAML',ROBOT_CONFIG:'机械臂 YAML',GRIPPER_CONFIG:'夹爪 YAML'};
 const usesROS=()=>true;
 const running=configLocked;
 const generated=path=>/^runtime\/web\/hardware\/(single_left|dual)-[a-f0-9]{64}\.json$/.test(path);
 const displayedValues=()=>sessionRunning()?(sessionProfile?.session_id===(current?.session_id||active?.session_id)&&sessionProfile.arms===shown?sessionProfile.values:null):drafts[shown];
 const normalize=value=>value.replace(/^\/dev\/serial\/by-id\//,'');
 function choices(key){return key.endsWith('_CAMERA_SERIAL')?inventory.cameras:key.endsWith('_PORT')||key.endsWith('_SERIAL_ID')?inventory.serials:null;}
 function notice(){
  const missing=[];
  for(const [key,value] of Object.entries(displayedValues()||{})){
   if(key.endsWith('_CAMERA_SERIAL')){const info=$('hardware-camera-status-'+key),found=inventory.cameras.some(item=>item.value===value);if(info){info.textContent=`RealSense SDK：${cameraLabel(inventory.cameras,value)||'未填写'} · ${found?'已检测到设备':'未检测到，请按实际安装位置选择设备'}`;info.dataset.missing=String(!found);}}
   const items=choices(key);if(items&&!items.some(x=>normalize(x.value)===normalize(value)))missing.push(labels[key]+(key.endsWith('_CAMERA_SERIAL')?'（'+cameraLabel(inventory.cameras,value)+'）':''));
  }
  $('hardwareNotice').textContent=[missing.length?'未检测到所填设备：'+missing.join('、')+'。保留原值，请从列表选择或检查接线。':'',...(inventory.warnings||[]),inventory.note||'',profiles?.[shown]?.warning||''].filter(Boolean).join('\n');
 }
 function renderSide(){
  const selected=cameraValue(inventory.cameras,side.value);
  side.replaceChildren(new Option('不添加 · 标准 base + wrist',''));
  for(const camera of inventory.cameras)side.add(cameraOption(camera));
  if(selected&&!inventory.cameras.some(c=>c.value===selected)){const option=new Option(selected,selected);option.title='当前未检测到该设备';side.add(option);}
  side.value=selected;
 }
 side.onchange=()=>{updateDatasetPath();clearPreflight();};
 $('sideCameraRefresh').onclick=async()=>{try{const result=await api('/api/hardware');inventory=result.inventory;renderSide();renderFields();$('sideCameraNotice').textContent=inventory.cameras.length?'选中后额外预览并保存 side 视角；不得与 base / wrist 使用同一设备。':'当前未检测到 RealSense 摄像头，请检查连接后刷新。';}catch(e){error(e.message);}};
 function lock(){
  const disabled=running()||saving;
  side.disabled=disabled;$('sideCameraRefresh').disabled=disabled;
  panel.querySelectorAll('button,input,select').forEach(el=>el.disabled=disabled);
  // The named path is locked independently of the helper controls.
  if(usesROS())form.elements.ros_config.readOnly=mode.value==='preset';else form.elements.ros_config.readOnly=false;
 }
 function renderFields(){
  fields.replaceChildren();
  const values=displayedValues();if(!values)return;
  for(const group of [['左臂',k=>k.startsWith('LEFT_')&&!k.includes('CAMERA')],['右臂',k=>k.startsWith('RIGHT_')&&!k.includes('CAMERA')],['相机角色',k=>k.includes('CAMERA')],['校准与启动配置',k=>k.endsWith('_CONFIG')]]){
   const keys=Object.keys(values).filter(group[1]);if(!keys.length)continue;
   for(const key of keys)if(key.endsWith('_CAMERA_SERIAL')&&!sessionRunning())values[key]=cameraValue(inventory.cameras,values[key]);
   const detail=document.createElement('details'),summary=document.createElement('summary');detail.open=group[0]==='相机角色';summary.textContent=group[0];detail.append(summary);
   for(const key of keys){
    const label=document.createElement('label'),input=document.createElement('input');input.id='hardware-'+key;input.value=key.endsWith('_CAMERA_SERIAL')?cameraLabel(inventory.cameras,values[key]):values[key];input.setAttribute('aria-label',labels[key]);label.append(document.createTextNode(labels[key]),input);
    const items=choices(key);
    if(items){
     const select=document.createElement('select');select.id='devices-'+key;select.setAttribute('aria-label',labels[key]+' · 检测设备');
     select.add(new Option('从检测设备选择（或在上方填写）',''));
     for(const item of items)select.add(key.endsWith('_CAMERA_SERIAL')?cameraOption(item):new Option(item.label+(item.accessible===false?' · 无读写权限':''),item.value));
     const matched=items.find(x=>normalize(x.value)===normalize(values[key]));if(matched)select.value=matched.value;
     select.onchange=()=>{if(!select.value)return;input.value=key.endsWith('_CAMERA_SERIAL')?cameraLabel(inventory.cameras,select.value):select.value;input.dispatchEvent(new Event('input'));};
     label.append(select);
     input.addEventListener('input',()=>{select.value=items.find(x=>normalize(x.value)===normalize(key.endsWith('_CAMERA_SERIAL')?cameraValue(inventory.cameras,input.value.trim()):input.value.trim()))?.value||'';});
    }
    input.addEventListener('input',()=>{values[key]=key.endsWith('_CAMERA_SERIAL')?cameraValue(inventory.cameras,input.value.trim()):input.value.trim();$('hardwareStatus').textContent='修改待保存；检查或启动时自动保存。';notice();});
    if(key.endsWith('_CAMERA_SERIAL')){const info=document.createElement('small');info.id='hardware-camera-status-'+key;info.className='camera-binding-status';info.setAttribute('role','status');label.append(info);}
    detail.append(label);
   }
   fields.append(detail);
  }
  notice();lock();
 }
 function view(){
  renderSide();
  panel.hidden=!usesROS();
  const path=form.elements.ros_config.value.trim(),id=current?.session_id||active?.session_id;
  if(id&&id!==restoredMode){restoredMode=id;mode.value=path&&!generated(path)?'custom':'preset';}
  if(path&&!generated(path))mode.value='custom';
  $('hardwareEditor').hidden=mode.value==='custom'&&!(sessionRunning()&&sessionProfile);
  if(mode.value==='custom')$('hardwareStatus').textContent='请在下方「环境与 ROS 配置」填写 JSON 路径；不会覆盖该文件。';
  const arms=config().arms;if(profiles&&shown!==arms){shown=arms;renderFields();}
  lock();
  syncSessionProfile();
 }
 async function syncSessionProfile(){
  const id=current?.session_id||active?.session_id;
  if(!id||!sessionRunning()||!usesROS()||profileSession===id)return;
  profileSession=id;sessionProfile=null;fields.replaceChildren();const seq=++profileSequence;
  $('hardwareStatus').textContent='正在读取当前会话实际使用的硬件配置…';
  try{const result=await api('/api/hardware/session');if(seq!==profileSequence)return;if(!result){$('hardwareStatus').textContent='当前会话未指定 ROS 硬件配置（模拟模式允许）';return;}if(result.session_id!==id)return;sessionProfile=result;drafts[result.arms]={...(profiles?.[result.arms]?.defaults||{}),...result.values};renderFields();$('hardwareEditor').hidden=false;$('hardwareStatus').textContent='当前会话实际使用的硬件配置（只读，来自启动快照）';}
  catch(e){if(seq===profileSequence){fields.replaceChildren();$('hardwareStatus').textContent='无法读取当前会话硬件配置：'+e.message;}}
 }
 const ready=api('/api/hardware').then(result=>{
  profiles=result.profiles;inventory=result.inventory;renderSide();
  for(const arms of Object.keys(profiles))drafts[arms]={...profiles[arms].values};
  view();
 }).catch(e=>{$('hardwareStatus').textContent='配置加载失败：'+e.message;});
 async function save(){
  await ready;if(!profiles)throw new Error('硬件预设加载失败，请刷新页面或使用自定义 JSON');
  if(saving)throw new Error('硬件配置正在保存');
  const c=config(),values={...drafts[c.arms]};saving=true;lock();
  try{
   const result=await api('/api/hardware',{arms:c.arms,values});
   if(config().arms!==c.arms||config().source!==c.source||config().workflow!==c.workflow)throw new Error('配置已切换，请重新检查');
   form.elements.ros_config.value=result.path;
   $('hardwareStatus').textContent='已保存：'+(c.arms==='dual'?'双臂':'单左臂')+'配置，JSON 路径已填入下方环境设置。';
   $('hardwareStatus').title=result.path;
   return result;
  }finally{saving=false;lock();}
 }
 window.prepareHardware=async()=>{if(usesROS()&&mode.value==='preset'&&!sessionRunning())await save();};
 $('hardwareSave').onclick=()=>{error('');save().catch(e=>error(e.message));};
 $('hardwareRefresh').onclick=async()=>{try{const result=await api('/api/hardware');inventory=result.inventory;renderSide();renderFields();}catch(e){error(e.message);}};
 $('hardwareReset').onclick=()=>{if(!profiles||!confirm('恢复当前单双臂的仓库预设？未保存的设备选择将被替换。'))return;drafts[shown]={...profiles[shown].defaults};renderFields();$('hardwareStatus').textContent='已恢复预设，待保存。';};
 mode.onchange=()=>{if(mode.value==='preset'){form.elements.ros_config.value='';$('hardwareStatus').textContent='检查或启动时自动保存预设。';}view();};
 form.elements.ros_config.addEventListener('input',()=>{mode.value='custom';view();});
 window.addEventListener('vla-config-change',view);window.addEventListener('vla-live-status',()=>{const now=sessionRunning();if(wasRunning&&!now){renderFields();view();}wasRunning=now;lock();syncSessionProfile();});
 view();
})();
