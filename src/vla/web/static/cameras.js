/* RealSense SDK camera selection for collection. */
'use strict';
(() => {
 const panel=$('cameraSetup'),fields=$('cameraFields'),status=$('cameraStatus');
 const roles={base:'Base · 全局相机',left_wrist:'Left wrist · 左腕相机'};
 let inventory=[],inventoryWarnings=[],defaults={},values={},key=null,sequence=0,restored=null,detected=false;
 const applicable=c=>c.workflow==='collect'&&c.arms==='single_left'&&c.source==='gamepad';
 const running=configLocked;
 window.readCameraSerials=c=>applicable(c)&&Object.keys(values).length?{...values}:null;
 function lock(){panel.querySelectorAll('input,select,button').forEach(el=>el.disabled=running());}
 function bindings(){
  const missing=[];
  for(const [role,title] of Object.entries(roles)){
   const serial=values[role]||defaults[role],found=Boolean(cameraIdentity(inventory,serial)?.value),info=$('camera-binding-'+role);
   if(!info)continue;
   const source=values[role]?'已选择':'YAML 配置默认（不是自动检测）';
   info.textContent=serial?`${source}：${cameraLabel(inventory,serial)} · ${!detected?'尚未检测':found?'已检测到设备':'未检测到，请从列表选择实际设备'}`:'正在读取配置默认值…';
   info.dataset.missing=String(detected&&Boolean(serial)&&!found);
   if(detected&&serial&&!found)missing.push(title+'：'+cameraLabel(inventory,serial));
  }
  if(detected)status.textContent=`检测到 ${inventory.length} 台 RealSense。`+(missing.length?'以下配置设备未检测到：'+missing.join('；')+'。':'所选设备均已检测到。')+(inventoryWarnings.length?' '+inventoryWarnings.join('；'):'')+' 设备检测无法判断安装视角，请按实际位置分配 Base／左腕。';
 }
 function draw(){
  fields.replaceChildren();
  for(const [role,title] of Object.entries(roles)){
   const label=document.createElement('label'),input=document.createElement('input'),select=document.createElement('select');
   input.id='camera-'+role;input.value=cameraLabel(inventory,values[role]||'');input.placeholder='YAML 默认：'+(cameraLabel(inventory,defaults[role])||'读取中');input.setAttribute('aria-label',title+' RealSense SDK 序列号');input.maxLength=128;
   select.id='camera-device-'+role;select.setAttribute('aria-label',title+'设备选择');select.add(new Option('使用 YAML 默认'+(defaults[role]?' · '+cameraLabel(inventory,defaults[role])+(detected&&!inventory.some(item=>item.value===defaults[role])?'（未检测到）':''):''),''));
   for(const item of inventory)select.add(cameraOption(item));
   if(values[role]&&!inventory.some(x=>x.value===values[role])){const option=new Option(values[role],values[role]);option.title='当前未检测到该设备';select.add(option);}
   select.value=values[role]||'';
   input.oninput=()=>{const value=cameraValue(inventory,input.value.trim());if(value)values[role]=value;else delete values[role];select.value=value;clearPreflight();bindings();};
   select.onchange=()=>{input.value=cameraLabel(inventory,select.value);if(select.value)values[role]=select.value;else delete values[role];clearPreflight();bindings();};
   const info=document.createElement('small');info.id='camera-binding-'+role;info.className='camera-binding-status';info.setAttribute('role','status');label.append(document.createTextNode(title),input,select,info);fields.append(label);
  }lock();bindings();
 }
 async function view(force=false){
  const c=config();panel.hidden=!applicable(c);if(panel.hidden)return;
  const saved=current?.config||active?.config,id=current?.session_id||active?.session_id;
  if(id&&id!==restored&&saved&&applicable(saved)){restored=id;values={...(saved.camera_serials||{})};key=null;}
  if(!token){draw();return;}
  if(!force&&key===c.config_path){lock();return;}key=c.config_path;
  const seq=++sequence;detected=false;draw();status.textContent='正在读取相机配置…';
  try{
   const [hardware,result]=await Promise.all([api('/api/hardware'),api('/api/camera-defaults',{task:'camera defaults',workflow:c.workflow,arms:c.arms,source:c.source,config_path:c.config_path})]);
   if(seq!==sequence)return;inventory=hardware.inventory.cameras;inventoryWarnings=hardware.inventory.warnings||[];for(const role of Object.keys(values))values[role]=cameraValue(inventory,values[role]);defaults=result.serials;detected=true;draw();
  }catch(e){if(seq===sequence){status.textContent='读取相机配置失败：'+e.message;key=null;}}
 }
 $('cameraRefresh').onclick=()=>view(true);
 $('cameraReset').onclick=()=>{values={};draw();clearPreflight();};
 form.elements.config_path.addEventListener('change',()=>view(true));
 window.addEventListener('vla-config-change',()=>view());
 window.addEventListener('vla-live-status',lock);
 // The bootstrap request may finish before this script, so force an initial read.
 view(true);
})();
