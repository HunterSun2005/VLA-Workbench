/* Franky collection owns its cameras independently of the ROS hardware profile. */
'use strict';
(() => {
 const panel=$('cameraSetup'),fields=$('cameraFields'),status=$('cameraStatus');
 const roles={base:'Base · 全局相机',left_wrist:'Left wrist · 左腕相机'};
 let inventory=[],defaults={},values={},key=null,sequence=0,restored=null;
 const applicable=c=>c.workflow==='collect'&&c.arms==='single_left'&&c.source==='gamepad';
 const running=configLocked;
 window.readCameraSerials=c=>applicable(c)&&Object.keys(values).length?{...values}:null;
 function lock(){panel.querySelectorAll('input,select,button').forEach(el=>el.disabled=running());}
 function draw(){
  fields.replaceChildren();
  for(const [role,title] of Object.entries(roles)){
   const label=document.createElement('label'),input=document.createElement('input'),select=document.createElement('select');
   input.id='camera-'+role;input.value=values[role]||'';input.placeholder='YAML 默认：'+(defaults[role]||'读取中');input.setAttribute('aria-label',title+'序列号');input.maxLength=128;
   select.id='camera-device-'+role;select.setAttribute('aria-label',title+'设备选择');select.add(new Option('使用 YAML 默认'+(defaults[role]?' · '+defaults[role]:''),''));
   for(const item of inventory)select.add(new Option(item.value,item.value));
   if(values[role]&&!inventory.some(x=>x.value===values[role])){const option=new Option(values[role],values[role]);option.title='当前未检测到该设备';select.add(option);}
   select.value=values[role]||'';
   input.oninput=()=>{const value=input.value.trim();if(value)values[role]=value;else delete values[role];select.value=value;clearPreflight();};
   select.onchange=()=>{input.value=select.value;if(select.value)values[role]=select.value;else delete values[role];clearPreflight();};
   label.append(document.createTextNode(title),input,select);fields.append(label);
  }lock();
 }
 async function view(force=false){
  const c=config();panel.hidden=!applicable(c);if(panel.hidden)return;
  const saved=current?.config||active?.config,id=current?.session_id||active?.session_id;
  if(id&&id!==restored&&saved&&applicable(saved)){restored=id;values={...(saved.camera_serials||{})};key=null;}
  if(!token){draw();return;}
  if(!force&&key===c.config_path){lock();return;}key=c.config_path;
  const seq=++sequence;draw();status.textContent='正在读取相机配置…';
  try{
   const [hardware,result]=await Promise.all([api('/api/hardware'),api('/api/camera-defaults',{task:'camera defaults',workflow:c.workflow,arms:c.arms,source:c.source,config_path:c.config_path})]);
   if(seq!==sequence)return;inventory=hardware.inventory.cameras;defaults=result.serials;draw();
   status.textContent=`检测到 ${inventory.length} 台 RealSense。序列号与角色的对应关系将用于预览和数据写入；不得重复分配同一设备。`;
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
