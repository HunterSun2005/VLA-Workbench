'use strict';
(() => {
 const panel=$('workspaceSetup'),custom=$('workspaceCustom'),container=$('workspaceFields'),status=$('workspaceStatus');
 let defaults={},values={},shown='',sequence=0,restored=null;
 const applicable=c=>c.workflow==='collect'&&c.source==='gamepad';
 const running=()=>current?.status?['starting','standby','active','recording','saving','fault'].includes(current.status.state)&&!current.status.final:Boolean(active?.alive);
 window.readWorkspaceBounds=c=>{
  if(!applicable(c)||!custom.checked)return null;
  const result={};
  for(const arm of c.arms==='dual'?['left','right']:['left']){
   result[arm]={min:[],max:[]};
   for(const bound of ['min','max'])for(let i=0;i<3;i++){
    const input=document.getElementById(`workspace-${arm}-${bound}-${i}`);
    result[arm][bound].push(input?.value.trim()?Number(input.value):null);
   }
  }
  return result;
 };
 function lock(){
  const busy=running()||panel.hidden;custom.disabled=busy;$('workspaceReset').disabled=busy;
  container.querySelectorAll('input').forEach(input=>input.disabled=busy||!custom.checked);
 }
 function draw(c){
  container.replaceChildren();
  for(const arm of c.arms==='dual'?['left','right']:['left']){
   const heading=document.createElement('strong');heading.textContent=arm==='left'?'左臂':'右臂';container.append(heading);
   const table=document.createElement('table');table.className='workspace-table';
   const header=document.createElement('tr');for(const title of ['轴','最小值 (m)','最大值 (m)']){const th=document.createElement('th');th.textContent=title;header.append(th);}table.append(header);
   for(let i=0;i<3;i++){
    const row=document.createElement('tr'),axis=document.createElement('th');axis.textContent='XYZ'[i];row.append(axis);
    for(const bound of ['min','max']){
     const td=document.createElement('td'),input=document.createElement('input');input.type='number';input.step='any';input.required=true;
     input.id=`workspace-${arm}-${bound}-${i}`;input.setAttribute('aria-label',`${arm==='left'?'左臂':'右臂'} ${'XYZ'[i]} ${bound==='min'?'最小值':'最大值'}（米）`);
     input.value=(values[arm]||defaults[arm])?.[bound]?.[i]??'';
     input.oninput=()=>{values=readWorkspaceBounds(c)||{};clearPreflight();};td.append(input);row.append(td);
    }table.append(row);
   }container.append(table);
  }lock();
 }
 async function view(){
  const c=config();panel.hidden=!applicable(c);if(panel.hidden){lock();return;}
  const saved=current?.config||active?.config,id=current?.session_id||active?.session_id;
  if(id&&id!==restored&&saved?.arms===c.arms&&saved?.workflow==='collect'&&saved?.source==='gamepad'){
   restored=id;custom.checked=Boolean(saved.workspace_bounds);values=saved.workspace_bounds||{};shown='';
  }
  if(!token)return;
  const key=`${c.arms}|${c.config_path}`;
  if(key===shown){lock();return;}shown=key;
  const seq=++sequence;draw(c);status.textContent='正在读取配置范围…';
  try{
   const result=await api('/api/workspace',{...c,task:c.task||'workspace',workspace_bounds:null});
   if(seq!==sequence)return;defaults=result.bounds;if(!custom.checked)values={};draw(c);
   status.textContent='范围来自当前采集配置；勾选后可修改。超出范围仍会拒绝启用，不会自动拉回。';
  }catch(e){if(seq===sequence){status.textContent='读取范围失败：'+e.message;shown='';}}
 }
 custom.onchange=()=>{if(!custom.checked)values={};draw(config());clearPreflight();};
 $('workspaceReset').onclick=()=>{custom.checked=false;values={};draw(config());clearPreflight();};
 form.elements.config_path.addEventListener('change',view);
 window.addEventListener('vla-config-change',view);
 window.addEventListener('vla-live-status',lock);
 view();
})();
