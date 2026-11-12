/* Presentation only: the server remains the authority for session lifetime. */
'use strict';
class SessionStage {
 constructor() {
  this.phase='checking';this.dismissed=new Set();this.id=null;this.saved=null;
  this.drawer=document.getElementById('configDrawer');
  document.getElementById('viewConfig').onclick=()=>this.openDrawer();
  document.getElementById('closeConfig').onclick=()=>this.drawer.close();
  this.drawer.addEventListener('keydown',event=>{
   if(event.key!=='Tab')return;
   const focusable=[...this.drawer.querySelectorAll('button,input,select,textarea,summary,a[href],[tabindex]')]
    .filter(el=>!el.disabled&&el.tabIndex>=0&&el.getClientRects().length);
   if(!focusable.length)return;
   const index=focusable.indexOf(document.activeElement);
   event.preventDefault();focusable[(index+(event.shiftKey?-1:1)+focusable.length)%focusable.length].focus();
  });
  this.drawer.addEventListener('close',()=>{
   document.getElementById('configHome').append(document.getElementById('sessionConfig'));
   const compliance=document.getElementById('compliancePanel');
   compliance.querySelectorAll('input,button').forEach(el=>el.disabled=false);
   configureView();
   document.getElementById('viewConfig').focus({preventScroll:true});
  });
  document.getElementById('nextSession').onclick=()=>{
   if(this.phase!=='finished')return;
   this.dismissed.add(this.id);this.set('configure');
   clearPreflight();lockSetup();configureView();
   window.dispatchEvent(new CustomEvent('vla-live-status'));
   document.getElementById('configTitle').focus({preventScroll:true});
  };
 }
 get locked(){return this.phase!=='configure';}
 ignores(message){return this.dismissed.has(message.session_id);}
 bootstrap(record){
  if(this.dismissed.has(record?.session_id))return;
  if(record?.alive){this.identify(record.session_id,record.config);this.set('running',false);}
  else if(!record&&this.phase==='checking')this.set('configure',false);
 }
 identify(id,saved){
  this.id=id;this.saved=saved||this.saved;this.configKey=JSON.stringify(this.saved);
  if(!this.saved)return;
  const c=this.saved,fields=[['任务',c.task],['机械臂',c.arms==='dual'?'双臂 · 左 + 右':'单左臂'],['输入源',c.source==='gello'?'GELLO':'Gamepad'],['输出位置',c.dataset_root],['数据集模式',c.resume?'续采已有数据集':'新建数据集'],['TCP XYZ 限制',c.workspace_limits_enabled===false?'关闭':c.workspace_limits_enabled===true?'开启':'使用 YAML 配置']];
  if(c.workflow==='deploy')fields.push(['部署模型',`${c.policy_kind==='sft'?'PI0.5 SFT':'RLT'} · ${c.policy_path||'未指定'}`]);
  const summary=document.getElementById('summaryFields');summary.replaceChildren();
  for(const [label,value] of fields){const pair=document.createElement('div'),term=document.createElement('dt'),description=document.createElement('dd');term.textContent=label;description.textContent=value||'—';pair.append(term,description);summary.append(pair);}
 }
 observe(message){
  if(this.ignores(message))return;
  const s=message.status||{},record=message.active;
  if(message.session_id){
   if(this.id!==message.session_id||this.configKey!==JSON.stringify(message.config||record?.config))this.identify(message.session_id,message.config||record?.config);
   if(s.final||s.state==='stopped')this.set('finished');
   else this.set('running');
  }else if(record?.alive){this.identify(record.session_id,record.config);this.set('running');}
  else if(message.connection_state==='idle'&&this.phase==='checking')this.set('configure',false);
 }
 set(next,animate=true){
  if(next===this.phase)return;
  const previous=this.phase;this.phase=next;
  if(this.drawer.open)this.drawer.close();
  const visible={connectionCheck:next==='checking',configHost:next==='configure',configActions:next==='configure',sessionSummary:['running','finished'].includes(next),sessionStatus:['running','finished'].includes(next),sessionWorkbench:next==='running',sessionEnd:next==='finished',sessionEvents:['running','finished'].includes(next)};
  const panel=document.getElementById('livePanel');panel.dataset.phase=next;
  for(const [id,show] of Object.entries(visible))document.getElementById(id).hidden=!show;
  document.getElementById('phaseTitle').textContent=next==='finished'?'会话结束摘要':'运行会话';
  document.querySelector('#sessionEvents details').open=next==='finished';
  if(animate&&previous!=='checking'&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
   const target=document.getElementById(next==='configure'?'configHost':next==='finished'?'sessionEnd':'sessionSummary');
   target.animate([{opacity:0,transform:'translateY(8px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'ease-out'});
  }
  window.dispatchEvent(new CustomEvent('vla-stage-change'));
  requestAnimationFrame(()=>{
   window.dispatchEvent(new CustomEvent('vla-layout-change'));
   if(previous!=='checking'&&!panel.hidden){
    const heading=document.getElementById(next==='configure'?'configTitle':'phaseTitle');
    heading.focus({preventScroll:true});panel.scrollIntoView({block:'start',behavior:'instant'});
   }
  });
 }
 openDrawer(){
  document.getElementById('drawerBody').append(document.getElementById('sessionConfig'));
  const compliance=document.getElementById('compliancePanel');
  document.getElementById('drawerBody').append(compliance);
  compliance.querySelectorAll('input,button').forEach(el=>el.disabled=true);
  this.drawer.showModal();
  if(!matchMedia('(prefers-reduced-motion: reduce)').matches)this.drawer.animate([{opacity:0,transform:'translateX(24px)'},{opacity:1,transform:'translateX(0)'}],{duration:220,easing:'ease-out'});
 }
}
