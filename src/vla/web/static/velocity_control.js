/* Separate the editable draft from the last worker acknowledgement. */
'use strict';
class VelocityControlEditor {
 constructor(root, send, stage) {
  this.root=root;this.send=send;this.stage=stage;this.effective={};this.dirty=new Set();this.revisions={};this.pending=null;this.failure='';this.sessionId=null;this.available=false;
  this.inputs=[...root.querySelectorAll('[data-velocity_control]')];
  try{this.profiles=JSON.parse(localStorage.getItem('vla.velocity_control.v1')||'{}');}catch{this.profiles={};}
  if(!this.profiles||typeof this.profiles!=='object'||Array.isArray(this.profiles))this.profiles={};
  for(const input of this.inputs){
   const key=input.dataset.velocity_control;
   input.addEventListener('input',()=>{this.dirty.add(key);this.revisions[key]=(this.revisions[key]||0)+1;this.failure='';this.remember();this.draw();});
   const effective=document.createElement('small');effective.dataset.effective=key;input.parentElement.append(effective);
  }
  root.querySelector('#applyVelocityControl').onclick=()=>this.apply();
  root.querySelector('#resetVelocityControl').onclick=()=>{this.dirty.clear();this.failure='';this.fill();this.remember();};
 }
 profile(key){
  if(this.profileKey===key)return;
  this.remember();this.profileKey=key;this.defaultKey=null;this.effective={};this.dirty.clear();this.revisions={};
  const saved=this.profiles[key];
  for(const input of this.inputs){
   const name=input.dataset.velocity_control,value=saved?.[name];
   input.value=typeof value==='string'?value:'';
   if(typeof value==='string')this.dirty.add(name);
  }
 }
 remember(){
  if(!this.profileKey||(!Object.keys(this.effective).length&&!this.dirty.size))return;
  // Preferences are editable startup values, never evidence of a worker ACK.
  const values=Object.fromEntries(this.inputs.map(input=>[input.dataset.velocity_control,input.value]));
  if(JSON.stringify(this.profiles[this.profileKey])===JSON.stringify(values))return;
  this.profiles[this.profileKey]=values;
  try{localStorage.setItem('vla.velocity_control.v1',JSON.stringify(this.profiles));}catch{}
 }
 values(){const result={};for(const input of this.inputs)if(input.value!=='')result[input.dataset.velocity_control]=input.dataset.velocity_control.endsWith('_enabled')?input.value==='true':Number(input.value);return result;}
 assertDraft(){for(const input of this.inputs)if(this.dirty.has(input.dataset.velocity_control)&&(input.value===''||!input.checkValidity())){input.reportValidity();throw new Error('请修正速度控制参数，修改已保留');}}
 valid(){return this.inputs.every(input=>input.value!==''&&input.checkValidity());}
 fill(){for(const input of this.inputs){const key=input.dataset.velocity_control;if(!this.dirty.has(key))input.value=this.effective[key]??'';}this.draw();}
 setEffective(values){if(values){this.effective={...values};this.fill();}}
 defaults(values,key){if(this.defaultKey===key&&Object.keys(this.effective).length)return;this.defaultKey=key;this.setEffective(values||{});}
 observe(id, velocity_control){
  if(!velocity_control)return;
  if(this.sessionId!==id){this.sessionId=id;this.pending=null;this.failure='';}
  const request=(velocity_control.requests||[]).find(x=>x.request_id===this.pending?.id)||velocity_control.request;
  if(this.pending&&request?.request_id===this.pending.id&&request.status!=='pending'){
   if(request.status==='applied'){
    for(const [key,revision] of Object.entries(this.pending.revisions))if((this.revisions[key]||0)===revision)this.dirty.delete(key);
    this.failure='';
   }else this.failure=request.error||'参数应用失败';
   this.pending=null;
  }
  this.workerPending=Boolean(velocity_control.pending);
  this.workerFault=velocity_control.fault||'';
  for(const input of this.inputs)if(input.value!==''&&this.values()[input.dataset.velocity_control]===velocity_control.values?.[input.dataset.velocity_control])this.dirty.delete(input.dataset.velocity_control);
  this.setEffective(velocity_control.values);
  // Entering the next session must retain the last observed values, rather
  // than treating the YAML defaults as a newly selected configuration.
  if(velocity_control.values){this.defaultKey=this.profileKey;this.remember();}
 }
 async apply(){
  if(!this.available||this.stage()!=='running'||this.pending||this.workerPending||!this.valid())return;
  const values={},revisions={};
  for(const [key,value] of Object.entries(this.values()))if(value!==this.effective[key]){values[key]=value;revisions[key]=this.revisions[key]||0;}
  if(!Object.keys(values).length){this.dirty.clear();this.draw();return;}
  const id=crypto.randomUUID();this.pending={id,revisions};this.failure='';this.draw();
  try{await this.send(values,id);}catch(error){if(this.pending?.id===id){this.pending=null;this.failure=error.message;}this.draw();}
 }
 commandFailure(commands){
  const failed=commands?.find(x=>x.command_id===this.pending?.id&&x.status==='failed');
  if(failed){this.pending=null;this.failure=failed.error||'参数请求失败';this.draw();}
 }
 draw(){
  this.root.dataset.pendingRequest=this.pending?.id||'';this.root.dataset.dirtyFields=[...this.dirty].join(',');
  const configure=this.stage()==='configure',waiting=Boolean(this.pending||this.workerPending),drawer=Boolean(this.root.closest('dialog')?.open);
  this.root.querySelector('#velocity_controlStatus').textContent=this.failure||(!configure&&this.workerFault)||
   (waiting?'参数应用中，等待 worker 确认':this.dirty.size?(configure?'未应用 · 将随启动提交':'未应用 · 已保留修改'):configure?'启动初值':'当前参数已确认');
  const apply=this.root.querySelector('#applyVelocityControl');apply.hidden=configure;
  apply.disabled=drawer||!this.available||this.stage()!=='running'||waiting||!!this.workerFault||!this.valid();
  this.root.querySelector('#resetVelocityControl').disabled=drawer||waiting;
  for(const node of this.root.querySelectorAll('[data-effective]'))node.textContent=`${configure?'初值':'有效值'}：${this.effective[node.dataset.effective]??'待检查配置'}`;
 }
}
window.VelocityControlEditor=VelocityControlEditor;
