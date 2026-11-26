/* Client-side session log: the server only snapshots its latest 16 command results and
   10 event strings, so the browser appends what it observes in arrival order. Times are
   the browser's receive time; entries already present on (re)connect are marked 「—」. */
'use strict';
class SessionLog {
 constructor(body,count,follow,copy){
  this.body=body;this.count=count;this.follow=follow;this.limit=500;
  this.reset();
  body.addEventListener('scroll',()=>{this.pinned=body.scrollHeight-body.scrollTop-body.clientHeight<24;follow.hidden=this.pinned;});
  follow.onclick=()=>{this.pinned=true;follow.hidden=true;body.scrollTop=body.scrollHeight;};
  copy.onclick=async()=>{
   const text=this.entries.map(e=>`${e.time} ${e.kind} ${e.text}`).join('\n');if(!text)return;
   const label=copy.textContent;
   try{await copyDiagnosticText(text);copy.textContent='已复制';}catch{copy.textContent='复制失败';}
   setTimeout(()=>{copy.textContent=label;},2000);
  };
 }
 reset(placeholder='等待会话'){
  this.session=null;this.entries=[];this.commands=new Map();this.events=[];this.primed=false;this.pinned=true;
  this.follow.hidden=true;this.body.replaceChildren();this.empty(placeholder);this.count.textContent='';
 }
 empty(text){const p=document.createElement('p');p.className='log-empty';p.textContent=text;this.body.append(p);}
 observe(message){
  if(!message.session_id)return;
  if(message.session_id!==this.session){this.reset('暂无记录');this.session=message.session_id;}
  const stamp=this.primed?new Date().toLocaleTimeString('zh-CN',{hour12:false}):'—';
  const added=[];
  for(const c of message.commands||[]){
   const key=c.command_id||`${c.operation}:${c.revision}`,status=this.commands.get(key);
   if(status===c.status)continue;
   this.commands.set(key,c.status);
   added.push({time:stamp,kind:'命令',level:c.status==='failed'?'error':c.status==='accepted'?'pending':'ok',
    text:`${SessionLog.operations[c.operation]||c.operation} · ${SessionLog.statuses[c.status]||c.status}${c.error?' — '+c.error:''}`});
  }
  while(this.commands.size>512)this.commands.delete(this.commands.keys().next().value);
  // Events are a rolling tail of plain strings: append only past the longest overlap.
  const next=message.status?.events||[];
  let overlap=Math.min(this.events.length,next.length);
  while(overlap>0&&!this.events.slice(-overlap).every((text,i)=>text===next[i]))overlap--;
  for(const text of next.slice(overlap))added.push({time:stamp,kind:'事件',level:/失败|错误|故障|error|fail/i.test(text)?'error':'info',text});
  this.events=next.slice();this.primed=true;
  if(added.length)this.append(added);
 }
 append(items){
  if(!this.entries.length)this.body.replaceChildren();
  for(const item of items){
   this.entries.push(item);
   const row=document.createElement('div');row.className='log-row';row.dataset.level=item.level;row.dataset.kind=item.kind==='命令'?'command':'event';
   for(const [cls,text] of [['log-time',item.time],['log-kind',item.kind],['log-text',item.text]]){const span=document.createElement('span');span.className=cls;span.textContent=text;row.append(span);}
   this.body.append(row);
  }
  while(this.entries.length>this.limit){this.entries.shift();this.body.firstElementChild.remove();}
  this.count.textContent=`${this.entries.length} 条`;
  if(this.pinned)this.body.scrollTop=this.body.scrollHeight;
 }
}
SessionLog.operations={enable:'启用控制',pause:'软件暂停',record:'开始回合',save:'保存回合',discard:'丢弃回合',critical:'Critical 标注',next_subtask:'下一阶段',recover:'恢复故障',stop:'停止会话',mode:'切换控制模式',intervention:'人工接管',compliance:'应用阻抗参数',pose_capture:'记录姿态',pose_restore:'恢复姿态'};
SessionLog.statuses={accepted:'执行中',completed:'完成',failed:'失败'};
window.SessionLog=SessionLog;
