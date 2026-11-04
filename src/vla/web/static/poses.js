'use strict';
(() => {
 const panel=$('posePanel'),list=$('poseRecords'),arm=$('poseArm'),note=$('poseNote');
 let records=[],lastResult='',loaded=false,loading=false;
 const picked=()=>records.find(r=>r.id===list.value);
 function fail(message=''){$('poseError').textContent=message;$('poseError').hidden=!message;}
 function detail(){
  const r=picked();if(!r){$('poseDetail').textContent='记录保存在 runtime/web/，跨会话保留。模拟记录与真实记录分开。';return;}
  $('poseDetail').textContent=`${r.pose.arm==='left'?'左臂':'右臂'} · ${r.pose.device} · ${r.pose.frame}\nXYZ (m)：${r.pose.position.map(x=>x.toFixed(4)).join(', ')}\n旋转矩阵：${r.pose.rotation.map(row=>row.map(x=>x.toFixed(3)).join(', ')).join(' / ')}\n${new Date(r.created_at).toLocaleString()}`;
 }
 async function refresh(){
  if(loading||!token)return;loading=true;
  try{const old=list.value,result=await api('/api/poses');records=result.records;list.replaceChildren();
   for(const r of records)list.add(new Option(`${r.pose.arm==='left'?'左':'右'} · ${r.note}`,r.id));
   if(!records.length)list.add(new Option('暂无记录',''));if(records.some(r=>r.id===old))list.value=old;
   loaded=true;fail();detail();
  }catch(e){fail(e.message);}finally{loading=false;loaded=true;update();}
 }
 function update(){
  panel.hidden=selectedPage!=='collect';
  const s=current?.status||{},c=current?.config||active?.config||config(),op=s.pose_operation;
  const moving=op?.state==='moving',live=connected&&!pending&&!s.final&&performance.now()-lastReceive<500&&current?.now-s.updated_at<.5;
  const supported=(s.available_commands||[]).includes('pose_capture')||moving;
  const ready=live&&supported&&c.workflow==='collect'&&s.state==='standby'&&!s.enabled&&!s.recording&&!s.saving&&!s.error&&!moving;
  arm.options[1].disabled=c.arms!=='dual';if(arm.options[1].disabled&&arm.value==='right')arm.value='left';
  $('poseCapture').disabled=!ready||!note.value.trim();
  $('poseRestore').disabled=!ready||!picked()||(c.arms!=='dual'&&picked().pose.arm==='right');
  $('poseCancel').disabled=!live||!moving;
  $('poseRename').disabled=$('poseDelete').disabled=!picked()||moving||loading;
  const names={idle:'尚未执行姿态操作',saved:'姿态已保存',moving:'正在恢复姿态 · 可随时软件暂停',completed:'恢复完成 · 待命',cancelled:'恢复已取消',failed:'恢复失败'};
  $('poseStatus').textContent=op?`${names[op.state]||op.state}${op.note?' · '+op.note:''}${op.message?' · '+op.message:''}`:'启动采集会话后，可在待命状态记录或恢复。';
  if(live&&c.workflow==='collect'&&!supported)$('poseStatus').textContent='当前会话未加载姿态库功能，请正常结束会话并重新启动。';
  if(op?.state==='failed')setIssue('pose-restore',op.message||'姿态恢复失败');else setIssue('pose-restore','');
  const key=`${current?.session_id}|${JSON.stringify(op)}`;
  if(key!==lastResult){lastResult=key;if(op?.state==='saved')refresh();}
  if(!loaded&&token&&!panel.hidden&&!loading)refresh();
 }
 $('poseCapture').onclick=()=>command('pose_capture',{arm:arm.value,note:note.value.trim()});
 $('poseRestore').onclick=()=>{const r=picked();if(r&&confirm(`恢复「${r.note}」：${r.pose.arm==='left'?'左':'右'}臂将低速移动到保存的 TCP 位置和朝向。请确认路径畅通，夹爪保持当前状态。`))command('pose_restore',{pose_id:r.id});};
 $('poseCancel').onclick=()=>command('pause');
 $('poseRefresh').onclick=refresh;
 $('poseRename').onclick=async()=>{const r=picked(),text=r&&prompt('修改姿态备注',r.note);if(text==null)return;try{await api('/api/poses/'+r.id,{operation:'rename',note:text});await refresh();}catch(e){fail(e.message);}};
 $('poseDelete').onclick=async()=>{const r=picked();if(!r||!confirm(`删除姿态记录「${r.note}」？`))return;try{await api('/api/poses/'+r.id,{operation:'delete'});await refresh();}catch(e){fail(e.message);}};
 list.onchange=()=>{detail();update();};arm.onchange=update;note.oninput=update;
 for(const event of ['vla-config-change','vla-live-status','vla-page-change'])window.addEventListener(event,update);
 setInterval(update,500);update();
})();
