/* Offline plots share uPlot and the live palette, but never change export fields. */
'use strict';
window.offlinePlots=(()=>{
 const el=id=>document.getElementById(id);
 let info=null,fields=[],selected=new Set(),plots=[],context=null,current=null,cache=null,palette=chartPalette();
 const finite=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
 const group=d=>`${d.arm==='left'?'左臂':d.arm==='right'?'右臂':'单臂'} · ${d.group} · ${d.unit} · ${d.source}`;
 const seconds=()=>Number(el('processingWindow').value);
 function defaults(){selected=new Set(fields.filter(d=>d.source==='状态'&&['TCP 位置','夹爪'].includes(d.group)).map(d=>d.key));}
 function choices(){
  const host=el('processingSignals');host.replaceChildren();const groups=new Map();
  for(const d of fields){const title=group(d);if(!groups.has(title))groups.set(title,[]);groups.get(title).push(d);}
  for(const [title,ds] of groups){const block=document.createElement('div'),heading=document.createElement('h3');heading.textContent=title;block.append(heading);
   for(const d of ds){const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=selected.has(d.key);check.onchange=()=>{check.checked?selected.add(d.key):selected.delete(d.key);rebuild();};label.append(check,document.createTextNode(d.name));block.append(label);}host.append(block);
  }
 }
 function mark(u,ds){
  if(!current||!context)return;const t=finite(current.timestamp);if(t===null)return;
  const x=u.valToPos(t,'x',true),ctx=u.ctx,ratio=devicePixelRatio||1;
  if(x<u.bbox.left||x>u.bbox.left+u.bbox.width)return;
  ctx.save();ctx.beginPath();ctx.rect(u.bbox.left,u.bbox.top,u.bbox.width,u.bbox.height);ctx.clip();
  ctx.strokeStyle=palette.axis;ctx.lineWidth=ratio;ctx.setLineDash([4*ratio,4*ratio]);ctx.beginPath();ctx.moveTo(x,u.bbox.top);ctx.lineTo(x,u.bbox.top+u.bbox.height);ctx.stroke();ctx.setLineDash([]);
  ds.forEach((d,i)=>{const y=finite(current[d.column]?.[d.index]);if(y===null||!u.series[i+1].show)return;ctx.fillStyle=palette.series[i%palette.series.length];ctx.beginPath();ctx.arc(x,u.valToPos(y,'y',true),3.5*ratio,0,2*Math.PI);ctx.fill();});ctx.restore();
  u.setLegend({idx:u.data[0].indexOf(t)},false);
 }
 function rebuild(){
  for(const p of plots)p.plot.destroy();plots=[];const host=el('processingPlots');host.replaceChildren();
  const groups=new Map();for(const d of fields.filter(d=>selected.has(d.key))){const k=group(d);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(d);}
  for(const [title,ds] of groups){const box=document.createElement('div');box.className='plot';const heading=document.createElement('h3');heading.textContent=title;box.append(heading);host.append(box);
   const plot=new uPlot({width:Math.max(160,box.clientWidth-12),height:205,scales:{x:{time:false}},series:[{label:'时间 (s)'},...ds.map((d,i)=>({label:d.name,stroke:()=>palette.series[i%palette.series.length],width:1.5,spanGaps:false,points:{show:false}}))],axes:[{label:'回合时间 (s)',size:45,stroke:()=>palette.axis,grid:{stroke:()=>palette.grid},values:(_u,t)=>t.map(x=>x.toFixed(2))},{size:60,stroke:()=>palette.axis,grid:{stroke:()=>palette.grid}}],cursor:{drag:{x:false,y:false}},legend:{show:true},hooks:{draw:[u=>mark(u,ds)]}},[[],...ds.map(()=>[])],box);
   plots.push({plot,box,ds});
  }el('processingPlotEmpty').hidden=plots.length>0;draw();
 }
 function draw(){
  if(!context||!current)return;
  const center=Number(current.timestamp),span=seconds(),all=context.rows;
  const first=Number(all[0]?.timestamp??center),last=Number(all.at(-1)?.timestamp??center);
  const begin=Math.max(first,Math.min(center-span/2,last-span)),end=Math.max(begin+1/(info?.fps||30),Math.min(last,begin+span));
  // Include the exact current row even when the context has been downsampled.
  const rows=all.filter(r=>r.timestamp>=begin&&r.timestamp<=end&&r.frame_index!==current.frame_index);
  rows.push(current);rows.sort((a,b)=>a.timestamp-b.timestamp);
  for(const p of plots){p.plot.setData([rows.map(r=>r.timestamp),...p.ds.map(d=>rows.map(r=>finite(r[d.column]?.[d.index]))) ]);p.plot.setScale('x',{min:begin,max:end});}
  el('processingPlotStatus').textContent=`当前点：帧 ${current.frame_index} · ${center.toFixed(3)} s；曲线：${begin.toFixed(2)}–${end.toFixed(2)} s${context.sampled?'（长窗口已抽样，当前点为原始值）':''}`;
 }
 function clear(){cache=null;context=null;current=null;for(const p of plots)p.plot.setData([[],...p.ds.map(()=>[])]);el('processingPlotStatus').textContent='选择回合后显示曲线；圆点与图像对应同一帧。';}
 function reset(summary){info=summary;clear();fields=info.fields.map((d,i)=>({...d,key:'state.'+i,column:'observation.state',index:i,source:'状态'}));
  for(const [i,name] of info.action_names.entries()){const arm=name.startsWith('left_')?'left':name.startsWith('right_')?'right':'single';fields.push({key:'action.'+i,name,arm,group:'动作',unit:info.action_units?.[i]||'未声明',column:'action',index:i,source:'记录动作'});}
  defaults();choices();rebuild();
 }
 async function windowFor(path,frame,fetcher){const span=seconds(),half=Math.ceil(span*(info?.fps||30)/2);
  if(cache?.path===path&&cache.seconds===span&&cache.data.start<=Math.max(0,frame-half)&&cache.data.stop>=Math.min(cache.data.length,frame+half+1))return cache.data;
  const data=await fetcher(`${path}/window?frame=${frame}&seconds=${span}`);cache={path,seconds:span,data};return data;
 }
 el('processingHidePlots').onclick=()=>{selected.clear();choices();rebuild();};el('processingDefaultPlots').onclick=()=>{defaults();choices();rebuild();};
 window.addEventListener('vla-theme-change',()=>{palette=chartPalette();for(const p of plots)p.plot.redraw(true,true);});
 new ResizeObserver(()=>{for(const p of plots)p.plot.setSize({width:Math.max(160,p.box.clientWidth-12),height:205});}).observe(el('processingPlots'));
 return {reset,clear,windowFor,commit(data,row){context=data;current=row;draw();}};
})();
