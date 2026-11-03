'use strict';
(() => {
 const storageKey='vla.columnRatios', narrow=matchMedia('(max-width:720px)');
 let ratios={},drag=null;
 try{const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');if(saved&&typeof saved==='object')ratios=saved;}catch{}
 const layouts=['livePanel','processingPanel'].map(id=>{
  const main=document.getElementById(id),bar=document.createElement('div');
  main.classList.add('resizable-layout');bar.className='column-resizer';bar.tabIndex=0;
  bar.setAttribute('role','separator');bar.setAttribute('aria-orientation','vertical');
  bar.setAttribute('aria-label','调整左右列宽度');bar.setAttribute('aria-controls',id);
  bar.title='拖动调整左右列；方向键微调；双击恢复默认';
  main.insertBefore(bar,main.querySelector('article'));
  const layout={main,bar,page:id==='processingPanel'?'processing':'collect',width:0};
  const save=()=>{try{localStorage.setItem(storageKey,JSON.stringify(ratios));}catch{}};
  function bounds(){
   const style=getComputedStyle(main),width=main.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight)-24;
   return {width,min:Math.min(layout.page==='processing'?260:240,width/2),max:Math.max(width/2,Math.min(width*.65,width-360))};
  }
  function setWidth(value,remember=false){
   if(main.hidden||narrow.matches)return;
   const b=bounds(),left=Math.max(b.min,Math.min(b.max,value));
   layout.width=left;main.style.setProperty('--left-column',`${left}px`);
   for(const [key,v] of Object.entries({min:b.min,max:b.max,now:left}))bar.setAttribute(`aria-value${key}`,String(Math.round(v/b.width*100)));
   bar.setAttribute('aria-valuetext',`左列 ${Math.round(left/b.width*100)}%，右列 ${100-Math.round(left/b.width*100)}%`);
   if(remember)ratios[layout.page]=left/b.width;
  }
  layout.apply=()=>{
   bar.tabIndex=narrow.matches?-1:0;
   if(main.hidden||narrow.matches)return;
   const ratio=ratios[layout.page],fallback=layout.page==='processing'?340:290;
   setWidth(typeof ratio==='number'&&Number.isFinite(ratio)&&ratio>0&&ratio<1?bounds().width*ratio:fallback);
  };
  const finish=()=>{if(drag?.layout!==layout)return;const pointer=drag.pointer;drag=null;document.body.classList.remove('resizing-columns');if(bar.hasPointerCapture(pointer))bar.releasePointerCapture(pointer);save();};
  layout.finish=finish;
  bar.addEventListener('pointerdown',event=>{
   if(event.button!==0||narrow.matches)return;
   event.preventDefault();bar.focus({preventScroll:true});bar.setPointerCapture(event.pointerId);
   drag={layout,pointer:event.pointerId,start:event.clientX,width:layout.width};document.body.classList.add('resizing-columns');
  });
  bar.addEventListener('pointermove',event=>{if(drag?.layout===layout&&drag.pointer===event.pointerId)setWidth(drag.width+event.clientX-drag.start,true);});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])bar.addEventListener(event,finish);
  bar.addEventListener('dblclick',()=>{delete ratios[layout.page];save();layout.apply();});
  bar.addEventListener('keydown',event=>{
   if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
   event.preventDefault();const b=bounds(),step=event.shiftKey?40:16;
   setWidth(event.key==='Home'?b.min:event.key==='End'?b.max:layout.width+(event.key==='ArrowRight'?step:-step),true);save();
  });
  new ResizeObserver(()=>layout.apply()).observe(main);
  return layout;
 });
 function activate(){
  const page=selectedPage;
  for(const layout of layouts){
   const next=layout.main.id==='processingPanel'?'processing':page==='deploy'?'deploy':'collect';
   if(layout.page!==next||layout.main.hidden||narrow.matches)layout.finish();
   layout.page=next;layout.apply();
  }
 }
 window.addEventListener('vla-page-change',activate);
 window.addEventListener('vla-config-change',activate);
 window.addEventListener('blur',()=>layouts.forEach(layout=>layout.finish()));
 narrow.addEventListener('change',activate);activate();
})();
