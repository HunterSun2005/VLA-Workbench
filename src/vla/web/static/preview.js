/* Reserve HTTP/1 connections for commands: one preview owner per browser/origin. */
'use strict';
(() => {
 const channel=typeof BroadcastChannel==='function'?new BroadcastChannel('vla-preview'):null;
 let wanted=false,suspended=false,acquiring=false,owned=false,release=null,generation=0;
 function clear(){for(const img of document.querySelectorAll('.camera img'))img.removeAttribute('src');}
 function notice(message){$('previewNotice').hidden=!wanted;$('previewStatus').textContent=message;$('previewTake').hidden=owned;}
 function relinquish(){generation++;clear();owned=false;release?.();release=null;}
 function attach(){for(const img of document.querySelectorAll('.camera img'))if(img.dataset.previewSrc)img.src=img.dataset.previewSrc;}
 async function acquire(){
  if(!wanted||suspended||document.hidden||acquiring||owned)return;
  if(!navigator.locks){owned=true;attach();notice('当前浏览器不支持标签页预览互斥，请只保留一个相机监控页。');return;}
  acquiring=true;const version=generation;
  try{
   await navigator.locks.request('vla-camera-preview',{ifAvailable:true},async lock=>{
    if(!wanted||suspended||document.hidden||version!==generation)return;
    if(!lock){notice('另一标签页正在显示相机；本页遥测与操作仍可用。');return;}
    owned=true;attach();notice('相机预览在本页；其他标签页保留遥测与操作。');
    await new Promise(resolve=>{release=resolve;});
   });
  }catch(e){relinquish();notice('无法启用相机预览：'+e.message);}
  finally{acquiring=false;}
 }
 function take(){if(!wanted)return;suspended=false;channel?.postMessage('take');acquire();}
 window.startPreviewImages=()=>{wanted=true;suspended=false;notice('正在申请相机预览…');acquire();};
 window.releasePreview=()=>{wanted=false;relinquish();notice('');};
 $('previewTake').onclick=take;
 if(channel)channel.onmessage=e=>{if(e.data==='take'&&wanted){suspended=true;relinquish();notice('相机预览已交给另一标签页；本页遥测与操作仍可用。');}};
 document.addEventListener('visibilitychange',()=>{if(document.hidden){relinquish();notice('后台标签页暂停相机预览，采集继续运行。');}else take();});
 window.addEventListener('pagehide',()=>{wanted=false;relinquish();channel?.close();});
 setInterval(acquire,500);
 if(document.querySelector('.camera img[data-preview-src]'))window.startPreviewImages();
})();
