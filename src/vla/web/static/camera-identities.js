/* Display and submit SDK serials; retain legacy USB lookup for saved selections. */
'use strict';
window.cameraIdentity=(items,value)=>value?items.find(c=>c.value===value||c.usb_serial===value):undefined;
window.cameraValue=(items,value)=>cameraIdentity(items,value)?.value||value;
window.cameraLabel=(items,value)=>cameraValue(items,value);
window.cameraOption=item=>{const option=new Option(item.label,item.value);option.disabled=item.selectable===false;return option;};
