/* Read-only CAM viewer adapted from tkcaccia/wsiTools' Tissue visualizer skill.
 * OpenSeadragon supplies viewport scheduling; source TIFF pixels are read only
 * for visible tiles, so GitHub Pages never redistributes a full source slide.
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const slide = new URLSearchParams(location.search).get('slide');
  const validSlide = /^(?:tumor|normal)_\d{3}$/.test(slide || '');
  const manifestURL = new URL('../geojson-inspectors.json', location.href);
  const canvas = $('overlay'), ctx = canvas.getContext('2d');
  const state = {viewer:null, features:[], opacity:.4, outlines:true, selected:'', queued:false,
    preview:null, source:null, sourceFailed:false, sourceWarning:false, supportPath:null};

  function status(message, error=false) {
    $('status').textContent = message;
    $('status').style.color = error ? '#a0393d' : '';
    if (error) { $('loading').textContent = message; $('loading').hidden = false; }
  }
  async function json(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes[0] === 31 && bytes[1] === 139) {
      if (!('DecompressionStream' in window)) throw new Error('This browser cannot read compressed GeoJSON.');
      return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  function pathFor(geometry, transform) {
    if (!geometry || !['Polygon','MultiPolygon'].includes(geometry.type)) return null;
    const path = new Path2D();
    const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
    for (const polygon of polygons) for (const ring of polygon) {
      ring.forEach(([x,y], i) => {
        const p = transform(x,y);
        if (i) path.lineTo(p[0],p[1]); else path.moveTo(p[0],p[1]);
      });
      path.closePath();
    }
    return path;
  }
  function boundsFor(geometry, transform) {
    const bounds = [Infinity,Infinity,-Infinity,-Infinity];
    const scan = node => {
      if (!Array.isArray(node)) return;
      if (node.length >= 2 && Number.isFinite(node[0]) && Number.isFinite(node[1])) {
        const [x,y] = transform(node[0],node[1]);
        bounds[0] = Math.min(bounds[0],x); bounds[1] = Math.min(bounds[1],y);
        bounds[2] = Math.max(bounds[2],x); bounds[3] = Math.max(bounds[3],y);
      } else for (const child of node) scan(child);
    };
    scan(geometry.coordinates);
    return bounds;
  }
  function schedule() {
    if (!state.queued) { state.queued = true; requestAnimationFrame(render); }
  }
  function render() {
    state.queued = false;
    const viewer = state.viewer;
    if (!viewer?.world?.getItemCount()) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1,Math.round(rect.width*dpr)), height = Math.max(1,Math.round(rect.height*dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width=width; canvas.height=height; }
    ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,width,height);
    const item = viewer.world.getItemAt(0);
    const p0 = viewer.viewport.pixelFromPoint(item.imageToViewportCoordinates(0,0),true);
    const px = viewer.viewport.pixelFromPoint(item.imageToViewportCoordinates(state.cropW,0),true);
    const py = viewer.viewport.pixelFromPoint(item.imageToViewportCoordinates(0,state.cropH),true);
    const sx = (px.x-p0.x)/state.cropW, sy = (py.y-p0.y)/state.cropH;
    if (!(sx>0 && sy>0)) return;
    const targetUm = 110*state.mppX/sx;
    const power = 10**Math.floor(Math.log10(targetUm));
    const nice = [1,2,5,10].map(v=>v*power).reduce((a,b)=>Math.abs(a-targetUm)<Math.abs(b-targetUm)?a:b);
    $('scale-line').style.width = `${nice/state.mppX*sx}px`;
    $('scale-label').textContent = nice>=1000 ? `${+(nice/1000).toPrecision(2)} mm` : `${+nice.toPrecision(2)} µm`;
    $('scale').hidden=false;
    ctx.setTransform(dpr*sx,0,0,dpr*sy,dpr*p0.x,dpr*p0.y);
    ctx.save(); ctx.clip(state.supportPath,'evenodd');
    ctx.lineJoin='round'; ctx.lineCap='round'; ctx.lineWidth=1.3/sx;
    for (const feature of state.features) {
      if (state.selected && String(feature.value)!==state.selected) continue;
      const b=feature.bounds;
      if (p0.x+b[2]*sx<0 || p0.x+b[0]*sx>rect.width || p0.y+b[3]*sy<0 || p0.y+b[1]*sy>rect.height) continue;
      ctx.fillStyle=feature.color; ctx.strokeStyle=feature.color;
      ctx.globalAlpha=state.opacity; ctx.fill(feature.path,'evenodd');
      if (state.outlines) {ctx.globalAlpha=.95; ctx.stroke(feature.path);}
    }
    ctx.restore(); ctx.globalAlpha=1;
  }

  function tileSource(record, crop, preview, supportPath) {
    const tileSize=512, maxLevel=Math.ceil(Math.log2(Math.max(crop.w,crop.h)));
    let tiffPromise=null;
    const finishCanvas=(context,tile)=>{
      const image=new Image();
      image.onload=()=>{if (!context.userData?.aborted) context.finish(image,null,'image');};
      image.onerror=()=>{if (!context.userData?.aborted) context.finish(null,'Could not render a tissue tile.');};
      image.src=tile.toDataURL('image/png');
    };
    const source=async()=>{
      if (!tiffPromise) tiffPromise=GeoTIFF.fromUrl(record.high_resolution.url);
      return tiffPromise;
    };
    const previewTile=(context, box, outW, outH)=>{
      const tile=document.createElement('canvas'); tile.width=outW; tile.height=outH;
      const c=tile.getContext('2d');
      c.drawImage(preview,box.x/crop.w*preview.naturalWidth,box.y/crop.h*preview.naturalHeight,
        box.w/crop.w*preview.naturalWidth,box.h/crop.h*preview.naturalHeight,0,0,outW,outH);
      finishCanvas(context,tile);
    };
    return {
      width:crop.w,height:crop.h,tileSize,tileOverlap:0,minLevel:0,maxLevel,
      getTileUrl(level,x,y){return `cam-tiff:${slide}:${level}:${x}:${y}`;},
      getTilePostData(level,x,y){return {level,x,y};},
      hasTransparency(){return true;},
      async downloadTileStart(context){
        const {level,x,y}=context.postData;
        const factor=2**(maxLevel-level), left=x*tileSize*factor, top=y*tileSize*factor;
        const box={x:left,y:top,w:Math.max(0,Math.min(crop.w-left,tileSize*factor)),h:Math.max(0,Math.min(crop.h-top,tileSize*factor))};
        const outW=Math.max(1,Math.ceil(box.w/factor)),outH=Math.max(1,Math.ceil(box.h/factor));
        if (factor>32 || state.sourceFailed) {previewTile(context,box,outW,outH); return;}
        try {
          const tiff=await source();
          const count=await tiff.getImageCount();
          const image=await tiff.getImage(Math.min(count-1,Math.max(0,Math.floor(Math.log2(factor)))));
          const ratioX=record.high_resolution.width/image.getWidth();
          const ratioY=record.high_resolution.height/image.getHeight();
          const window=[Math.floor((crop.x+box.x)/ratioX),Math.floor((crop.y+box.y)/ratioY),
            Math.min(image.getWidth(),Math.ceil((crop.x+box.x+box.w)/ratioX)),
            Math.min(image.getHeight(),Math.ceil((crop.y+box.y+box.h)/ratioY))];
          const rgb=await image.readRGB({window,width:outW,height:outH,interleave:true});
          const tile=document.createElement('canvas'); tile.width=outW; tile.height=outH;
          const c=tile.getContext('2d');
          const rgba=new Uint8ClampedArray(outW*outH*4);
          for(let i=0,j=0;i<rgb.length;i+=3,j+=4){rgba[j]=rgb[i];rgba[j+1]=rgb[i+1];rgba[j+2]=rgb[i+2];rgba[j+3]=255;}
          c.putImageData(new ImageData(rgba,outW,outH),0,0);
          c.globalCompositeOperation='destination-in';
          c.setTransform(1/factor,0,0,1/factor,-box.x/factor,-box.y/factor);
          c.fillStyle='#fff'; c.fill(supportPath,'evenodd');
          c.setTransform(1,0,0,1,0,0); c.globalCompositeOperation='source-over';
          finishCanvas(context,tile);
        } catch(error) {
          state.sourceFailed=true;
          if (!state.sourceWarning) {state.sourceWarning=true; status('Official TIFF detail unavailable; the tissue overview remains visible. '+error.message,true);}
          previewTile(context,box,outW,outH);
        }
      },
      downloadTileAbort(context){if (context.userData) context.userData.aborted=true;}
    };
  }

  async function init() {
    if (!validSlide) throw new Error('Choose a valid CAMELYON16 slide.');
    if (!window.OpenSeadragon || !window.GeoTIFF) throw new Error('The tissue viewer libraries did not load.');
    const manifest=await json(manifestURL);
    const record=manifest.records.find(r=>r.id===`benchmark3-${slide}`);
    if (!record?.high_resolution || !record.tissue_support || !record.geometry?.endsWith('-bora.geojson.gz'))
      throw new Error('A completed Bora review vector is unavailable for this slide.');
    const resolve=path=>new URL(path,manifestURL).href;
    const [support,vector]=await Promise.all([json(resolve(record.tissue_support)),json(resolve(record.geometry))]);
    if (support.coordinate_system!=='calibrated_image_pixel_xy' || !support.features?.length ||
      vector.type!=='FeatureCollection' || vector.features.length!==record.source_feature_count)
      throw new Error('The reviewed tissue or saved vector did not match the manifest.');
    const preview=new Image(); preview.src=resolve(record.image); await preview.decode();
    const [bx,by,bw,bh]=record.image_bounds_xywh;
    const scaleX=record.high_resolution.width/record.high_resolution.calibrated_width;
    const scaleY=record.high_resolution.height/record.high_resolution.calibrated_height;
    const crop={x:bx*scaleX,y:by*scaleY,w:bw*scaleX,h:bh*scaleY};
    const transform=(x,y)=>[(x-bx)*scaleX,(y-by)*scaleY];
    const supportPath=new Path2D();
    for (const feature of support.features) {
      const polygons=feature.geometry.type==='Polygon'?[feature.geometry.coordinates]:feature.geometry.coordinates;
      for (const polygon of polygons) for (const ring of polygon) {
        ring.forEach(([x,y],i)=>{const p=transform(x,y);if(i)supportPath.lineTo(p[0],p[1]);else supportPath.moveTo(p[0],p[1]);});
        supportPath.closePath();
      }
    }
    state.supportPath=supportPath; state.cropW=crop.w; state.cropH=crop.h;
    state.mppX=record.native_mpp_xy[0]/scaleX; state.preview=preview;
    const displayed=vector.features.filter(feature=>feature.properties.value!==2 || Number(feature.properties.area_um2)>=20000);
    const groups=new Map(record.groups.map(g=>[g.value,g]));
    state.features=displayed.map(feature=>{
      const group=groups.get(feature.properties.value);
      if (!group) throw new Error('Unknown prediction class in saved vector.');
      return {value:group.value,color:group.color,path:pathFor(feature.geometry,transform),bounds:boundsFor(feature.geometry,transform)};
    });
    for (const group of record.groups) {
      const option=document.createElement('option');option.value=String(group.value);option.textContent=group.name; $('class').append(option);
      const key=document.createElement('span');key.className='key';
      const swatch=document.createElement('i');swatch.className='swatch';swatch.style.backgroundColor=group.color;
      key.append(swatch,`${group.name}: ${(group.area_um2/1e6).toFixed(3)} mm²`);$('legend').append(key);
    }
    $('title').textContent=`${slide} · Bora prediction`;
    $('download-saved').href=resolve(record.geometry);
    const displayCopy={...vector,display_only:true,source_review_geojson_sha256:record.stored_geojson_sha256,
      display_filter:{class_value:2,minimum_area_um2:20000,omitted_polygons:vector.features.length-displayed.length},features:displayed};
    $('download-displayed').href=URL.createObjectURL(new Blob([JSON.stringify(displayCopy)],{type:'application/geo+json'}));
    $('download-displayed').download=`${slide}-bora-display.geojson`;$('download-displayed').hidden=false;
    const viewer=OpenSeadragon({element:$('osd'),tileSources:tileSource(record,crop,preview,supportPath),
      showNavigationControl:false,showNavigator:false,minZoomImageRatio:.8,maxZoomPixelRatio:1.5,
      animationTime:.25,blendTime:.12,imageLoaderLimit:8,
      gestureSettingsMouse:{clickToZoom:false,dblClickToZoom:true,scrollToZoom:true,dragToPan:true},
      gestureSettingsTouch:{pinchToZoom:true,dragToPan:true}});
    state.viewer=viewer;
    viewer.addHandler('open',()=>{
      $('loading').hidden=true;
      status(`${displayed.length} displayed polygons; ${vector.features.length-displayed.length} red regions below 0.02 mm² hidden for display only. Source H&E appears as you zoom.`);
      schedule();
    });
    viewer.addHandler('open-failed',event=>status(`Tissue image could not open: ${event.message||'unknown error'}`,true));
    viewer.addHandler('tile-load-failed',()=>status('A tissue tile could not load; retry zooming or reload.',true));
    for (const name of ['animation','animation-finish','resize','update-viewport']) viewer.addHandler(name,schedule);
    addEventListener('resize',schedule);
    $('zoom-in').onclick=()=>{viewer.viewport.zoomBy(1.5);viewer.viewport.applyConstraints();};
    $('zoom-out').onclick=()=>{viewer.viewport.zoomBy(1/1.5);viewer.viewport.applyConstraints();};
    $('fit').onclick=()=>viewer.viewport.goHome();
    $('class').onchange=event=>{state.selected=event.target.value;schedule();};
    $('opacity').oninput=event=>{state.opacity=Number(event.target.value);$('opacity-value').textContent=`${Math.round(state.opacity*100)}%`;schedule();};
    $('outlines').onchange=event=>{state.outlines=event.target.checked;schedule();};
    $('he').onchange=event=>{viewer.world.getItemAt(0).setOpacity(event.target.checked?1:0);};
  }
  init().catch(error=>status(error.message||String(error),true));
})();
