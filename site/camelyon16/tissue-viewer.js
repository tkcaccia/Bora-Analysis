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
  const state = {viewer:null, features:[], opacity:.16, outlines:true, selected:'', queued:false,
    preview:null, displayPath:null, tissuePoints:[], sourceWarning:false, detailRequests:0,
    overlayHidden:false};

  function detailStatus() {
    const node=$('detail-loading');
    node.hidden=state.detailRequests===0;
    if (!node.hidden) node.textContent=`Loading source detail… ${state.detailRequests} tile${state.detailRequests===1?'':'s'}`;
  }

  // A separate visual contour fit softens the 4 µm raster stair steps. The
  // saved review GeoJSON and all measured areas remain byte-for-byte intact.
  function smoothRing(ring) {
    let points=ring.slice(0,-1);
    if (points.length<5) return ring;
    for (let pass=0; pass<2; pass++) {
      const next=[];
      for (let i=0; i<points.length; i++) {
        const a=points[i], b=points[(i+1)%points.length];
        next.push([.75*a[0]+.25*b[0],.75*a[1]+.25*b[1]],
                  [.25*a[0]+.75*b[0],.25*a[1]+.75*b[1]]);
      }
      points=next;
    }
    return [...points,points[0]];
  }
  function smoothGeometry(geometry) {
    if (geometry.type==='Polygon') return {...geometry,coordinates:geometry.coordinates.map(smoothRing)};
    if (geometry.type==='MultiPolygon') return {...geometry,coordinates:geometry.coordinates.map(p=>p.map(smoothRing))};
    return geometry;
  }

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
  if (window.frameElement) {
    document.documentElement.classList.add('embedded');
    const sizeFrame=()=>{
      const height=Math.ceil(document.querySelector('.shell').getBoundingClientRect().height)+4;
      if (Math.abs(window.frameElement.getBoundingClientRect().height-height)>2)
        window.frameElement.style.height=`${height}px`;
    };
    new ResizeObserver(sizeFrame).observe(document.querySelector('.shell'));
    addEventListener('load',sizeFrame);
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
    if (state.overlayHidden) return;
    ctx.setTransform(dpr*sx,0,0,dpr*sy,dpr*p0.x,dpr*p0.y);
    ctx.save(); ctx.clip(state.displayPath,'evenodd');
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

  function tileSource(record, crop, preview, displayPath) {
    const tileSize=512, maxLevel=Math.ceil(Math.log2(Math.max(crop.w,crop.h)));
    let tiffPromise=null, imageCountPromise=null;
    const imagePromises=new Map(), levelInfos=new Map();
    const finishCanvas=(context,tile)=>{
      const image=new Image();
      image.onload=()=>{if (!context.userData?.aborted) context.finish(image,null,'image');};
      image.onerror=()=>{if (!context.userData?.aborted) context.finish(null,'Could not render a tissue tile.');};
      image.src=tile.toDataURL('image/png');
    };
    const source=async()=>{
      if (!tiffPromise) tiffPromise=GeoTIFF.fromUrl(record.high_resolution.url)
        .catch(error=>{tiffPromise=null;throw error;});
      return tiffPromise;
    };
    const levelAt=async index=>{
      if (!levelInfos.has(index)) {
        const info=(async()=>{
          const tiff=await source();
          if (!imagePromises.has(index)) imagePromises.set(index,tiff.getImage(index));
          const image=await imagePromises.get(index);
          const ratioX=record.high_resolution.width/image.getWidth();
          const ratioY=record.high_resolution.height/image.getHeight();
          if (!(ratioX>=1 && ratioY>=1 && Number.isFinite(ratioX) && Number.isFinite(ratioY)))
            throw new Error(`Invalid official TIFF pyramid level ${index}.`);
          return {index,ratioX,ratioY,downsample:Math.max(ratioX,ratioY)};
        })().catch(error=>{levelInfos.delete(index);imagePromises.delete(index);throw error;});
        levelInfos.set(index,info);
      }
      return levelInfos.get(index);
    };
    const chooseLevel=async factor=>{
      if (!imageCountPromise) imageCountPromise=source().then(tiff=>tiff.getImageCount())
        .catch(error=>{imageCountPromise=null;throw error;});
      const count=await imageCountPromise;
      if (!count) throw new Error('The official TIFF has no image levels.');
      // Start near the usual 2x overview, then check actual dimensions. This
      // needs only nearby IFDs even when the TIFF has an irregular pyramid.
      let index=Math.min(count-1,Math.max(0,Math.floor(Math.log2(factor))));
      let info=await levelAt(index);
      while (info.downsample>factor*1.01 && index>0) info=await levelAt(--index);
      while (index+1<count) {
        const next=await levelAt(index+1);
        if (next.downsample<=info.downsample || next.downsample>factor*1.01) break;
        info=next; index++;
      }
      return info;
    };
    const previewTile=(context, box, outW, outH)=>{
      const tile=document.createElement('canvas'); tile.width=outW; tile.height=outH;
      const c=tile.getContext('2d');
      c.imageSmoothingEnabled=true; c.imageSmoothingQuality='high';
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
        if (factor>32) {previewTile(context,box,outW,outH); return;}
        state.detailRequests++; detailStatus();
        try {
          const levelInfo=await chooseLevel(factor);
          const image=await imagePromises.get(levelInfo.index);
          const ratioX=levelInfo.ratioX, ratioY=levelInfo.ratioY;
          const window=[Math.floor((crop.x+box.x)/ratioX),Math.floor((crop.y+box.y)/ratioY),
            Math.min(image.getWidth(),Math.ceil((crop.x+box.x+box.w)/ratioX)),
            Math.min(image.getHeight(),Math.ceil((crop.y+box.y+box.h)/ratioY))];
          const rgb=await image.readRGB({window,width:outW,height:outH,
            interleave:true,resampleMethod:'bilinear'});
          const tile=document.createElement('canvas'); tile.width=outW; tile.height=outH;
          const c=tile.getContext('2d');
          const rgba=new Uint8ClampedArray(outW*outH*4);
          for(let i=0,j=0;i<rgb.length;i+=3,j+=4){rgba[j]=rgb[i];rgba[j+1]=rgb[i+1];rgba[j+2]=rgb[i+2];rgba[j+3]=255;}
          c.putImageData(new ImageData(rgba,outW,outH),0,0);
          c.globalCompositeOperation='destination-in';
          c.setTransform(1/factor,0,0,1/factor,-box.x/factor,-box.y/factor);
          c.fillStyle='#fff'; c.fill(displayPath,'evenodd');
          c.setTransform(1,0,0,1,0,0);
          c.globalCompositeOperation='source-over';
          finishCanvas(context,tile);
        } catch(error) {
          // A transient range failure must not disable source detail for every
          // later tile in this session. Only the affected tile uses the preview.
          if (!state.sourceWarning) {
            state.sourceWarning=true;
            status('One source-detail tile fell back to the tissue preview; reload to retry. '+error.message);
          }
          previewTile(context,box,outW,outH);
        } finally {
          state.detailRequests=Math.max(0,state.detailRequests-1); detailStatus();
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
    const preview=new Image();
    preview.src=`tissue-viewer-previews/${slide}-tissue.webp`;
    await preview.decode();
    const displayMask=new Image();
    displayMask.src=`tissue-viewer-previews/${slide}-mask.png`;
    await displayMask.decode();
    const displaySupport=await json(`tissue-viewer-previews/${slide}-display-support.geojson.gz`);
    if (displaySupport.coordinate_system!=='viewer_preview_pixel_xy' ||
        displaySupport.source_sha256!==record.high_resolution.source_sha256 ||
        displaySupport.width!==displayMask.naturalWidth ||
        displaySupport.height!==displayMask.naturalHeight || !displaySupport.features?.length)
      throw new Error('The display tissue geometry is not registered to the source.');
    if (preview.naturalWidth!==displayMask.naturalWidth || preview.naturalHeight!==displayMask.naturalHeight)
      throw new Error('The tissue overview and its display mask are not registered.');
    const maskCanvas=document.createElement('canvas');
    maskCanvas.width=displayMask.naturalWidth; maskCanvas.height=displayMask.naturalHeight;
    const maskContext=maskCanvas.getContext('2d',{willReadFrequently:true});
    maskContext.drawImage(displayMask,0,0);
    const alpha=maskContext.getImageData(0,0,maskCanvas.width,maskCanvas.height).data;
    let minX=maskCanvas.width,minY=maskCanvas.height,maxX=-1,maxY=-1;
    for (let y=0;y<maskCanvas.height;y++) for (let x=0;x<maskCanvas.width;x++) {
      if (alpha[(y*maskCanvas.width+x)*4+3]<16) continue;
      minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y);
      if (x%16===0 && y%16===0 && alpha[(y*maskCanvas.width+x)*4+3]>127)
        state.tissuePoints.push([x/maskCanvas.width,y/maskCanvas.height]);
    }
    if (maxX<minX) throw new Error('No display tissue was found.');
    const [bx,by,bw,bh]=record.image_bounds_xywh;
    const scaleX=record.high_resolution.width/record.high_resolution.calibrated_width;
    const scaleY=record.high_resolution.height/record.high_resolution.calibrated_height;
    const crop={x:bx*scaleX,y:by*scaleY,w:bw*scaleX,h:bh*scaleY};
    const displayPath=new Path2D();
    const toCrop=(x,y)=>[x/displayMask.naturalWidth*crop.w,y/displayMask.naturalHeight*crop.h];
    for (const feature of displaySupport.features) {
      const geometry=smoothGeometry(feature.geometry);
      const polygons=geometry.type==='Polygon'?[geometry.coordinates]:geometry.coordinates;
      for (const polygon of polygons) for (const ring of polygon) {
        ring.forEach(([x,y],i)=>{const p=toCrop(x,y);if(i)displayPath.lineTo(p[0],p[1]);else displayPath.moveTo(p[0],p[1]);});
        displayPath.closePath();
      }
    }
    const margin=0.035;
    const tissueBounds={x:Math.max(0,(minX/maskCanvas.width-margin)*crop.w),
      y:Math.max(0,(minY/maskCanvas.height-margin)*crop.h),
      right:Math.min(crop.w,((maxX+1)/maskCanvas.width+margin)*crop.w),
      bottom:Math.min(crop.h,((maxY+1)/maskCanvas.height+margin)*crop.h)};
    const transform=(x,y)=>[(x-bx)*scaleX,(y-by)*scaleY];
    state.displayPath=displayPath; state.cropW=crop.w; state.cropH=crop.h;
    state.mppX=record.native_mpp_xy[0]/scaleX; state.preview=preview;
    const displayed=vector.features.filter(feature=>feature.properties.value!==2 || Number(feature.properties.area_um2)>=20000);
    const visual=displayed.map(feature=>({...feature,geometry:smoothGeometry(feature.geometry)}));
    const groups=new Map(record.groups.map(g=>[g.value,g]));
    state.features=visual.map(feature=>{
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
      display_filter:{class_value:2,minimum_area_um2:20000,omitted_polygons:vector.features.length-displayed.length},
      visual_contour_fit:'two-pass Chaikin; display only; saved review geometry unchanged',features:visual};
    $('download-displayed').href=URL.createObjectURL(new Blob([JSON.stringify(displayCopy)],{type:'application/geo+json'}));
    $('download-displayed').download=`${slide}-bora-display.geojson`;$('download-displayed').hidden=false;
    const viewer=OpenSeadragon({element:$('osd'),tileSources:tileSource(record,crop,preview,displayPath),
      showNavigationControl:false,showNavigator:false,minZoomImageRatio:.8,maxZoomPixelRatio:1.5,
      animationTime:.25,blendTime:.35,alwaysBlend:true,imageLoaderLimit:8,
      gestureSettingsMouse:{clickToZoom:false,dblClickToZoom:true,scrollToZoom:true,dragToPan:true},
      gestureSettingsTouch:{pinchToZoom:true,dragToPan:true}});
    state.viewer=viewer;
    const fitTissue=()=>{
      const item=viewer.world.getItemAt(0);
      if (!item) return;
      const a=item.imageToViewportCoordinates(tissueBounds.x,tissueBounds.y);
      const b=item.imageToViewportCoordinates(tissueBounds.right,tissueBounds.bottom);
      viewer.viewport.fitBounds(new OpenSeadragon.Rect(a.x,a.y,b.x-a.x,b.y-a.y),true);
    };
    const zoomIntoTissue=()=>{
      const item=viewer.world.getItemAt(0);
      if (!item || !state.tissuePoints.length) return;
      const center=viewer.viewport.getCenter();
      const p0=item.imageToViewportCoordinates(0,0);
      const p1=item.imageToViewportCoordinates(crop.w,crop.h);
      const nx=(center.x-p0.x)/(p1.x-p0.x), ny=(center.y-p0.y)/(p1.y-p0.y);
      let best=state.tissuePoints[0], distance=Infinity;
      for (const point of state.tissuePoints) {
        const d=(point[0]-nx)**2+(point[1]-ny)**2;
        if (d<distance) {distance=d; best=point;}
      }
      const focus=item.imageToViewportCoordinates(best[0]*crop.w,best[1]*crop.h);
      viewer.viewport.panTo(focus,true);
      viewer.viewport.zoomBy(1.5,focus,true);
      viewer.viewport.applyConstraints();
    };
    viewer.addHandler('open',()=>{
      $('loading').hidden=true;
      fitTissue();
      status(`${displayed.length} displayed polygons; ${vector.features.length-displayed.length} small red regions hidden. H&E display support is refined from source color; saved predictions and scores are unchanged.`);
      schedule();
    });
    viewer.addHandler('open-failed',event=>status(`Tissue image could not open: ${event.message||'unknown error'}`,true));
    viewer.addHandler('tile-load-failed',()=>status('A tissue tile could not load; retry zooming or reload.',true));
    for (const name of ['animation','animation-finish','resize','update-viewport']) viewer.addHandler(name,schedule);
    addEventListener('resize',schedule);
    $('zoom-in').onclick=zoomIntoTissue;
    $('zoom-out').onclick=()=>{viewer.viewport.zoomBy(1/1.5);viewer.viewport.applyConstraints();};
    $('fit').onclick=fitTissue;
    $('class').onchange=event=>{state.selected=event.target.value;schedule();};
    $('opacity').oninput=event=>{state.opacity=Number(event.target.value);$('opacity-value').textContent=`${Math.round(state.opacity*100)}%`;schedule();};
    $('outlines').onchange=event=>{state.outlines=event.target.checked;schedule();};
    $('he').onchange=event=>{viewer.world.getItemAt(0).setOpacity(event.target.checked?1:0);};
    $('toggle-overlay').onclick=()=>{
      state.overlayHidden=!state.overlayHidden;
      if (state.overlayHidden && !$('he').checked) {
        $('he').checked=true;
        viewer.world.getItemAt(0).setOpacity(1);
      }
      $('toggle-overlay').setAttribute('aria-pressed',String(state.overlayHidden));
      $('toggle-overlay').textContent=state.overlayHidden?'Show Bora contours':'Inspect H&E only';
      schedule();
    };
  }
  init().catch(error=>status(error.message||String(error),true));
})();
