'use strict';
// Keep scored polygons intact; selected CAMELYON16 views use review-only geometry.
(() => {
  const ns = 'http://www.w3.org/2000/svg';
  const manifests = new Map();
  const make = (tag, text, attrs = {}) => {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  };
  async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`File unavailable (${response.status}).`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      if (!('DecompressionStream' in window)) {
        throw new Error('This browser cannot open the compressed polygons. Download the saved copy or use a newer browser.');
      }
      return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  function svgNode(tag, attrs) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  }
  function polygonPath(geometry) {
    if (!['Polygon', 'MultiPolygon'].includes(geometry.type)) throw new Error('Unsupported polygon geometry.');
    const parts = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
    return parts.map(polygon => polygon.map(ring =>
      ring.map(([x, y], i) => `${i ? 'L' : 'M'}${x},${y}`).join('') + 'Z').join('')).join('');
  }
  function polygonParts(geometry) {
    return geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  }
  function ringArea(ring) {
    let twiceArea = 0;
    for (let i = 0, previous = ring.length - 1; i < ring.length; previous = i++) {
      twiceArea += ring[previous][0] * ring[i][1] - ring[i][0] * ring[previous][1];
    }
    return Math.abs(twiceArea) / 2;
  }
  function polygonArea(polygon) {
    return Math.max(0, ringArea(polygon[0]) - polygon.slice(1).reduce((sum, hole) => sum + ringArea(hole), 0));
  }
  function boundsOf(ring) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of ring) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    return [minX, minY, maxX, maxY];
  }
  function inRing(ring, x, y) {
    let inside = false;
    for (let i = 0, previous = ring.length - 1; i < ring.length; previous = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[previous];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  function inPolygon(polygon, x, y) {
    return inRing(polygon[0], x, y) && !polygon.slice(1).some(hole => inRing(hole, x, y));
  }
  function formatArea(areaUm2) {
    return `${(areaUm2 / 1e6).toFixed(4)} mm² (${Math.round(areaUm2).toLocaleString()} µm²)`;
  }
  async function init(host) {
    const manifestURL = new URL(host.dataset.manifest, document.baseURI);
    if (!manifests.has(manifestURL.href)) manifests.set(manifestURL.href, fetchJSON(manifestURL));
    const manifest = await manifests.get(manifestURL.href);
    const record = manifest.records.find(row => row.id === host.dataset.record);
    if (!record) throw new Error('Missing tissue entry.');
    const url = path => new URL(path, manifestURL).href;
    const displayFilter = record.section === 'camelyon16' && record.geometry.endsWith('-bora.geojson.gz');
    const refinedReview = record.id === 'benchmark3-tumor_087' && record.review_geometry;
    const minDisplayRedAreaUm2 = 20000;
    const heading = make('h2', refinedReview ? 'Inspect the smoothed review Bora GeoJSON' :
      displayFilter ? 'Inspect the display-filtered Bora GeoJSON' : 'Inspect the simplified GeoJSON');
    const description = displayFilter
      ? refinedReview
        ? `${record.label_description} This review view smooths vector borders over 12 µm and omits polygons smaller than 0.02 mm² in both classes. It is display-only: the original saved Bora vector, scored mask and metrics are unchanged. The 4 µm prediction grid cannot resolve a histological boundary more finely.`
        : `${record.label_description} For readability, this view omits metastasis-like polygons smaller than 0.02 mm² (20,000 µm²). This is a display filter only: the scored mask, metrics and original saved Bora vector are unchanged.`
      : record.label_description;
    const note = make('p', description, {class: 'gi-note'});
    const controls = make('div', '', {class: 'gi-controls'});
    const viewBytes = refinedReview ? record.review_geojson_bytes : record.geojson_bytes;
    const size = viewBytes < 100000 ? `${Math.ceil(viewBytes / 1000)} kB` : `${(viewBytes / 1e6).toFixed(1)} MB`;
    const load = make('button', `Load interactive polygons (${size})`, {type: 'button', class: 'gi-load'});
    const select = make('select', '', {'aria-label': 'Show assigned group', disabled: ''});
    select.append(make('option', 'All groups', {value: ''}));
    const groupLabel = make('label', 'Assigned group ');
    groupLabel.append(select);
    const opacity = make('input', '', {type: 'range', min: '0', max: '90', value: '40', disabled: '', 'aria-label': 'Polygon opacity'});
    const opacityLabel = make('label', 'Opacity ');
    opacityLabel.append(opacity);
    const outlines = make('input', '', {type: 'checkbox', checked: '', disabled: ''});
    const outlineLabel = make('label', ''); outlineLabel.append(outlines, ' Outlines');
    const he = make('input', '', {type: 'checkbox', checked: ''});
    const heLabel = make('label', ''); heLabel.append(he, ' H&E');
    const zoomIn = make('button', 'Zoom +', {type: 'button'});
    const zoomOut = make('button', 'Zoom −', {type: 'button'});
    const fit = make('button', 'Fit', {type: 'button'});
    const boundedImage = record.section === 'camelyon16';
    // Only tumor_087 has a source-TIFF range reader. Other slides retain the
    // existing 8× sampled-preview limit until their own source mapping is checked.
    const sourceDetail = record.id === 'benchmark3-tumor_087' && window.GeoTIFF
      ? record.high_resolution : null;
    if (sourceDetail) host.classList.add('gi-source-detail');
    const maxZoom = sourceDetail ? sourceDetail.maximum_zoom : 8;
    const zoomInfo = make('span', '', {class: 'gi-zoom-info', role: 'status', 'aria-live': 'polite'});
    const download = make('a', displayFilter ? 'Download original saved Bora GeoJSON (.gz)' : 'Download saved GeoJSON' + (record.geometry.endsWith('.gz') ? ' (.gz)' : ''), {href: url(record.geometry), download: ''});
    const displayDownload = displayFilter ? refinedReview
      ? make('a', 'Download smoothed review GeoJSON (.gz)', {href: url(record.review_geometry), download: ''})
      : make('a', 'Download displayed GeoJSON', {href: '#', download: `${record.id}-display-filtered.geojson`, hidden: ''}) : null;
    controls.append(load, groupLabel, opacityLabel, outlineLabel, heLabel, zoomIn, zoomOut, fit);
    if (displayDownload) controls.append(displayDownload);
    controls.append(download);
    if (boundedImage) controls.append(zoomInfo);
    const info = make('p', 'Load the polygons to inspect their assigned groups. The H&E preview is shown below.', {class: 'gi-status', role: 'status', 'aria-live': 'polite'});
    const svg = svgNode('svg', {class: 'gi-map', tabindex: '0', role: 'img', 'aria-label': `${record.title}: interactive assigned groups over H&E. Drag or use arrow keys to pan; plus and minus zoom; zero fits the image.`});
    const [x, y, w, h] = record.image_bounds_xywh;
    const image = svgNode('image', {href: url(record.image), x, y, width: w, height: h, preserveAspectRatio: 'none'});
    svg.append(image);
    const detailImage = sourceDetail ? svgNode('image', {x, y, width: w, height: h, preserveAspectRatio: 'none'}) : null;
    if (detailImage) {
      detailImage.style.display = 'none';
      svg.append(detailImage);
    }
    const legend = make('div', '', {class: 'gi-legend', 'aria-label': 'Assigned group legend'});
    const zoomHelp = sourceDetail
      ? ' This slide zooms to 64×. At 2× and above, the visible H&E window is read at its needed resolution from the official source TIFF; the site does not host the full slide.'
      : boundedImage ? ' CAMELYON16 zoom stops at 8× and panning remains inside the H&E frame.' : '';
    const help = make('p', 'Hover or click a colored region to see the selected geometry area and its full group area. Choose a group to isolate it. Drag to pan; use Zoom or the mouse wheel to magnify. Keyboard: arrows pan, +/− zoom, 0 fits.' + zoomHelp, {class: 'gi-help'});
    const content = [heading, note, controls, info, svg, legend, help];
    if (record.section === 'camelyon16') {
      const slide = /^benchmark3-((?:tumor|normal)_\d{3})(?:-|$)/.exec(record.id)?.[1];
      const original = make('div', '', {class: 'gi-original'});
      original.append(make('h3', 'Original H&E source and overlay-free preview'));
      const frame = make('figure');
      const openPreview = make('a', '', {href: url(record.image), 'aria-label': `Open the H&E image for ${slide || record.title} at its saved preview resolution`});
      openPreview.append(make('img', '', {src: url(record.image), alt: `${slide || record.title} H&E image without GeoJSON polygons`, loading: 'lazy'}));
      frame.append(openPreview, make('figcaption', 'H&E without prediction overlays. This sampled preview uses the GeoJSON viewer’s registered frame; its crop or exterior background can reflect the reviewed tissue display. Open the dataset-hosted TIFF below for the unchanged full-resolution source slide.'));
      original.append(frame);
      if (slide) original.append(make('p', '', {class: 'gi-original-source'}));
      if (slide) original.lastChild.append(make('a', 'Open the original full-resolution H&E slide at CAMELYON16 AWS (large TIFF)', {href: `https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/images/${slide}.tif`}));
      content.push(original);
    }
    host.replaceChildren(...content);
    let box = [x, y, w, h], paths = [], drag = null, loaded = false;
    let detailTimer = null, detailRequest = 0, detailObjectURL = null, sourceTiff = null;
    const sourceTiffForDetail = async () => {
      if (!sourceTiff) sourceTiff = window.GeoTIFF.fromUrl(sourceDetail.url);
      return sourceTiff;
    };
    const renderSourceWindow = async (request, view) => {
      try {
        const tiff = await sourceTiffForDetail();
        const scaleX = sourceDetail.width / w;
        const scaleY = sourceDetail.height / h;
        const sourceBox = [
          Math.max(0, Math.floor((view[0] - x) * scaleX)),
          Math.max(0, Math.floor((view[1] - y) * scaleY)),
          Math.min(sourceDetail.width, Math.ceil((view[0] + view[2] - x) * scaleX)),
          Math.min(sourceDetail.height, Math.ceil((view[1] + view[3] - y) * scaleY))
        ];
        // Keep every source pixel in the narrowest 64× window. Wider windows
        // are sampled only to the viewport's display resolution.
        const wantedWidth = Math.max(1200, Math.min(2048, Math.round(svg.clientWidth * (window.devicePixelRatio || 1))));
        const overviewScale = 2 ** Math.min(4, Math.max(0, Math.floor(Math.log2((sourceBox[2] - sourceBox[0]) / wantedWidth))));
        const overview = await tiff.getImage(Math.log2(overviewScale));
        const sampleWindow = [
          Math.max(0, Math.floor(sourceBox[0] / overviewScale)),
          Math.max(0, Math.floor(sourceBox[1] / overviewScale)),
          Math.min(overview.getWidth(), Math.ceil(sourceBox[2] / overviewScale)),
          Math.min(overview.getHeight(), Math.ceil(sourceBox[3] / overviewScale))
        ];
        const outputWidth = Math.max(1, Math.min(wantedWidth, sampleWindow[2] - sampleWindow[0]));
        const outputHeight = Math.max(1, Math.round(outputWidth * (sampleWindow[3] - sampleWindow[1]) / (sampleWindow[2] - sampleWindow[0])));
        const rgb = await overview.readRGB({window: sampleWindow, width: outputWidth, height: outputHeight, interleave: true});
        if (request !== detailRequest) return;
        const rgba = new Uint8ClampedArray(rgb.width * rgb.height * 4);
        for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
          rgba[j] = rgb[i]; rgba[j + 1] = rgb[i + 1]; rgba[j + 2] = rgb[i + 2]; rgba[j + 3] = 255;
        }
        const canvas = document.createElement('canvas');
        canvas.width = rgb.width; canvas.height = rgb.height;
        canvas.getContext('2d').putImageData(new ImageData(rgba, rgb.width, rgb.height), 0, 0);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .92));
        if (!blob) throw new Error('The source window could not be drawn.');
        if (request !== detailRequest) return;
        const nextURL = URL.createObjectURL(blob);
        const oldURL = detailObjectURL;
        detailObjectURL = nextURL;
        detailImage.setAttribute('href', nextURL);
        detailImage.setAttribute('x', x + sampleWindow[0] * overviewScale / scaleX);
        detailImage.setAttribute('y', y + sampleWindow[1] * overviewScale / scaleY);
        detailImage.setAttribute('width', (sampleWindow[2] - sampleWindow[0]) * overviewScale / scaleX);
        detailImage.setAttribute('height', (sampleWindow[3] - sampleWindow[1]) * overviewScale / scaleY);
        detailImage.style.display = '';
        if (oldURL) URL.revokeObjectURL(oldURL);
        zoomInfo.textContent = `${(w / view[2]).toFixed(1)}× zoom · source H&E detail · ${maxZoom}× limit`;
      } catch (error) {
        if (request !== detailRequest) return;
        console.warn('Source H&E window unavailable:', error);
        zoomInfo.textContent = `${(w / view[2]).toFixed(1)}× zoom · source detail unavailable; preview shown`;
      }
    };
    const scheduleSourceWindow = () => {
      if (!sourceDetail) return;
      clearTimeout(detailTimer);
      const request = ++detailRequest;
      detailImage.style.display = 'none';
      const level = w / box[2];
      if (level < 2 || !he.checked) return;
      const view = box.slice();
      zoomInfo.textContent = `${level.toFixed(1)}× zoom · loading source H&E detail`;
      detailTimer = setTimeout(() => renderSourceWindow(request, view), 160);
    };
    const apply = () => {
      if (boundedImage) {
        box[2] = Math.max(w / maxZoom, Math.min(w, box[2]));
        box[3] = box[2] * h / w;
        box[0] = Math.max(x, Math.min(x + w - box[2], box[0]));
        box[1] = Math.max(y, Math.min(y + h - box[3], box[1]));
        const level = w / box[2];
        zoomInfo.textContent = `${level.toFixed(1)}× zoom · ${maxZoom}× limit`;
        zoomIn.disabled = level >= maxZoom - 1e-8;
        zoomOut.disabled = level <= 1 + 1e-8;
      }
      svg.setAttribute('viewBox', box.join(' '));
      scheduleSourceWindow();
    };
    const reset = () => { box = [x, y, w, h]; apply(); };
    const zoom = (factor, fx = .5, fy = .5) => {
      const width = boundedImage
        ? Math.min(w, Math.max(w / maxZoom, box[2] * factor))
        : Math.min(w * 4, Math.max(w / 500, box[2] * factor));
      const ratio = width / box[2];
      const focusX = boundedImage ? Math.max(0, Math.min(1, fx)) : fx;
      const focusY = boundedImage ? Math.max(0, Math.min(1, fy)) : fy;
      box = [box[0] + box[2] * focusX * (1 - ratio), box[1] + box[3] * focusY * (1 - ratio), width, box[3] * ratio];
      apply();
    };
    const describe = (group, component) => {
      info.textContent = `${group.name} · ${record.label_source}` +
        (component ? ` · Selected geometry area ${formatArea(component.areaPixels * record.native_mpp_xy[0] * record.native_mpp_xy[1])}` : '') +
        (group.area_um2 == null ? '' : ` · Global group area ${(group.area_um2 / 1e6).toFixed(3)} mm²`);
    };
    const pointAt = (clientX, clientY) => {
      const rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      return {
        x: box[0] + (clientX - rect.left) * box[2] / rect.width,
        y: box[1] + (clientY - rect.top) * box[3] / rect.height
      };
    };
    const componentAt = (row, clientX, clientY) => {
      const point = pointAt(clientX, clientY);
      if (!point) return null;
      for (const component of row.components) {
        const [minX, minY, maxX, maxY] = component.bounds;
        if (point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY &&
            inPolygon(component.polygon, point.x, point.y)) {
          if (component.areaPixels == null) component.areaPixels = polygonArea(component.polygon);
          return component;
        }
      }
      return null;
    };
    const style = () => {
      for (const {path, group} of paths) {
        path.style.display = !select.value || select.value === String(group.value) ? '' : 'none';
        path.setAttribute('fill-opacity', opacity.value / 100);
        path.setAttribute('stroke', outlines.checked ? '#18303b' : 'none');
      }
      for (const button of legend.children) button.setAttribute('aria-pressed', String(select.value === button.dataset.value));
    };
    const choose = value => {
      select.value = String(value);
      style();
      const group = record.groups.find(row => String(row.value) === select.value);
      if (group) {
        describe(group);
        if (host.dataset.focusGroup === 'true') {
          const matching = paths.filter(row => row.group.value === group.value);
          const bounds = matching.flatMap(row => row.components.map(part => part.bounds));
          if (bounds.length) {
            const left = Math.min(...bounds.map(row => row[0]));
            const top = Math.min(...bounds.map(row => row[1]));
            const right = Math.max(...bounds.map(row => row[2]));
            const bottom = Math.max(...bounds.map(row => row[3]));
            const width = Math.max(1, right - left) * 1.25;
            const height = Math.max(1, bottom - top) * 1.25;
            const fittedWidth = Math.max(width, height * w / h);
            const fittedHeight = fittedWidth * h / w;
            box = [(left + right - fittedWidth) / 2, (top + bottom - fittedHeight) / 2,
                   fittedWidth, fittedHeight];
            apply();
          }
        }
      }
      else info.textContent = `${record.groups.length} assigned groups across ${record.source_feature_count} polygons · ${record.label_source}. Hover or click a polygon to inspect it.`;
    };
    select.onchange = () => choose(select.value);
    opacity.oninput = style; outlines.onchange = style;
    he.onchange = () => {
      image.style.display = he.checked ? '' : 'none';
      if (detailImage) scheduleSourceWindow();
    };
    fit.onclick = reset;
    zoomIn.onclick = () => zoom(.7); zoomOut.onclick = () => zoom(1 / .7);
    // The fixed viewport matches the exact saved overview frame, including cropped WSIs.
    svg.style.aspectRatio = `${w} / ${h}`;
    reset();
    svg.addEventListener('wheel', event => {
      if (event.ctrlKey) return;
      event.preventDefault();
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      zoom(Math.exp(Math.max(-.4, Math.min(.4, event.deltaY * .002))), (point.x - box[0]) / box[2], (point.y - box[1]) / box[3]);
    }, {passive: false});
    svg.onkeydown = event => {
      if (['+', '=', '-', '0'].includes(event.key)) {
        event.preventDefault();
        if (event.key === '0') reset(); else zoom(event.key === '-' ? 1 / .7 : .7);
      } else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
        event.preventDefault();
        box[0] += (event.key === 'ArrowLeft' ? -.1 : event.key === 'ArrowRight' ? .1 : 0) * box[2];
        box[1] += (event.key === 'ArrowUp' ? -.1 : event.key === 'ArrowDown' ? .1 : 0) * box[3];
        apply();
      }
    };
    svg.onpointerdown = event => {
      if (event.button !== 0) return;
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      const inverse = matrix.inverse();
      drag = {start: new DOMPoint(event.clientX, event.clientY).matrixTransform(inverse), inverse, box: box.slice(), target: event.target, moved: false, clientX: event.clientX, clientY: event.clientY};
      svg.setPointerCapture(event.pointerId);
    };
    svg.onpointermove = event => {
      if (!drag) return;
      if (Math.hypot(event.clientX - drag.clientX, event.clientY - drag.clientY) < 4 && !drag.moved) return;
      drag.moved = true;
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(drag.inverse);
      box = [drag.box[0] - (point.x - drag.start.x), drag.box[1] - (point.y - drag.start.y), drag.box[2], drag.box[3]];
      apply();
    };
    svg.onpointerup = event => {
      if (drag && !drag.moved && loaded) {
        const picked = paths.find(row => row.path === drag.target);
        if (picked) describe(picked.group, componentAt(picked, drag.clientX, drag.clientY));
        else info.textContent = 'No exported polygon at this location. This does not establish that the image contains background here.';
      }
      drag = null;
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    };
    svg.onpointercancel = svg.onlostpointercapture = () => { drag = null; };
    load.onclick = async () => {
      load.disabled = true; host.setAttribute('aria-busy', 'true');
      info.textContent = 'Loading saved polygons… Large whole slides may take a few seconds.';
      try {
        const vector = await fetchJSON(url(refinedReview ? record.review_geometry : record.geometry));
        if (vector.type !== 'FeatureCollection' || vector.features.length !==
            (refinedReview ? record.review_feature_count : record.source_feature_count)) throw new Error('Unexpected polygon file.');
        if (refinedReview && (!vector.display_only || vector.source_saved_geojson_sha256 !== record.stored_geojson_sha256))
          throw new Error('Review vector does not match the saved Bora geometry.');
        const visible = refinedReview ? vector.features : displayFilter ? vector.features.filter(feature => {
          if (feature.properties.value !== 2) return true;
          const area = Number(feature.properties.area_um2);
          if (!Number.isFinite(area)) throw new Error('Missing physical area for a Bora polygon.');
          return area >= minDisplayRedAreaUm2;
        }) : vector.features;
        if (displayDownload && !refinedReview) {
          const omitted = vector.features.length - visible.length;
          const omittedArea = vector.features.reduce((sum, feature) =>
            sum + (feature.properties.value === 2 && Number(feature.properties.area_um2) < minDisplayRedAreaUm2
              ? Number(feature.properties.area_um2) : 0), 0);
          const displayed = {
            type: 'FeatureCollection',
            coordinate_space: vector.coordinate_space,
            native_shape_yx: vector.native_shape_yx,
            native_mpp_xy: vector.native_mpp_xy,
            display_only: true,
            source_scored_geojson_sha256: record.stored_geojson_sha256,
            display_filter: {class_value: 2, minimum_area_um2: minDisplayRedAreaUm2, omitted_polygons: omitted, omitted_area_um2: omittedArea},
            features: visible
          };
          displayDownload.href = URL.createObjectURL(new Blob([JSON.stringify(displayed)], {type: 'application/geo+json'}));
          displayDownload.hidden = false;
        }
        const staged = [];
        for (const feature of visible) {
          const group = record.groups.find(row => row.value === feature.properties.value);
          if (!group) throw new Error('Unrecognized saved group.');
          const path = svgNode('path', {d: polygonPath(feature.geometry), fill: group.color, 'fill-rule': 'evenodd', 'vector-effect': 'non-scaling-stroke', 'stroke-width': '.7'});
          const title = svgNode('title', {}); title.textContent = `${group.name} · ${record.label_source}`; path.append(title);
          const row = {path, group, components: polygonParts(feature.geometry).map(polygon => ({polygon, bounds: boundsOf(polygon[0]), areaPixels: null}))};
          const show = event => { if (!drag) describe(group, componentAt(row, event.clientX, event.clientY)); };
          path.onpointerenter = show;
          path.onpointermove = show;
          staged.push(row);
          await new Promise(resolve => requestAnimationFrame(resolve));
        }
        paths = staged;
        svg.append(...paths.map(row => row.path));
        for (const group of record.groups) {
          select.append(make('option', group.name, {value: group.value}));
          const button = make('button', '', {type: 'button', 'data-value': group.value, 'aria-pressed': 'false'});
          const swatch = make('i', '', {'aria-hidden': 'true'}); swatch.style.backgroundColor = group.color;
          button.append(swatch, group.name); button.onclick = () => choose(select.value === String(group.value) ? '' : group.value);
          legend.append(button);
        }
        loaded = true;
        select.disabled = opacity.disabled = outlines.disabled = false;
        load.hidden = true;
        choose('');
        if (refinedReview) info.textContent = `${visible.length} smoothed review polygons · ${record.review_omitted_count} small polygons omitted across both classes below 0.02 mm². Original vector and scores are unchanged.`;
        else if (displayFilter) info.textContent = `${visible.length} displayed polygons · ${vector.features.length - visible.length} small metastasis-like polygons omitted below 0.02 mm². Original saved vector and scores are unchanged.`;
      } catch (error) {
        info.textContent = `The interactive polygons could not load. ${error.message} You can retry or use the saved GeoJSON download.`;
        load.disabled = false;
      } finally {
        host.removeAttribute('aria-busy');
      }
    };
  }
  document.querySelectorAll('.geojson-inspector').forEach(host => init(host).catch(error => {
    host.append(make('p', 'The inspector could not initialize. ' + error.message, {role: 'alert'}));
  }));
})();
