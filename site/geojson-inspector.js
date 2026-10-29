'use strict';
// Reuse saved native polygons; no classification or geometry changes in the browser.
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
  async function init(host) {
    const manifestURL = new URL(host.dataset.manifest, document.baseURI);
    if (!manifests.has(manifestURL.href)) manifests.set(manifestURL.href, fetchJSON(manifestURL));
    const manifest = await manifests.get(manifestURL.href);
    const record = manifest.records.find(row => row.id === host.dataset.record);
    if (!record) throw new Error('Missing tissue entry.');
    const url = path => new URL(path, manifestURL).href;
    const heading = make('h2', 'Inspect the simplified GeoJSON');
    const note = make('p', record.label_description, {class: 'gi-note'});
    const controls = make('div', '', {class: 'gi-controls'});
    const load = make('button', `Load interactive polygons (${(record.geojson_bytes / 1e6).toFixed(1)} MB)`, {type: 'button', class: 'gi-load'});
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
    const download = make('a', 'Download saved GeoJSON' + (record.geometry.endsWith('.gz') ? ' (.gz)' : ''), {href: url(record.geometry), download: ''});
    controls.append(load, groupLabel, opacityLabel, outlineLabel, heLabel, zoomIn, zoomOut, fit, download);
    const info = make('p', 'Load the polygons to inspect their assigned groups. The H&E preview is shown below.', {class: 'gi-status', role: 'status', 'aria-live': 'polite'});
    const svg = svgNode('svg', {class: 'gi-map', tabindex: '0', role: 'img', 'aria-label': `${record.title}: interactive assigned groups over H&E. Drag or use arrow keys to pan; plus and minus zoom; zero fits the image.`});
    const [x, y, w, h] = record.image_bounds_xywh;
    const image = svgNode('image', {href: url(record.image), x, y, width: w, height: h, preserveAspectRatio: 'none'});
    svg.append(image);
    const legend = make('div', '', {class: 'gi-legend', 'aria-label': 'Assigned group legend'});
    const help = make('p', 'Hover or click a colored region for its assigned group. Choose a group to isolate it. Drag to pan; use Zoom or the mouse wheel to magnify. Keyboard: arrows pan, +/− zoom, 0 fits.', {class: 'gi-help'});
    host.replaceChildren(heading, note, controls, info, svg, legend, help);
    let box = [x, y, w, h], paths = [], drag = null, loaded = false;
    const apply = () => svg.setAttribute('viewBox', box.join(' '));
    const reset = () => { box = [x, y, w, h]; apply(); };
    const zoom = (factor, fx = .5, fy = .5) => {
      const width = Math.min(w * 4, Math.max(w / 500, box[2] * factor));
      const ratio = width / box[2];
      box = [box[0] + box[2] * fx * (1 - ratio), box[1] + box[3] * fy * (1 - ratio), width, box[3] * ratio];
      apply();
    };
    const describe = group => {
      info.textContent = `${group.name} · ${record.label_source}` +
        (group.area_um2 == null ? '' : ` · Global group area ${(group.area_um2 / 1e6).toFixed(3)} mm²`);
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
      if (group) describe(group);
      else info.textContent = `${record.source_feature_count} assigned groups · ${record.label_source}. Hover or click a polygon to inspect it.`;
    };
    select.onchange = () => choose(select.value);
    opacity.oninput = style; outlines.onchange = style;
    he.onchange = () => image.style.display = he.checked ? '' : 'none';
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
        if (picked) describe(picked.group);
        else info.textContent = 'No exported Bora polygon at this location. This does not establish that the image contains background here.';
      }
      drag = null;
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    };
    svg.onpointercancel = svg.onlostpointercapture = () => { drag = null; };
    load.onclick = async () => {
      load.disabled = true; host.setAttribute('aria-busy', 'true');
      info.textContent = 'Loading saved native polygons… Large whole slides may take a few seconds.';
      try {
        const vector = await fetchJSON(url(record.geometry));
        if (vector.type !== 'FeatureCollection' || vector.features.length !== record.source_feature_count) throw new Error('Unexpected polygon file.');
        const staged = [];
        for (const feature of vector.features) {
          const group = record.groups.find(row => row.value === feature.properties.value);
          if (!group) throw new Error('Unrecognized saved group.');
          const path = svgNode('path', {d: polygonPath(feature.geometry), fill: group.color, 'fill-rule': 'evenodd', 'vector-effect': 'non-scaling-stroke', 'stroke-width': '.7'});
          const title = svgNode('title', {}); title.textContent = `${group.name} · ${record.label_source}`; path.append(title);
          path.onpointerenter = () => { if (!drag) describe(group); };
          staged.push({path, group});
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
