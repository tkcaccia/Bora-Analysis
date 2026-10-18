'use strict';
let data, shown = [], idx = 0, model, box, drag, token = 0;
const el = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const pc = x => x == null ? 'n/a' : (100 * x).toFixed(1) + '%';
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Cannot load ${url} (${r.status})`);
  return r.json();
}
function fail(error) {
  el('status').className = 'error';
  el('status').textContent = 'The viewer could not load a required file. ' + error.message;
}
async function init() {
  data = await json('data.json');
  const archive = data.page_kind === 'comparison';
  document.body.classList.add(archive ? 'archive' : 'selected');
  document.title = archive ? 'Bora · classifier comparison' : 'Bora · Neighbor context benchmark';
  el('title').textContent = archive ? 'Classifier comparison' : 'Neighbor context';
  el('mask-title').textContent = archive ? 'Final classified masks, side by side' : 'Final classified tissue mask · Neighbor context';
  el('intro').textContent = archive
    ? 'Compare tissue identification methods on the same images and fixed segmentation geometry. The main Bora workflow uses Neighbor context.'
    : 'A complete visual record of Bora refinement and tissue-group identification on 116 calibrated histopathology images.';
  const chosen = data.models.find(m => m.id === 'neighbor-context');
  el('metrics').innerHTML = `<div class="metric"><small>Mean final GeoJSON IoU</small><strong>${chosen.mean_final_geojson_iou.toFixed(3)}</strong><small>Mean across all ${data.cohort_cases} images</small></div><div class="metric"><small>Whole-group accuracy</small><strong>${pc(chosen.top1)}</strong><small>Balanced accuracy ${pc(chosen.balanced_accuracy)}</small></div><div class="metric"><small>Mean cached deployment</small><strong>${chosen.mean_cached_deployment_seconds.toFixed(2)} s</strong><small>Includes recorded cleanup and fitting; encoder excluded</small></div>`;
  el('scope').textContent = `${data.cohort_cases} multiclass images: BCSS ${data.dataset_counts.bcss}, TIGER ${data.dataset_counts.tiger}, GCSS ${data.dataset_counts.gcss}, IGNITE ${data.dataset_counts.ignite}. Original split labels are retained (${data.split_counts.test} test, ${data.split_counts.validation} validation). This exploratory benchmark is not an independent final validation. PUMA is excluded from evaluation.`;
  el('status').textContent = `Loaded ${data.cohort_cases} images. Choose a dataset or image below.`;
  el('score-label').textContent = archive ? 'Aggregate classifier scores' : 'Aggregate scores for Neighbor context';
  el('score').innerHTML = '<tr><th>Classifier</th><th>Final IoU</th><th>Group accuracy</th><th>Balanced accuracy</th><th>Cached s/image</th></tr>' + data.models.map(m => `<tr><td>${esc(m.name)}</td><td>${m.mean_final_geojson_iou.toFixed(3)}</td><td>${pc(m.top1)}</td><td>${pc(m.balanced_accuracy)}</td><td>${m.mean_cached_deployment_seconds.toFixed(3)}</td></tr>`).join('');
  el('mechanisms').innerHTML = '<tr><th>Classifier</th><th>Mechanism</th></tr>' + data.mechanisms.map(m => `<tr><td>${esc(m.name)}</td><td>${esc(m.description)}</td></tr>`).join('');
  el('dataset').innerHTML = '<option value="">All datasets</option>' + [...new Set(data.cases.map(c => c.dataset))].sort().map(s => `<option value="${esc(s)}">${esc(s.toUpperCase())}</option>`).join('');
  model = data.models[0].id;
  el('method').innerHTML = data.models.map(m => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join('');
  el('method').onchange = () => { model = el('method').value; draw().catch(fail); };
  el('dataset').onchange = () => filter();
  el('case').onchange = () => select(+el('case').value);
  el('prev').onclick = () => select((idx - 1 + shown.length) % shown.length);
  el('next').onclick = () => select((idx + 1) % shown.length);
  el('opacity').oninput = style;
  el('outlines').onchange = style;
  el('fit').onclick = fit;
  el('zoom-in').onclick = () => zoom(.7);
  el('zoom-out').onclick = () => zoom(1.43);
  window.addEventListener('hashchange', () => {
    const wanted = decodeURIComponent(location.hash.slice(1)).split('/')[0];
    if (shown[idx] && shown[idx].case === wanted) return;
    const found = shown.findIndex(c => c.case === wanted);
    if (found >= 0) select(found);
  });
  filter(decodeURIComponent(location.hash.slice(1)).split('/')[0]);
}
function filter(wanted) {
  shown = data.cases.filter(c => !el('dataset').value || c.dataset === el('dataset').value);
  el('case').innerHTML = shown.map((c, i) => `<option value="${i}">${esc(c.case)}</option>`).join('');
  select(Math.max(0, shown.findIndex(c => c.case === wanted)));
}
function expertKey(c) {
  return '<p>' + c.expert_legend.map(g => `<span><i class="swatch" style="background:${g.role ? 'rgb(' + data.colours[g.role] + ')' : '#eee'}"></i>${esc(g.source_name)}${g.scored ? '' : ' (excluded)'}</span>`).join('<br>') + '</p><small>Unannotated and unmapped pixels do not enter quality scores.</small>';
}
function select(i) {
  idx = i;
  const c = shown[i];
  location.hash = c.case;
  el('case').value = i;
  el('info').textContent = `${i + 1}/${shown.length} · original split: ${c.split} · ${c.native_shape_yx[1]} × ${c.native_shape_yx[0]} native pixels · x/y: ${c.native_mpp_xy.join(' × ')} µm/px`;
  const panels = [['Original H&E', 'original'], ['Expert tissue annotation', 'expert'], ['Input tiles: anonymous expert-derived groups', 'tiles'], ['Fixed Virchow2 RGB conversion', 'converted'], ['Bora-AF anonymous tissue groups', 'bora']];
  el('context').innerHTML = panels.map(([label, file]) => `<figure><figcaption>${label}</figcaption><a target="_blank" rel="noopener" href="${c.asset}/${file}.jpg"><img loading="lazy" src="${c.asset}/${file}.jpg" alt="${esc(c.case + ': ' + label)}"></a>${file === 'expert' ? expertKey(c) : ''}</figure>`).join('');
  el('methods').innerHTML = c.models.map(m => `<figure data-model="${esc(m.id)}"><figcaption>${esc(m.name)}</figcaption><p><small>Final IoU ${m.iou.toFixed(3)} · raw IoU ${m.raw_iou.toFixed(3)}<br>Group accuracy ${pc(m.group_top1)} · ${m.seconds.toFixed(3)} cached s/image<br>Boundary segments ${m.boundary_segments_before.toLocaleString()} → ${m.boundary_segments_after.toLocaleString()}<br>Filled ${m.cleanup.filled_regions} small regions and ${m.cleanup.filled_holes} enclosed holes</small></p><a target="_blank" rel="noopener" href="${m.preview}"><img loading="lazy" src="${m.preview}" alt="${esc(c.case + ': ' + m.name + ' final classified mask')}"></a><button type="button" data-select="${esc(m.id)}">Inspect GeoJSON</button></figure>`).join('');
  document.querySelectorAll('[data-select]').forEach(b => b.onclick = () => { model = b.dataset.select; draw().catch(fail); });
  el('legend').innerHTML = data.roles.map((r, j) => `<span><i class="swatch" style="background:rgb(${data.colours[j + 1]})"></i>${esc(r.replaceAll('_', ' '))}</span>`).join(' ');
  draw().catch(fail);
}
async function draw() {
  const c = shown[idx], m = c.models.find(m => m.id === model) || c.models[0];
  model = m.id;
  el('method').value = model;
  document.querySelectorAll('[data-model]').forEach(f => f.classList.toggle('active', f.dataset.model === model));
  el('download').href = m.vector;
  el('download').download = `${c.case}-${m.id}.geojson`;
  el('vector-info').textContent = 'Loading native ROI polygons…';
  const t = ++token, g = await json(m.vector);
  if (t !== token) return;
  const svg = el('geo');
  svg.replaceChildren();
  const image = document.createElementNS(ns, 'image');
  image.setAttribute('href', c.asset + '/original.jpg');
  image.setAttribute('width', c.native_shape_yx[1]);
  image.setAttribute('height', c.native_shape_yx[0]);
  image.setAttribute('preserveAspectRatio', 'none');
  svg.append(image);
  for (const f of g.features) {
    const p = document.createElementNS(ns, 'path');
    const parts = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
    p.setAttribute('d', parts.flatMap(poly => poly.map(r => r.map((xy, i) => (i ? 'L' : 'M') + xy.join(',')).join(' ') + 'Z')).join(' '));
    p.setAttribute('fill-rule', 'evenodd');
    p.setAttribute('fill', f.properties.classification.color);
    p.setAttribute('vector-effect', 'non-scaling-stroke');
    p.dataset.vector = 'true';
    const title = document.createElementNS(ns, 'title');
    title.textContent = f.properties.predicted_tissue;
    p.append(title);
    p.onpointerenter = () => { el('vector-info').textContent = f.properties.predicted_tissue; };
    svg.append(p);
  }
  fit(); style();
  el('vector-info').textContent = `${g.features.length} predicted tissue features · native ROI pixel coordinates · simplified shared boundaries. Drag to pan.`;
}
function style() {
  document.querySelectorAll('[data-vector]').forEach(p => {
    p.setAttribute('fill-opacity', el('opacity').value / 100);
    p.setAttribute('stroke', el('outlines').checked ? '#18303b' : 'none');
    p.setAttribute('stroke-width', '.7');
  });
}
function fit() {
  const c = shown[idx];
  box = [0, 0, c.native_shape_yx[1], c.native_shape_yx[0]];
  el('geo').style.aspectRatio = box[2] + '/' + box[3];
  apply();
}
function apply() { if (box) el('geo').setAttribute('viewBox', box.join(' ')); }
function zoom(f) {
  if (!box) return;
  const w = box[2] * f, h = box[3] * f;
  box = [box[0] + (box[2] - w) / 2, box[1] + (box[3] - h) / 2, w, h]; apply();
}
el('geo').onpointerdown = e => {
  if (!box) return;
  drag = { x: e.clientX, y: e.clientY, box: box.slice() };
  el('geo').setPointerCapture(e.pointerId);
};
el('geo').onpointermove = e => {
  if (!drag) return;
  const r = el('geo').getBoundingClientRect();
  box = [drag.box[0] - (e.clientX - drag.x) * drag.box[2] / r.width, drag.box[1] - (e.clientY - drag.y) * drag.box[3] / r.height, drag.box[2], drag.box[3]]; apply();
};
el('geo').onpointerup = e => { drag = null; if (el('geo').hasPointerCapture(e.pointerId)) el('geo').releasePointerCapture(e.pointerId); };
el('geo').onpointercancel = () => { drag = null; };
init().catch(fail);
