"use strict";
const datasetSelect=document.getElementById("dataset"),caseSelect=document.getElementById("case");
const position=document.getElementById("position"),title=document.getElementById("case-title");
const image=document.getElementById("case-image"),fullImage=document.getElementById("full-image");
const scores=document.getElementById("scores");
const pipeline=document.getElementById("pipeline-summary"),kodamaFigure=document.getElementById("kodama-figure");
const kodamaImage=document.getElementById("kodama-image"),kodamaCaption=document.getElementById("kodama-caption");
const kodamaUnavailable=document.getElementById("kodama-unavailable");
let all=[],visible=[];
const pct=x=>x===null||x===undefined?"Not defined":(100*x).toFixed(1)+"%";
function card(name,score,roles,detail){return `<article><strong>${name}</strong><div>${pct(score)}</div><div class="small">${roles}</div><div class="small">${detail}</div></article>`}
function show(id){
  const row=visible.find(x=>x.id===id)||visible[0];if(!row)return;
  caseSelect.value=row.id;history.replaceState(null,"","#"+row.id);
  position.textContent=`${visible.indexOf(row)+1} of ${visible.length}`;
  title.textContent=`${row.dataset.toUpperCase()} · ${row.source_group} · ${row.id}`;
  image.src=row.image;image.alt=`Seven aligned H&E and model output panels for ${row.dataset} ${row.source_group}`;
  fullImage.href=row.image;
  const roles=row.expert_present_roles.join(", ")||"No mapped expert role";
  scores.innerHTML=card("Bora",row.bora_same_unet_roles,roles,"Same expert-present roles as U-Net")+
    card("Class-aware U-Net",row.unet_score,roles,"Same expert-present roles as Bora and DeepLabV3+")+
    card("DeepLabV3+",row.deeplab_score,roles,"Same expert-present roles as Bora and U-Net. Inference: "+row.deeplab_inference_seconds.toFixed(1)+" s")+
    card("VISTA-PATH",row.vista_score,row.vista_matched_roles.join(", ")||"No matched role","Bora on these roles: "+pct(row.bora_same_vista_roles))+
    card("MedSAM",row.medsam_score,row.medsam_reference_roles.join(", ")||"No eligible role",
      "Automatic boxes: tumor "+row.medsam_box_counts.tumor+", stroma "+row.medsam_box_counts.stroma+", necrosis "+row.medsam_box_counts.necrosis+". Bora on these roles: "+pct(row.bora_same_medsam_roles));
  const p=row.bora_pipeline;
  const counts=p.tile_counts_by_group.map((count,index)=>`Group ${index+1}: ${count}`).join(" · ");
  pipeline.innerHTML=`<article><strong>Eligible tissue tiles</strong>${p.eligible_tiles}<div class="small">Strictly less than 10% background, including padding</div></article>`+
    `<article><strong>Anonymous clusters</strong>${p.clusters}<div class="small">${p.grouping_route==="small_roi_coverage"?"Small-image coverage route":"KODAMA + consensus/projection graph"}</div></article>`+
    `<article><strong>Tiles per cluster</strong>${counts}<div class="small">All eligible tiles accounted for</div></article>`;
  kodamaFigure.hidden=!p.kodama_plot;
  kodamaUnavailable.hidden=!!p.kodama_plot;
  if(p.kodama_plot){
    kodamaImage.src=p.kodama_plot;
    kodamaImage.alt=`KODAMA coordinates for ${p.eligible_tiles} eligible tiles in ${p.clusters} anonymous groups`;
    kodamaCaption.textContent=`Saved image-only KODAMA fit · ${p.eligible_tiles} tiles · ${p.clusters} groups. A 2D separation does not prove different tissue identity.`;
  }else{
    kodamaImage.removeAttribute("src");
    kodamaUnavailable.textContent=`KODAMA plot omitted: ${p.eligible_tiles} eligible tiles, below the 30-tile display threshold. ${p.grouping_route==="small_roi_coverage"?"This image used the small-image coverage route instead of KODAMA.":"The saved clustering remains included in the counts above."}`;
  }
}
function filter(){
  visible=all.filter(x=>!datasetSelect.value||x.dataset===datasetSelect.value);
  caseSelect.replaceChildren(...visible.map(x=>new Option(`${x.dataset.toUpperCase()} · ${x.source_group} · ${x.id}`,x.id)));
  show(location.hash.slice(1));
}
fetch("cases.json").then(r=>{if(!r.ok)throw Error(`Gallery manifest: HTTP ${r.status}`);return r.json()}).then(data=>{
  if(data.status!=="AUDITED_PUBLIC_BENCHMARK1_COMPARATOR_VISUALS_WITH_BORA_PIPELINE"||data.case_count!==106||data.cases.length!==106)throw Error("Gallery audit is incomplete");
  all=data.cases;
  [...new Set(all.map(x=>x.dataset))].sort().forEach(x=>datasetSelect.add(new Option(x.toUpperCase(),x)));
  const requested=all.find(x=>x.id===location.hash.slice(1));
  if(requested)datasetSelect.value=requested.dataset;
  filter();
  datasetSelect.addEventListener("change",filter);
  caseSelect.addEventListener("change",()=>show(caseSelect.value));
  document.getElementById("previous").addEventListener("click",()=>show(visible[Math.max(0,visible.findIndex(x=>x.id===caseSelect.value)-1)].id));
  document.getElementById("next").addEventListener("click",()=>show(visible[Math.min(visible.length-1,visible.findIndex(x=>x.id===caseSelect.value)+1)].id));
  window.addEventListener("hashchange",()=>{const row=all.find(x=>x.id===location.hash.slice(1));if(row){datasetSelect.value=row.dataset;filter();}});
}).catch(e=>{title.className="error";title.textContent="The comparison gallery could not be loaded: "+e.message});
