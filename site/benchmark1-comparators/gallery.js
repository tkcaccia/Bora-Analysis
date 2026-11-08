"use strict";
const datasetSelect=document.getElementById("dataset"),caseSelect=document.getElementById("case");
const position=document.getElementById("position"),title=document.getElementById("case-title");
const image=document.getElementById("case-image"),fullImage=document.getElementById("full-image");
const scores=document.getElementById("scores");
let all=[],visible=[];
const pct=x=>x===null||x===undefined?"Not defined":(100*x).toFixed(1)+"%";
function card(name,score,roles,detail){return `<article><strong>${name}</strong><div>${pct(score)}</div><div class="small">${roles}</div><div class="small">${detail}</div></article>`}
function show(id){
  const row=visible.find(x=>x.id===id)||visible[0];if(!row)return;
  caseSelect.value=row.id;history.replaceState(null,"","#"+row.id);
  position.textContent=`${visible.indexOf(row)+1} of ${visible.length}`;
  title.textContent=`${row.dataset.toUpperCase()} · ${row.source_group} · ${row.id}`;
  image.src=row.image;image.alt=`Six aligned H&E and model output panels for ${row.dataset} ${row.source_group}`;
  fullImage.href=row.image;
  const roles=row.expert_present_roles.join(", ")||"No mapped expert role";
  scores.innerHTML=card("Bora",row.bora_same_unet_roles,roles,"Same expert-present roles as U-Net")+
    card("Class-aware U-Net",row.unet_score,roles,"Same expert-present roles as Bora")+
    card("VISTA-PATH",row.vista_score,row.vista_matched_roles.join(", ")||"No matched role","Bora on these roles: "+pct(row.bora_same_vista_roles))+
    card("MedSAM",row.medsam_score,row.medsam_reference_roles.join(", ")||"No eligible role",
      "Automatic boxes: tumor "+row.medsam_box_counts.tumor+", stroma "+row.medsam_box_counts.stroma+", necrosis "+row.medsam_box_counts.necrosis+". Bora on these roles: "+pct(row.bora_same_medsam_roles));
}
function filter(){
  visible=all.filter(x=>!datasetSelect.value||x.dataset===datasetSelect.value);
  caseSelect.replaceChildren(...visible.map(x=>new Option(`${x.dataset.toUpperCase()} · ${x.source_group} · ${x.id}`,x.id)));
  show(location.hash.slice(1));
}
fetch("cases.json").then(r=>{if(!r.ok)throw Error(`Gallery manifest: HTTP ${r.status}`);return r.json()}).then(data=>{
  if(data.status!=="AUDITED_PUBLIC_BENCHMARK1_COMPARATOR_VISUALS"||data.case_count!==106||data.cases.length!==106)throw Error("Gallery audit is incomplete");
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
