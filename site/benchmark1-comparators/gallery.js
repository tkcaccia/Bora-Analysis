"use strict";
const dataset=document.getElementById("dataset"),picker=document.getElementById("case"),title=document.getElementById("case-title");
const image=document.getElementById("case-image"),full=document.getElementById("full-image"),position=document.getElementById("position");
let all=[],visible=[],names={};
const percent=x=>x==null?"Not defined":(100*x).toFixed(1)+"%";
function card(label,value,detail){const a=document.createElement("article"),b=document.createElement("strong"),c=document.createElement("div"),d=document.createElement("div");b.textContent=label;c.className="value";c.textContent=percent(value);d.className="small";d.textContent=detail;a.append(b,c,d);return a}
function show(id){
  const row=visible.find(x=>x.id===id)||visible[0];if(!row)return;
  picker.value=row.id;history.replaceState(null,"","#"+row.id);position.textContent=`${visible.indexOf(row)+1} of ${visible.length}`;
  title.textContent=`${row.dataset.toUpperCase()} · ${row.source_group} · ${row.id}`;
  image.src=row.figure;full.href=row.figure;image.alt=`Eight-class H&E, expert, supervised Bora and four external method panels for ${row.id}`;
  document.getElementById("bora-scores").replaceChildren(
    card("Supervised Bora",row.scores.supervised,"New eight-class Virchow2 head, fitted on training images"),
    card("KODAMA–Bora + fastPLS",row.scores.kodama_bora,"Frozen group route, exactly rescored; spatial mask unavailable locally"));
  document.getElementById("external-scores").replaceChildren(
    card("Class-aware U-Net",row.scores.unet,"Prior RGB prediction mapped to eight classes"),
    card("DeepLabV3+",row.scores.deeplabv3plus,"Prior RGB prediction mapped to eight classes"),
    card("VISTA-PATH",row.scores.vista_path,"Prior pretrained prediction mapped to eight classes"),
    card("MedSAM",row.scores.medsam,"Prompted tumor, stroma and necrosis only"));
  const box=document.getElementById("kodama-details");box.replaceChildren();
  const note=document.createElement("p");note.textContent=`The frozen route produced ${row.kodama_groups} anonymous groups. Its broad-label score uses the recorded full pixel confusion counts. No new spatial group mask is claimed.`;box.append(note);
  if(row.kodama_plot){const plot=document.createElement("img");plot.src=row.kodama_plot;plot.alt=`Original KODAMA embedding with ${row.kodama_groups} anonymous groups`;box.append(plot)}
  const grid=document.getElementById("response-grid");grid.replaceChildren(...Array.from({length:8},(_,i)=>{const k=i+1,path=row.bw_root+`class-${String(k).padStart(2,"0")}.webp`,fig=document.createElement("figure"),link=document.createElement("a"),img=document.createElement("img"),caption=document.createElement("figcaption");link.href=path;img.src=path;img.loading="lazy";img.alt=`Black-and-white supervised response for ${names[k]}`;caption.textContent=`${k} · ${names[k]}`;link.append(img);fig.append(link,caption);return fig}));
}
function filter(){visible=all.filter(r=>!dataset.value||r.dataset===dataset.value);picker.replaceChildren(...visible.map(r=>new Option(`${r.dataset.toUpperCase()} · ${r.source_group} · ${r.id}`,r.id)));show(location.hash.slice(1))}
fetch("cases-eight.json?v=eight-class-v1").then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(data=>{
  if(data.status!=="EIGHT_CLASS_DEVELOPMENT_GALLERY_STAGED"||data.case_count!==87||data.cases.length!==87||data.kodama_spatial_mask_available!==false)throw Error("Incomplete audited gallery");
  names=data.class_names;
  document.getElementById("cohort-scores").replaceChildren(
    card("Supervised Bora · four multiclass datasets",data.cohort_scores.supervised_four_multiclass_datasets,"Equal-weight average of dataset-level case macro IoU"),
    card("KODAMA–Bora · four multiclass datasets",data.cohort_scores.kodama_bora_four_multiclass_datasets,"Frozen result mapped to eight classes"),
    card("Supervised Bora · all 87",data.cohort_scores.supervised_all_87,"Includes 19 single-class PEsO images"));
  all=data.cases;[...new Set(all.map(r=>r.dataset))].sort().forEach(x=>dataset.add(new Option(x.toUpperCase(),x)));
  const requested=all.find(r=>r.id===location.hash.slice(1));if(requested)dataset.value=requested.dataset;filter();
  dataset.addEventListener("change",filter);picker.addEventListener("change",()=>show(picker.value));
  document.getElementById("previous").addEventListener("click",()=>show(visible[Math.max(0,visible.findIndex(r=>r.id===picker.value)-1)].id));
  document.getElementById("next").addEventListener("click",()=>show(visible[Math.min(visible.length-1,visible.findIndex(r=>r.id===picker.value)+1)].id));
  window.addEventListener("hashchange",()=>{const row=all.find(r=>r.id===location.hash.slice(1));if(row){dataset.value=row.dataset;filter()}});
}).catch(error=>{title.className="error";title.textContent=`Gallery unavailable: ${error.message}`});
