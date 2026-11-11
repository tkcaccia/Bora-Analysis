"use strict";
const dataset=document.getElementById("dataset"),casePicker=document.getElementById("case"),title=document.getElementById("case-title");
const image=document.getElementById("case-image"),full=document.getElementById("full-image"),position=document.getElementById("position");
const names={1:"Tumor",2:"In situ tumor",3:"Stroma",4:"Inflamed stroma",5:"Necrosis",6:"Normal glandular epithelium",7:"Epithelial tissue",8:"Blood",9:"Blood vessel",10:"Lymphatic vessel",11:"Adipose",12:"Lymphocytic infiltrate",13:"Other immune infiltrate",14:"Plasma cells",15:"Metaplasia",16:"Alveolar tissue",17:"Bronchial epithelium",18:"Cartilage or bone",19:"Inflammatory infiltrate",20:"Liver tissue",21:"Macrophages",22:"Mucus, plasma or fluids",23:"Muscle",24:"Keratinized tissue"};
let all=[],visible=[];
const percent=x=>x==null?"Not defined":(100*x).toFixed(1)+"%";
function card(label,value,detail){const a=document.createElement("article"),b=document.createElement("strong"),c=document.createElement("div"),d=document.createElement("div");b.textContent=label;c.className="value";c.textContent=percent(value);d.className="small";d.textContent=detail;a.append(b,c,d);return a}
function show(id){
  const row=visible.find(x=>x.id===id)||visible[0];if(!row)return;
  casePicker.value=row.id;history.replaceState(null,"","#"+row.id);position.textContent=`${visible.indexOf(row)+1} of ${visible.length}`;
  title.textContent=`${row.dataset.toUpperCase()} · ${row.source_group} · ${row.id}`;
  image.src=row.image;full.href=row.image;image.alt=`Aligned H&E, expert, KODAMA–Bora, supervised Bora, U-Net, DeepLabV3+, VISTA-PATH and MedSAM for ${row.id}`;
  const roles=row.expert_present_roles.join(", ")||"No mapped expert class";
  document.getElementById("bora-scores").replaceChildren(
    card("KODAMA–Bora + fastPLS",row.bora_cluster_view.identification_end_to_end_macro_iou_expert_present,"Full 1,280-channel input; image-only groups, followed by tissue naming"),
    card("Supervised Bora",row.supervised_final.expert_present_macro_iou,"Training-selected joint + PLS, dataset ontology gate, polished contour"));
  document.getElementById("external-scores").replaceChildren(
    card("Class-aware U-Net",row.unet_score,roles),card("DeepLabV3+",row.deeplab_score,roles),
    card("VISTA-PATH",row.vista_score,row.vista_matched_roles.join(", ")||"No matched class"),
    card("MedSAM",row.medsam_score,row.medsam_reference_roles.join(", ")||"No eligible class"));
  const p=row.bora_pipeline,m=row.bora_cluster_view.metrics,box=document.getElementById("kodama-details");box.replaceChildren();
  const intro=document.createElement("p");intro.textContent=`${p.eligible_tiles} eligible tiles formed ${p.clusters} anonymous ${p.clusters===1?"group":"groups"}. Matched segmentation IoU on covered expert pixels: ${percent(m.segmentation_best_one_to_one_macro_iou_covered)}. Expert-annotated tissue reached: ${percent(m.annotated_coverage)}. Group-majority fastPLS name agreement: ${percent(m.identification_group_majority_accuracy)}.`;box.append(intro);
  if(p.kodama_plot){const plot=document.createElement("img");plot.src=p.kodama_plot;plot.alt=`KODAMA plot of ${p.clusters} anonymous groups`;box.append(plot)}
  const list=document.createElement("p");list.className="small";list.textContent=Object.entries(m.group_details).map(([id,d])=>`Group ${id}: ${names[d.predicted_fastpls_class_id]||"unmapped prediction"}; expert majority ${names[d.expert_majority_class_id]||"unmapped"}`).join(" · ");box.append(list);
  const grid=document.getElementById("response-grid");grid.replaceChildren(...Array.from({length:24},(_,i)=>{const n=i+1,path=`bora-supported-bw-v1/${row.id}/class-${String(n).padStart(2,"0")}.webp`,fig=document.createElement("figure"),link=document.createElement("a"),img=document.createElement("img"),caption=document.createElement("figcaption");link.href=path;img.src=path;img.loading="lazy";img.alt=`Black-and-white response for ${names[n]}`;caption.textContent=`${n} · ${names[n]}`;link.append(img);fig.append(link,caption);return fig}));
}
function filter(){visible=all.filter(r=>!dataset.value||r.dataset===dataset.value);casePicker.replaceChildren(...visible.map(r=>new Option(`${r.dataset.toUpperCase()} · ${r.source_group} · ${r.id}`,r.id)));show(location.hash.slice(1))}
fetch("cases.json?v=ontology-gate-v1").then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(data=>{
  if(data.case_count!==87||data.cases.length!==87||data.supervised_virchow2_score?.development_evaluation_only!==true)throw Error("Incomplete audited gallery");
  const multiclass=data.cases.filter(r=>r.dataset!=="peso"),average=(rows,key)=>rows.reduce((sum,r)=>sum+key(r),0)/rows.length;
  if(multiclass.length!==68)throw Error("Unexpected multiclass case count");
  document.getElementById("cohort-scores").replaceChildren(
    card("Supervised Bora · 68 multiclass images",average(multiclass,r=>r.supervised_final.expert_present_macro_iou),"Mean case macro IoU across BCSS, GCSS, IGNITE and TIGER"),
    card("KODAMA–Bora · 68 multiclass images",average(multiclass,r=>r.bora_cluster_view.identification_end_to_end_macro_iou_expert_present),"Same expert-present harmonized class rule"),
    card("Supervised Bora · all 87 images",data.supervised_virchow2_score.mean_case_expert_present_macro_iou,"Includes 19 PEsO images with one scorable class"));
  all=data.cases;[...new Set(all.map(r=>r.dataset))].sort().forEach(x=>dataset.add(new Option(x.toUpperCase(),x)));
  const requested=all.find(r=>r.id===location.hash.slice(1));if(requested)dataset.value=requested.dataset;filter();
  dataset.addEventListener("change",filter);casePicker.addEventListener("change",()=>show(casePicker.value));
  document.getElementById("previous").addEventListener("click",()=>show(visible[Math.max(0,visible.findIndex(r=>r.id===casePicker.value)-1)].id));
  document.getElementById("next").addEventListener("click",()=>show(visible[Math.min(visible.length-1,visible.findIndex(r=>r.id===casePicker.value)+1)].id));
  window.addEventListener("hashchange",()=>{const row=all.find(r=>r.id===location.hash.slice(1));if(row){dataset.value=row.dataset;filter()}});
}).catch(error=>{title.className="error";title.textContent=`Gallery unavailable: ${error.message}`});
