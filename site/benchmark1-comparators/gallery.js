"use strict";
const datasetSelect=document.getElementById("dataset"),caseSelect=document.getElementById("case");
const position=document.getElementById("position"),title=document.getElementById("case-title");
const image=document.getElementById("case-image"),fullImage=document.getElementById("full-image");
const scores=document.getElementById("scores");
const pipeline=document.getElementById("pipeline-summary"),kodamaFigure=document.getElementById("kodama-figure");
const kodamaImage=document.getElementById("kodama-image"),kodamaCaption=document.getElementById("kodama-caption");
const kodamaUnavailable=document.getElementById("kodama-unavailable");
const clusterImage=document.getElementById("cluster-image"),clusterFullImage=document.getElementById("cluster-full-image");
const rgbImage=document.getElementById("rgb-image"),rgbFullImage=document.getElementById("rgb-full-image");
const predictionImage=document.getElementById("prediction-image"),predictionFullImage=document.getElementById("prediction-full-image");
const clusterCaption=document.getElementById("cluster-caption"),clusterLegend=document.getElementById("cluster-legend");
const predictionLegend=document.getElementById("prediction-legend");
const segmentationScores=document.getElementById("segmentation-scores"),identificationScores=document.getElementById("identification-scores");
const groupTable=document.querySelector("#group-identification tbody");
const classNames={1:"Tumor",2:"In situ tumor",3:"Stroma",4:"Inflamed stroma",5:"Necrosis",6:"Normal glandular epithelium",7:"Epithelial tissue",8:"Blood",9:"Blood vessel",10:"Lymphatic vessel",11:"Adipose",12:"Lymphocytic infiltrate",13:"Other immune infiltrate",14:"Plasma cells",15:"Metaplasia",16:"Alveolar tissue",17:"Bronchial epithelium",18:"Cartilage or bone",19:"Inflammatory infiltrate",20:"Liver tissue",21:"Macrophages",22:"Mucus, plasma or fluids",23:"Muscle",24:"Keratinized tissue"};
let all=[],visible=[];
const pct=x=>x===null||x===undefined?"Not defined":(100*x).toFixed(1)+"%";
const decimal=x=>x===null||x===undefined?"Not defined":Number(x).toFixed(3);
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
  const stage=row.bora_cluster_view;
  const m=stage.metrics;
  clusterImage.src=stage.image;
  clusterFullImage.href=stage.image;
  rgbImage.src=stage.rgb_image;
  rgbFullImage.href=stage.rgb_image;
  rgbImage.alt=`Standalone RGB Bora mask for ${p.clusters} anonymous groups on a white background`;
  predictionImage.src=stage.fastpls_rgb_image;
  predictionFullImage.href=stage.fastpls_rgb_image;
  predictionImage.alt=`Standalone RGB Bora mask colored by fastPLS tissue predictions for ${p.clusters} groups`;
  clusterImage.alt=`Original H&E with the final Bora geometry colored by ${p.clusters} anonymous clusters`;
  clusterCaption.textContent=`Same ${p.clusters} anonymous Bora clusters over H&E; geometry matches both RGB maps`;
  clusterLegend.innerHTML=stage.palette_hex.map((color,index)=>`<span><i style="background:${color}"></i>Group ${index+1} · ${p.tile_counts_by_group[index]} eligible tiles</span>`).join("");
  predictionLegend.innerHTML=Object.entries(stage.fastpls_palette_hex).map(([cls,color])=>`<span><i style="background:${color}"></i>${classNames[cls]}</span>`).join("");
  segmentationScores.innerHTML=`<article><strong>Adjusted Rand index</strong>${decimal(m.segmentation_adjusted_rand_covered)}<div class="small">Anonymous groups versus expert tissue regions; 1 is exact agreement</div></article>`+
    `<article><strong>Optimally matched IoU</strong>${pct(m.segmentation_best_one_to_one_macro_iou_covered)}<div class="small">Best one-to-one group matching on covered expert pixels</div></article>`+
    `<article><strong>Annotated tissue reached</strong>${pct(m.annotated_coverage)}<div class="small">${m.expert_annotated_pixels_with_bora_group.toLocaleString()} of ${m.expert_annotated_pixels.toLocaleString()} annotated pixels</div></article>`;
  identificationScores.innerHTML=`<article><strong>fastPLS group names</strong>${pct(m.identification_group_majority_accuracy)}<div class="small">${m.identification_majority_correct_groups} of ${m.identification_scorable_groups} groups match their expert-majority tissue name; area-weighted: ${pct(m.identification_majority_pixel_weighted_accuracy)}</div></article>`+
    `<article><strong>End-to-end tissue macro IoU</strong>${pct(stage.identification_end_to_end_macro_iou_expert_present)}<div class="small">Original published expert-present class score, including segmentation and naming</div></article>`;
  groupTable.replaceChildren(...Object.entries(m.group_details).map(([id,detail])=>{
    const cells=[`Group ${id}`,`${p.tile_counts_by_group[Number(id)-1]}`,classNames[detail.predicted_fastpls_class_id],classNames[detail.expert_majority_class_id],detail.correct_name?"Matches":"Differs"];
    const tr=document.createElement("tr");
    cells.forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td)});
    return tr;
  }));
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
  showSelected(row);
  showResponses(row);
}
function showResponses(row){
  const grid=document.getElementById("response-grid");
  const items=[];
  for(let cls=1;cls<=24;cls++){
    const path=`bw-responses-v2/${row.id}/class-${String(cls).padStart(2,"0")}.webp`;
    const figure=document.createElement("figure");
    const link=document.createElement("a");link.href=path;
    const map=document.createElement("img");map.src=path;map.loading="lazy";
    map.alt=`Grayscale U-Net response for ${classNames[cls]} on ${row.dataset} ${row.source_group}; black is 0, white is 1`;
    const caption=document.createElement("figcaption");caption.textContent=`${cls} · ${classNames[cls]}`;
    link.append(map);figure.append(link,caption);items.push(figure);
  }
  grid.replaceChildren(...items);
}
function showSelected(row){
  const v=row.bora_training_selected_view,m=v.metrics;
  const classic=row.bora_cluster_view,cp=row.bora_pipeline;
  const routeTable=document.querySelector("#route-comparison tbody");
  routeTable.replaceChildren(...[
    ["Classic",1280,cp.eligible_tiles,cp.clusters,classic.metrics.segmentation_best_one_to_one_macro_iou_covered,classic.metrics.identification_group_majority_accuracy,classic.identification_end_to_end_macro_iou_expert_present],
    ["Training-selected",256,v.eligible_tiles,v.clusters,m.segmentation_best_one_to_one_macro_iou_covered,m.identification_group_majority_accuracy,v.end_to_end_macro_iou]
  ].map(values=>{const tr=document.createElement("tr");values.forEach((value,i)=>{const td=document.createElement("td");td.textContent=i>3?pct(value):value;tr.append(td)});return tr;}));
  const assets=[["selected-rgb",v.images.cluster_rgb],["selected-prediction",v.images.fastpls_rgb],["selected-overlay",v.images.overlay]];
  for(const [id,path] of assets){document.getElementById(id).src=path;document.getElementById(id+"-full").href=path;}
  document.getElementById("selected-pipeline-summary").innerHTML=
    `<article><strong>Eligible tissue tiles</strong>${v.eligible_tiles}<div class="small">Same strict tile admission rule as classic Bora</div></article>`+
    `<article><strong>Anonymous clusters</strong>${v.clusters}<div class="small">${v.grouping_route==="small_roi_coverage"?"Small-image coverage route":"KODAMA + consensus/projection graph"}</div></article>`+
    `<article><strong>Tiles per cluster</strong>${v.tile_counts_by_group.map((n,i)=>`Group ${i+1}: ${n}`).join(" · ")}</article>`;
  document.getElementById("selected-cluster-legend").innerHTML=v.cluster_palette_hex.map((color,i)=>`<span><i style="background:${color}"></i>Group ${i+1} · ${v.tile_counts_by_group[i]} eligible tiles</span>`).join("");
  document.getElementById("selected-prediction-legend").innerHTML=Object.entries(v.tissue_palette_hex).map(([cls,color])=>`<span><i style="background:${color}"></i>${classNames[cls]}</span>`).join("");
  document.getElementById("selected-segmentation-scores").innerHTML=
    `<article><strong>Adjusted Rand index</strong>${decimal(m.segmentation_adjusted_rand_covered)}<div class="small">Anonymous groups versus expert regions on covered pixels</div></article>`+
    `<article><strong>Optimally matched IoU</strong>${pct(m.segmentation_best_one_to_one_macro_iou_covered)}<div class="small">One-to-one group matching on covered expert pixels</div></article>`+
    `<article><strong>Annotated tissue reached</strong>${pct(m.annotated_coverage)}<div class="small">${m.expert_annotated_pixels_with_bora_group.toLocaleString()} of ${m.expert_annotated_pixels.toLocaleString()} pixels</div></article>`;
  document.getElementById("selected-identification-scores").innerHTML=
    `<article><strong>fastPLS group names</strong>${pct(m.identification_group_majority_accuracy)}<div class="small">${m.identification_majority_correct_groups} of ${m.identification_scorable_groups} groups match their expert-majority name</div></article>`+
    `<article><strong>End-to-end tissue macro IoU</strong>${pct(v.end_to_end_macro_iou)}<div class="small">Full expert-present class rule, including missed tissue</div></article>`;
  const table=document.querySelector("#selected-group-identification tbody");
  table.replaceChildren(...Object.entries(m.group_details).map(([id,d])=>{
    const tr=document.createElement("tr");
    [`Group ${id}`,v.tile_counts_by_group[Number(id)-1],classNames[d.predicted_fastpls_class_id],classNames[d.expert_majority_class_id],d.correct_name?"Matches":"Differs"].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td)});
    return tr;
  }));
  const figure=document.getElementById("selected-kodama-figure"),unavailable=document.getElementById("selected-kodama-unavailable");
  figure.hidden=!v.images.kodama;unavailable.hidden=!!v.images.kodama;
  if(v.images.kodama){document.getElementById("selected-kodama-image").src=v.images.kodama;
    document.getElementById("selected-kodama-caption").textContent=`Training-selected KODAMA · ${v.eligible_tiles} tiles · ${v.clusters} groups. A 2D separation does not establish tissue identity.`;
  }else{document.getElementById("selected-kodama-image").removeAttribute("src");
    unavailable.textContent=`No KODAMA plot for this ${v.eligible_tiles}-tile small-image route.`;}
}
function filter(){
  visible=all.filter(x=>!datasetSelect.value||x.dataset===datasetSelect.value);
  caseSelect.replaceChildren(...visible.map(x=>new Option(`${x.dataset.toUpperCase()} · ${x.source_group} · ${x.id}`,x.id)));
  show(location.hash.slice(1));
}
fetch("cases.json?v=training-selected-v6").then(r=>{if(!r.ok)throw Error(`Gallery manifest: HTTP ${r.status}`);return r.json()}).then(data=>{
  if(data.status!=="AUDITED_BENCHMARK1_FULL_VS_TRAINING_SELECTED_DEVELOPMENT_ONLY"||data.case_count!==87||data.cases.length!==87||data.cases.some(x=>!x.bora_training_selected_view))throw Error("Gallery audit is incomplete");
  document.getElementById("selected-summary").textContent=`Across 87 development images, training-selected Bora had ${data.training_selected_summary.one_group_cases} one-group outputs. Mean covered-pixel matched segmentation IoU: ${pct(data.training_selected_summary.mean_matched_segmentation_iou_covered)}; mean group-majority name agreement: ${pct(data.training_selected_summary.mean_group_name_accuracy)}; mean end-to-end tissue macro IoU: ${pct(data.training_selected_summary.mean_end_to_end_macro_iou)}. Compare these stage-specific values with classic Bora above; neither route was evaluated on an untouched test cohort.`;
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
