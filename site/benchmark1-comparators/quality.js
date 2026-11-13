"use strict";

const qualityMethods=["supervised_bora","unet","deeplabv3plus"];
const qualityRepositories=["bcss","gcss","ignite","tiger","peso"];
const qualityFmt=x=>x==null?"—":Number(x).toFixed(3);
const qualityCount=x=>Number(x).toLocaleString("en-US");
const qualityHd=x=>x==null?"—":Number(x).toFixed(1);

function qualityRow(tableId,cells){
  const tr=document.createElement("tr");
  cells.forEach((value,i)=>{
    const cell=document.createElement(i===0?"th":"td");
    if(i===0)cell.scope="row";
    cell.textContent=value;
    tr.append(cell);
  });
  document.querySelector(`#${tableId} tbody`).append(tr);
}

function renderQuality(data){
  if(data.status!=="MATCHED_BENCHMARK1_QUALITY_V1"||
     data.cohort_counts.expert_reference!==68||
     data.cohort_counts.peso_weak_reference!==19||
     data.cases.length!==87*3*2)throw Error("Incomplete matched quality data");
  const native=data.cohorts.native;
  for(const [key,label] of [["expert_68","Expert tissue (68)"],["weak_peso_19","PEsO weak reference (19)"]]){
    for(const method of qualityMethods){
      const x=native[key][method];
      qualityRow("quality-cohorts",[label,data.methods[method],
        `${x.completed_cases}/${x.selected_cases}`,qualityCount(x.annotated_pixels),
        qualityFmt(x.union_present_iou.mean),qualityFmt(x.boundary_f1_5um.mean),
        qualityFmt(x.region_covering.mean),qualityHd(x.hd95_um.mean)]);
    }
  }
  for(const repo of qualityRepositories){
    for(const method of qualityMethods){
      const x=data.repositories.native[repo][method];
      qualityRow("quality-repositories",[repo.toUpperCase(),data.methods[method],
        `${x.completed_cases}/${x.selected_cases}`,qualityCount(x.annotated_pixels),
        qualityFmt(x.union_present_iou.mean),qualityFmt(x.boundary_f1_5um.mean),
        qualityFmt(x.region_covering.mean)]);
    }
  }
  const classSelect=document.getElementById("quality-class-repository");
  function renderClasses(){
    const body=document.querySelector("#quality-classes tbody");body.replaceChildren();
    const scope=classSelect.value;
    const source=scope==="expert_68"?data.cohorts.native.expert_68:data.repositories.native[scope];
    for(const method of qualityMethods){
      for(const c of source[method].per_class){
        qualityRow("quality-classes",[data.methods[method],c.class_name,
          qualityCount(c.reference_cases),qualityCount(c.reference_pixels),
          qualityFmt(c.precision),qualityFmt(c.recall),qualityFmt(c.dice),qualityFmt(c.iou)]);
      }
    }
  }
  classSelect.addEventListener("change",renderClasses);renderClasses();
  const caseSelect=document.getElementById("quality-case-repository");
  const cases=data.cases.filter(x=>x.condition==="native");
  function renderCases(){
    document.querySelector("#quality-cases tbody").replaceChildren();
    for(const x of cases){
      if(caseSelect.value&&x.dataset!==caseSelect.value)continue;
      qualityRow("quality-cases",[x.case_id,x.dataset.toUpperCase(),x.method_name,
        qualityCount(x.annotated_pixels),qualityFmt(x.union_present_iou),
        qualityFmt(x.boundary_f1_5um),qualityFmt(x.region_covering),qualityHd(x.hd95_um)]);
    }
  }
  caseSelect.addEventListener("change",renderCases);renderCases();
}

fetch("quality-v1.json?v=matched-v1")
  .then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()})
  .then(renderQuality)
  .catch(error=>{
    const p=document.createElement("p");p.className="error";
    p.textContent=`Matched quality tables unavailable: ${error.message}`;
    document.getElementById("validation-performance").after(p);
  });
