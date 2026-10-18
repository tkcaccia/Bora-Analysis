"""One reviewable example using the previous tile mask with anonymous IDs.

No clustering or new RGB conversion. A whole anonymous group, including all
its disconnected pieces, receives one tissue ranking after Bora refinement.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tifffile
import cv2
from threadpoolctl import threadpool_limits
from bora.appearance import refine_appearance
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.predict import load_model
from experiments.bora_supervised.viewer import palette,overlay
from .geometry import token_evidence
from .labels import semantic_mask


def anonymize(coarse,image_sha256):
    ids=np.unique(coarse[coarse>0]).astype(int)
    seed=int(hashlib.sha256(('bora-anonymous-coarse-example-v1:'+image_sha256).encode()).hexdigest()[:16],16)
    source_order=np.random.default_rng(seed).permutation(ids)
    lookup=np.zeros(int(coarse.max())+1,np.uint8)
    for anonymous,source in enumerate(source_order,1):lookup[source]=anonymous
    anonymous=lookup[coarse]
    restore=np.r_[0,source_order].astype(np.uint8)
    if not np.array_equal(restore[anonymous],coarse):raise ValueError('Anonymization moved or changed a tile assignment')
    return anonymous,restore


def run(registry_root,case,model,output):
    registry_root,model,output=map(Path,(registry_root,model,output))
    if output.exists():raise ValueError('Preserve the previous example')
    registry=load(registry_root/'REGISTRY.json');row=next(r for r in registry['cases'] if r['case']==case)
    src=Path(row['original_path']).parent
    if sha(src/'coarse.tif')!=row['sample']['coarse_sha256']:raise ValueError('Previous benchmark tile mask changed')
    if sha(row['original_path'])!=row['original_sha256'] or sha(row['feature']['path'])!=row['feature']['sha256']:
        raise ValueError('Image or full feature tensor changed')
    output.mkdir(parents=True)
    he=np.asarray(Image.open(row['original_path']).convert('RGB'));coarse=tifffile.imread(src/'coarse.tif')
    anonymous,restore=anonymize(coarse,row['original_sha256'])
    rgb=np.asarray(Image.open(src/'ensemble/representation.png').convert('RGB'))
    support=tifffile.imread(row['tissue_support_path'])>0;native_support=tifffile.imread(src/'tissue.native.tif')>0
    began=time.monotonic()
    refined,details=refine_appearance(rgb,anonymous,support,row['sample']['working_mpp_xy'],he=he,method='af',
        output_shape_yx=row['sample']['native_shape_yx'],output_tissue=native_support)
    bora_seconds=time.monotonic()-began
    source_refined,_=refine_appearance(rgb,coarse,support,row['sample']['working_mpp_xy'],he=he,method='af',
        output_shape_yx=row['sample']['native_shape_yx'],output_tissue=native_support)
    differing=int(np.count_nonzero(restore[refined]!=source_refined))
    if differing:raise ValueError('Anonymized Bora changed geometry relative to the original input')
    # The head receives only pooled full Virchow2 features, never A/B IDs,
    # input colors, restored source IDs, class names or the fine reference.
    tokens=np.load(row['feature']['path'],mmap_mode='r');mean,_,mass,_=token_evidence(tokens,refined,he.shape[:2])
    predictor,metadata,startup=load_model(model)
    if metadata['feature']!='mean':raise ValueError('This example uses the frozen mean-feature head')
    began=time.monotonic();ranks=predictor(np.asarray(mean,dtype=float));head_seconds=time.monotonic()-began
    identified=semantic_mask(refined,ranks)
    tifffile.imwrite(output/'coarse.anonymous.tif',anonymous,compression='deflate')
    tifffile.imwrite(output/'bora.anonymous.native.tif',refined,compression='deflate')
    tifffile.imwrite(output/'identified.native.tif',identified,compression='deflate')
    np.savez_compressed(output/'ranked_groups.npz',top5=ranks)
    shutil.copyfile(src/'ensemble/representation.png',output/'unchanged-converted.png')
    save(output/'PREDICTIONS_FROZEN.json',dict(prediction_unit='One whole anonymous Bora group, including all disconnected pieces',
        source_tile_mask_sha256=sha(src/'coarse.tif'),anonymized_tile_mask_sha256=sha(output/'coarse.anonymous.tif'),
        tile_assignments_preserved_exactly=True,number_of_input_groups=len(restore)-1,new_clustering_used=False,
        source_rgb_sha256=sha(src/'ensemble/representation.png'),copied_rgb_sha256=sha(output/'unchanged-converted.png'),
        rgb_recomputed=False,bora_original_vs_anonymous_differing_pixels=differing,
        predictions_sha256=sha(output/'ranked_groups.npz'),semantic_mask_sha256=sha(output/'identified.native.tif'),
        model=metadata['name'],model_backend=metadata['engine'],classifier_inputs='Full Virchow2 means inside whole refined groups only',
        fine_reference_opened_before_predictions=False,source_identity_names_given_to_head=False,
        example_only=True,provisional_head='Existing fragment-trained R02-c48 used to demonstrate the corrected flow; not a retrained or validated whole-group model.',
        bora_seconds=bora_seconds,head_seconds=head_seconds,startup_refit_seconds=startup,bora_details=details))
    # Names, the reversible source key and the expert reference are revealed
    # only for the example display and audit after the predictions freeze.
    source_to_role=row['ontology_raw_to_role'];roles=registry['ontology']['roles'];groups=[]
    for g,rank in enumerate(ranks,1):
        source=int(restore[g]);expected=source_to_role.get(str(source),0)
        groups.append(dict(anonymous_id=g,anonymous_name=chr(64+g),native_pixels=int(mass[g-1]),
            disconnected_pieces=int(ndi.label(refined==g,np.ones((3,3)))[1]),
            predicted=roles[int(rank[0])-1],top5=[roles[int(j)-1] for j in rank],
            source_identity_revealed_after_prediction=row['sample']['class_names'][str(source)],
            source_role=roles[expected-1] if expected else 'unmapped',correct_identity=int(rank[0])==expected))
    colours=palette(len(roles))
    for code,role in source_to_role.items():
        if role:colours[role]=row['sample']['palette'][int(code)-1]
    anonymous_colours=np.array([[0,0,0],[65,125,235],[235,170,55]],np.uint8) if len(groups)==2 else palette(len(groups))
    named=np.zeros(coarse.shape,np.uint8)
    for code,role in source_to_role.items():named[coarse==int(code)]=role
    expert_raw=tifffile.imread(row['reference_path']);expert=np.zeros(expert_raw.shape,np.uint8)
    for code,role in source_to_role.items():expert[expert_raw==int(code)]=role
    image=Image.fromarray(he);image.thumbnail((900,850));im=np.asarray(image)
    image.save(output/'original.jpg',quality=92)
    for filename,mask,pal in [('previous-tile',named,colours),('anonymous-tile',anonymous,anonymous_colours),
                              ('bora-anonymous',refined,anonymous_colours),('identified',identified,colours),('expert',expert,colours)]:
        small=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST))
        Image.fromarray(overlay(im,small,pal)).save(output/(filename+'.jpg'),quality=94)
    converted=Image.open(output/'unchanged-converted.png');converted.thumbnail((900,850));converted.save(output/'converted.jpg',quality=94)
    save(output/'data.json',dict(case=case,groups=groups,model=metadata['name'],roles=roles,colours=colours.tolist(),
        source_tile_mask_sha256=row['sample']['coarse_sha256'],tile_partition_identical=True,rgb_unchanged=True,
        anonymous_refinement_matches_previous=True,mpp_xy=row['sample']['working_mpp_xy'],
        annotation_role='Expert reference is displayed only after predictions freeze. Previous benchmark tile input remains reference-derived.'))
    (output/'index.html').write_text(HTML)
    print(__import__('json').dumps(dict(case=case,groups=groups,tile_partition_identical=True,rgb_unchanged=True,anonymous_refinement_matches_previous=True)),flush=True)


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><title>One example · anonymize the previous Bora tile mask</title>
<style>body{font:16px system-ui;margin:20px;background:#f3f6f7;color:#132d3c}h1{font-size:25px;margin-bottom:8px}p{line-height:1.45;max-width:1200px}.flow{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:12px;overflow-x:auto}figure{margin:0;padding:9px;background:white}img{width:100%}figcaption{font-weight:650;min-height:48px}.context{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:12px}table{border-collapse:collapse;background:white;margin:18px 0}td,th{border:1px solid #cbd5db;padding:10px;text-align:left}.proof{background:#e2f0e7;padding:12px}.note{font-size:14px}a{color:#00658b}.swatch{display:inline-block;width:13px;height:13px;margin-right:5px}</style>
<h1 id="title">Previous tile mask → anonymous groups → Bora → tissue identity</h1>
<p><b>The tile partition is preserved.</b> Only the IDs and display names become A/B. Bora refines those existing groups. The classifier then gives <b>one tissue identity to each whole group</b>, shared by all its disconnected pieces.</p>
<p class="proof">Verified: zero changed input tile assignments · identical RGB conversion bytes · identical Bora geometry after reversing anonymous IDs · no new clustering.</p>
<div class="flow">
<figure><figcaption>1 Previous benchmark tile mask<br><small>Names shown for comparison</small></figcaption><a href="previous-tile.jpg"><img src="previous-tile.jpg"></a></figure>
<figure><figcaption>2 Identical input, anonymized<br><small><span style="color:#417deb">■</span> A · <span style="color:#ebaa37">■</span> B</small></figcaption><a href="anonymous-tile.jpg"><img src="anonymous-tile.jpg"></a></figure>
<figure><figcaption>3 After Bora, still anonymous<br><small>Same group identities A/B</small></figcaption><a href="bora-anonymous.jpg"><img src="bora-anonymous.jpg"></a></figure>
<figure><figcaption>4 Tissue identity predicted afterward<br><small id="prediction"></small></figcaption><a href="identified.jpg"><img src="identified.jpg"></a></figure>
</div><div id="legend"></div><table id="groups"></table>
<p class="note">This is a single example to check the workflow. It uses the existing native fastPLS CPU mean-feature classifier provisionally. That classifier was trained on the preceding fragment experiment; it has not yet been retrained or validated for whole groups. Its input here contains only pooled full Virchow2 features inside the refined group. A/B IDs, source tissue names and expert annotation are excluded from the head. Top-five entries are ranks, not probabilities.</p>
<div class="context"><figure><figcaption>Original H&E</figcaption><a href="original.jpg"><img src="original.jpg"></a></figure><figure><figcaption>Expert annotation for comparison</figcaption><a href="expert.jpg"><img src="expert.jpg"></a></figure><figure><figcaption>Previous RGB conversion, unchanged</figcaption><a href="converted.jpg"><img src="converted.jpg"></a></figure></div>
<script>fetch('data.json').then(r=>r.json()).then(d=>{document.getElementById('title').textContent=d.case+' · Previous tiles → anonymous groups → Bora → tissue identity';document.getElementById('prediction').textContent=d.groups.map(g=>g.anonymous_name+' = '+g.predicted.replaceAll('_',' ')).join(' · ');document.getElementById('groups').innerHTML='<tr><th>Anonymous group</th><th>Native pixels</th><th>Separate pieces sharing one label</th><th>Predicted tissue</th><th>Reference tissue class</th></tr>'+d.groups.map(g=>'<tr><td>'+g.anonymous_name+'</td><td>'+g.native_pixels.toLocaleString()+'</td><td>'+g.disconnected_pieces+'</td><td>'+g.predicted.replaceAll('_',' ')+'</td><td>'+g.source_identity_revealed_after_prediction+'</td></tr>').join('');let used=[...new Set(d.groups.flatMap(g=>[g.predicted,g.source_role]))];document.getElementById('legend').innerHTML=used.filter(n=>n!=='unmapped').map(n=>'<span style="margin-right:15px"><i class="swatch" style="background:rgb('+d.colours[d.roles.indexOf(n)+1]+')"></i>'+n.replaceAll('_',' ')+'</span>').join('')});</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','model','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--case',default='tiger_06');a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.registry_root,a.case,a.model,a.output)


if __name__=='__main__':main()
