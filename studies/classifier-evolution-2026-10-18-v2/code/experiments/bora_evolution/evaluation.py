"""Score frozen predictions through actual Bora simplified GeoJSON exports."""
import hashlib,time
from pathlib import Path
import numpy as np
import tifffile
from bora.multicpu_geojson import convert
from bora.mask_cleanup import fill_small_regions,finalize_vector_regions
from bora.compact_geojson import compact
from experiments.bora_group.bora_geojson import PROFILE,annotate,render,geometry_stats
from experiments.bora_group.targets import identity_labels
from experiments.bora_segment.targets import target_counts
from experiments.bora_segment.labels import semantic_mask
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.evaluate import broad_truth,pixel_scores
from experiments.bora_supervised.train import score_units
from experiments.bora_supervised.viewer import palette


def cache_vector(study,data,case,ranks):
    study=Path(study);row=data.rows[case];geo=data.geo/'cases'/case;frozen=load(geo/'GEOMETRY_FROZEN.json')
    import bora.multicpu_geojson,bora.mask_cleanup,bora.compact_geojson
    key=hashlib.sha256((frozen['files_sha256']['regions.native.tif']+repr(ranks[:,0].tolist())+repr(PROFILE)+
         'fill20-shared-rdp-3um-class97-area1p5-v1'+sha(Path(bora.compact_geojson.__file__))+sha(Path(bora.mask_cleanup.__file__))+sha(Path(bora.multicpu_geojson.__file__))).encode()).hexdigest()
    folder=study/'vector-cache'/case/key;record=folder/'RECORD.json'
    if record.exists():
        item=load(record)
        if sha(folder/'mask.geojson')!=item['geojson_sha256'] or sha(folder/'mask.tif')!=item['raster_sha256']:raise ValueError('Cached vector changed')
        return item,folder
    folder.mkdir(parents=True,exist_ok=True);regions=tifffile.imread(geo/'regions.native.tif');mask=semantic_mask(regions,ranks)
    start=time.monotonic();cleaned,cleanup=fill_small_regions(mask,20);cleanup_seconds=time.monotonic()-start
    tifffile.imwrite(folder/'input.tif',cleaned,compression='deflate')
    report=convert(folder/'input.tif',folder/'mask.geojson',**PROFILE)
    post_start=time.monotonic()
    vector=annotate(load(folder/'mask.geojson'),data.registry['ontology']['roles'],palette(len(data.registry['ontology']['roles'])),row['sample']['native_mpp_xy'],mask.shape)
    vector['small_region_cleanup']=cleanup;vector=finalize_vector_regions(vector,mask.shape,20);vector=compact(vector,row['sample']['native_mpp_xy'],mask.shape);save(folder/'mask.geojson',vector)
    restored=render(vector,mask.shape);final_cleanup_seconds=time.monotonic()-post_start;tifffile.imwrite(folder/'mask.tif',restored,compression='deflate')
    (folder/'input.tif').unlink() # avoid duplicated large deployment rasters
    item=dict(key=key,case=case,geojson_sha256=sha(folder/'mask.geojson'),raster_sha256=sha(folder/'mask.tif'),
        final_cleanup_seconds=final_cleanup_seconds,cleanup=cleanup,final_native_pixel_cleanup=vector.get('final_native_pixel_region_cleanup'),compact_boundary_report=vector['compact_boundary_report'],vector_hole_cleanup=vector['small_vector_hole_cleanup'],cleanup_seconds=cleanup_seconds,conversion=report,geometry=geometry_stats(vector['features']),
        all_predictions_frozen_before_scoring=True,annotation_used_by_export=False)
    save(record,item);return item,folder


def evaluate_model(study,data,folder):
    study,folder=Path(study),Path(folder);manifest=load(folder/'PREDICTIONS.json');all_y=[];all_ranks=[];cases=[];roles=len(data.registry['ontology']['roles'])
    for p in manifest['predictions']:
        case=p['case'];row=data.rows[case];path=folder/(case+'.npz')
        if sha(path)!=p['sha256']:raise ValueError('Prediction changed before evaluation')
        with np.load(path) as f:ranks=f['top5']
        item,vector_folder=cache_vector(study,data,case,ranks)
        truth=broad_truth(row);regions=tifffile.imread(data.geo/'cases'/case/'regions.native.tif');fractions,coverage=target_counts(regions,truth,roles)
        y=identity_labels(row,data.geo);keep=(y>0)&(fractions.sum(1)>0);all_y.append(y[keep]);all_ranks.append(ranks[keep])
        raw=semantic_mask(regions,ranks);restored=tifffile.imread(vector_folder/'mask.tif')
        native=pixel_scores(truth,raw,roles);final=pixel_scores(truth,restored,roles)
        geometry_time=load(data.geo/'cases'/case/'GEOMETRY_FROZEN.json')['timing']['total_cached_seconds']
        descriptors=load(study/'descriptors/cases'/case/'DESCRIPTORS.json');required=manifest['required_descriptors']
        descriptor_time=(descriptors['distribution_seconds'] if set(required)&{'covariance','quantile','spatial'} else 0.)+(descriptors['neighbor_seconds'] if 'neighbor' in required else 0.)
        total=geometry_time+descriptor_time+p['warm_head_seconds']+p['predictor_input_seconds']+item['cleanup_seconds']+item['final_cleanup_seconds']+item['conversion']['runtime_seconds']
        cases.append(dict(case=case,dataset=row['dataset'],source_group=row['group'],split=row['split'],
            group=score_units(y[keep],ranks[keep],data.classes),raw=native,final=final,
            cached_deployment_seconds=total,geometry_seconds=geometry_time,additional_descriptor_seconds=descriptor_time,
            warm_head_seconds=p['warm_head_seconds'],predictor_input_seconds=p['predictor_input_seconds'],
            cleanup_seconds=item['cleanup_seconds']+item['final_cleanup_seconds'],conversion_seconds=item['conversion']['runtime_seconds'],
            vector=str(vector_folder/'mask.geojson'),vector_record=item,top1=ranks[:,0].tolist()))
    scores=score_units(np.concatenate(all_y),np.concatenate(all_ranks),data.classes)
    iou=float(np.mean([c['final']['macro_iou'] for c in cases]));raw=float(np.mean([c['raw']['macro_iou'] for c in cases]))
    excess=float(np.mean([max(c['cached_deployment_seconds']-1,0) for c in cases]));latency=.05*excess
    elegance=.002*np.log2(1+manifest['native_head_count'])+.002*manifest['additional_structural_stages']
    accuracy=.5*iou+.3*scores['balanced_accuracy']+.2*scores['top1']
    result=dict(id=manifest['id'],name=manifest['recipe']['name'],description=manifest['recipe']['description'],**scores,
        mean_final_geojson_iou=iou,mean_raw_raster_iou=raw,mean_cached_deployment_seconds=float(np.mean([c['cached_deployment_seconds'] for c in cases])),
        images_above_one_second=sum(c['cached_deployment_seconds']>1 for c in cases),mean_head_seconds=float(np.mean([c['warm_head_seconds'] for c in cases])),
        accuracy_score=accuracy,latency_penalty=latency,elegance_penalty=float(elegance),selection_score=float(accuracy-latency-elegance),
        native_head_count=manifest['native_head_count'],additional_structural_stages=manifest['additional_structural_stages'],cases=cases,
        per_dataset={dataset:float(np.mean([c['final']['macro_iou'] for c in cases if c['dataset']==dataset])) for dataset in sorted({c['dataset'] for c in cases})})
    save(folder/'RESULTS.json',result);return result
