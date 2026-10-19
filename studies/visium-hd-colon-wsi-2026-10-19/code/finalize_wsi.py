"""CPU-only whole-slide group readout, cleanup, and native vector export."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import tifffile


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def geometry(root):
    marker=root/"GEOMETRY_COMPLETE.json"
    if not marker.exists():raise FileNotFoundError("Full-slide geometry is not complete")
    data=json.loads(marker.read_text())
    if not data["all_windows_completed"] or data["processed"]+data["background_windows"]!=data["total_windows"]:
        raise ValueError("Incomplete slide geometry")
    return data


def group_adjacency(mask,mpp,n_groups=6,block_pixels=2048):
    h,w=mask.shape
    graph=np.zeros((n_groups,n_groups),np.float64)
    counts=np.zeros(n_groups+1,np.int64)
    for y in range(0,h,block_pixels):
        for x in range(0,w,block_pixels):
            ch,cw=min(block_pixels,h-y),min(block_pixels,w-x)
            block=np.asarray(mask[y:min(h,y+ch+1),x:min(w,x+cw+1)],np.uint8)
            counts+=np.bincount(block[:ch,:cw].ravel(),minlength=n_groups+1)
            right_width=min(cw,w-x-1)
            down_height=min(ch,h-y-1)
            for left,right,length in ((block[:ch,:right_width],block[:ch,1:right_width+1],mpp[1]),
                                      (block[:down_height,:cw],block[1:down_height+1,:cw],mpp[0])):
                use=(left>0)&(right>0)&(left!=right)
                if use.any():
                    a=left[use].astype(np.int16)-1;b=right[use].astype(np.int16)-1
                    edges=np.bincount((a*n_groups+b).ravel(),minlength=n_groups*n_groups)
                    graph+=edges.reshape(n_groups,n_groups)*length
                    graph+=edges.reshape(n_groups,n_groups).T*length
    return graph,counts


def classify(source,output,model_path):
    from bora.classifier import NeighborContextClassifier
    info=geometry(source)
    h,w=info["native_shape_xy"][1],info["native_shape_xy"][0]
    shape=(h,w)
    groups=np.memmap(source/"groups.native.dat",mode="r",dtype=np.uint8,shape=shape)
    graph,counts=group_adjacency(groups,info["native_mpp_xy"])
    mass=np.asarray(info["group_mass"],np.float64)
    if not np.array_equal(counts[1:],mass.astype(np.int64)):
        raise ValueError("Checkpointed group masses disagree with native raster")
    means=np.zeros((6,1280),np.float64)
    present=mass>0
    sums=np.asarray(info["group_sums"],np.float64)
    means[present]=sums[present]/mass[present,None]
    neighbours=means.copy();total=graph.sum(1)
    has=total>0
    neighbours[has]=(graph@means)[has]/total[has,None]
    descriptor=np.concatenate([means,means-neighbours],axis=1)
    model=NeighborContextClassifier.load(model_path)
    predicted=np.zeros(6,np.uint8)
    predicted[present]=model.predict(descriptor[present])
    lookup=np.r_[np.uint8(0),predicted]
    output.mkdir(parents=True,exist_ok=False)
    raw=np.memmap(output/"classified.raw.dat",mode="w+",dtype=np.uint8,shape=shape)
    for y in range(0,h,512):
        last=min(h,y+512)
        raw[y:last]=lookup[np.asarray(groups[y:last])]
    raw.flush()
    report=dict(native_shape_yx=[h,w],native_mpp_xy=info["native_mpp_xy"],
        group_mass=mass.tolist(),group_class_ids=predicted.tolist(),
        group_class_names={str(i+1):model.names_.get(int(predicted[i])) for i in range(6) if present[i]},
        group_adjacency_um=graph.tolist(),head_input_shape=list(descriptor[present].shape),
        classifier="Neighbor context, native fastPLS",classifier_fit_seconds=model.fit_seconds_,
        expert_annotation_read=False,source_geometry=str(source))
    atomic_json(output/"CLASSIFIED_COMPLETE.json",report)
    return report


def cleanup(output):
    from bora.mask_cleanup import fill_small_regions
    from scipy import ndimage as ndi
    def four_connected_tiny_parts(image,minimum=20):
        """Remove 4-connected pixel parts that become distinct GeoJSON parts."""
        result=image.copy();removed=changed=0
        cross=ndi.generate_binary_structure(2,1)
        for _ in range(4):
            pass_removed=0
            for value in np.unique(result):
                if value==0:continue
                components,n=ndi.label(result==value,structure=cross)
                sizes=np.bincount(components.ravel())
                slices=ndi.find_objects(components)
                for component in np.flatnonzero((sizes<minimum)&(np.arange(len(sizes))>0)):
                    sl=slices[component-1]
                    if sl is None:continue
                    y0=max(0,sl[0].start-1);y1=min(result.shape[0],sl[0].stop+1)
                    x0=max(0,sl[1].start-1);x1=min(result.shape[1],sl[1].stop+1)
                    local_components=components[y0:y1,x0:x1]
                    at=local_components==component
                    ring=ndi.binary_dilation(at,structure=cross)&~at
                    values,contacts=np.unique(result[y0:y1,x0:x1][ring],return_counts=True)
                    possibilities=[(int(c),-int(v),int(v)) for v,c in zip(values,contacts) if v>0 and v!=value]
                    winner=max(possibilities)[-1] if possibilities else 0
                    result[y0:y1,x0:x1][at]=winner
                    pass_removed+=1;changed+=int(sizes[component])
            removed+=pass_removed
            if not pass_removed:break
        else:raise ValueError("Four-connected tiny-part cleanup did not converge")
        return result,dict(four_connected_parts_filled=removed,four_connected_pixels_filled=changed)
    def four_connected_tiny_holes(image,minimum=20):
        result=image.copy();cross=ndi.generate_binary_structure(2,1)
        components,n=ndi.label(result==0,structure=cross)
        sizes=np.bincount(components.ravel())
        border=set(np.r_[components[0],components[-1],components[:,0],components[:,-1]].tolist())
        slices=ndi.find_objects(components)
        filled=pixels=0
        for component in np.flatnonzero((sizes<minimum)&(np.arange(len(sizes))>0)):
            if component in border:continue
            sl=slices[component-1]
            if sl is None:continue
            y0=max(0,sl[0].start-1);y1=min(result.shape[0],sl[0].stop+1)
            x0=max(0,sl[1].start-1);x1=min(result.shape[1],sl[1].stop+1)
            at=components[y0:y1,x0:x1]==component
            ring=ndi.binary_dilation(at,structure=cross)&~at
            values,contacts=np.unique(result[y0:y1,x0:x1][ring],return_counts=True)
            possible=[(int(c),-int(v),int(v)) for v,c in zip(values,contacts) if v>0]
            if not possible:continue
            local=result[y0:y1,x0:x1]
            local[at]=max(possible)[-1]
            filled+=1;pixels+=int(sizes[component])
        return result,dict(four_connected_holes_filled=filled,four_connected_hole_pixels_filled=pixels)
    report=json.loads((output/"CLASSIFIED_COMPLETE.json").read_text())
    h,w=report["native_shape_yx"]
    raw=np.memmap(output/"classified.raw.dat",mode="r",dtype=np.uint8,shape=(h,w))
    cleaned=np.memmap(output/"classified.cleaned.dat",mode="w+",dtype=np.uint8,shape=(h,w))
    totals=defaultdict(int);started=monotonic();windows=0
    for y in range(0,h,4096):
        for x in range(0,w,4096):
            y1,x1=min(h,y+4096),min(w,x+4096)
            ay0,ax0=max(0,y-32),max(0,x-32)
            ay1,ax1=min(h,y1+32),min(w,x1+32)
            local=np.asarray(raw[ay0:ay1,ax0:ax1])
            if not local.any():continue
            result,stats=fill_small_regions(local,20)
            result,extra=four_connected_tiny_parts(result,20)
            result,holes=four_connected_tiny_holes(result,20)
            cleaned[y:y1,x:x1]=result[y-ay0:y1-ay0,x-ax0:x1-ax0]
            for k in ("filled_regions","filled_pixels","filled_holes","hole_pixels",
                      "isolated_regions_filled_with_background"):
                totals[k]+=stats[k]
            for k in extra:totals[k]+=extra[k]
            for k in holes:totals[k]+=holes[k]
            windows+=1
    cleaned.flush()
    audit=dict(native_shape_yx=[h,w],minimum_pixels=20,strictly_less_than_20=True,
        halo_pixels=32,window_core_pixels=4096,nonempty_windows=windows,
        raster_connectivity_8_then_4_to_match_vector_parts=True,
        local_cleanup_event_sums=dict(totals),seconds=monotonic()-started)
    atomic_json(output/"CLEANED_COMPLETE.json",audit)
    return audit


def write_ome(output):
    report=json.loads((output/"CLASSIFIED_COMPLETE.json").read_text())
    if not (output/"VECTOR_COMPLETE.json").exists():raise FileNotFoundError("Vector export incomplete")
    h,w=report["native_shape_yx"]
    mask=np.memmap(output/"classified.final.dat",mode="r",dtype=np.uint8,shape=(h,w))
    mpp=report["native_mpp_xy"]
    tifffile.imwrite(output/"mask.ome.tif",mask,bigtiff=True,tile=(512,512),
        compression="deflate",photometric="minisblack",ome=True,
        metadata={"axes":"YX","PhysicalSizeX":float(mpp[0]),"PhysicalSizeY":float(mpp[1]),
                  "PhysicalSizeXUnit":"µm","PhysicalSizeYUnit":"µm"})
    with tifffile.TiffFile(output/"mask.ome.tif") as t:
        if t.pages[0].shape!=(h,w):raise ValueError("Written OME mask has wrong geometry")
    return output/"mask.ome.tif"


def polygons(output):
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import Affine
    import shapely
    from shapely.geometry import shape,mapping,Polygon,MultiPolygon
    from shapely.ops import unary_union
    report=json.loads((output/"CLASSIFIED_COMPLETE.json").read_text())
    h,w=report["native_shape_yx"]
    mask=np.memmap(output/"classified.cleaned.dat",mode="r",dtype=np.uint8,shape=(h,w))
    by_class=defaultdict(list)
    for y in range(0,h,4096):
        for x in range(0,w,4096):
            tile=np.asarray(mask[y:min(h,y+4096),x:min(w,x+4096)])
            if not tile.any():continue
            transform=Affine(1,0,x,0,1,y)
            for geom,value in shapes(tile,mask=tile>0,connectivity=4,transform=transform):
                candidate=shape(geom)
                by_class[int(value)].append(candidate if candidate.is_valid else shapely.make_valid(candidate))
    source_classes=sorted(by_class)
    originals=[]
    for value in source_classes:
        merged=unary_union(by_class[value])
        if not merged.is_valid:raise ValueError(f"Invalid merged class geometry {value}")
        originals.append(merged)
    del by_class
    original_area=sum(g.area for g in originals)
    tolerances=[min(3/float(v) for v in report["native_mpp_xy"]),6.,3.,1.,0.]
    fitted=originals;selected=0.;agreements={}
    for tolerance in tolerances:
        if tolerance==0:break
        try:candidate=shapely.coverage_simplify(originals,tolerance)
        except Exception:continue
        candidate=list(candidate)
        if len(candidate)!=len(originals) or not all(g.is_valid for g in candidate):continue
        per=[];bad_small=False
        for a,b in zip(originals,candidate):
            intersection=a.intersection(b).area
            per.append(intersection/max(1e-10,a.area+b.area-intersection))
            parts=[b] if b.geom_type=="Polygon" else list(b.geoms)
            if any(p.area<20 or any(Polygon(r).area<20 for r in p.interiors) for p in parts):
                bad_small=True
        altered=sum(a.symmetric_difference(b).area for a,b in zip(originals,candidate))
        candidate_area=sum(g.area for g in candidate)
        if (min(per)>=.97 and altered/max(original_area,1)<.03 and
                abs(candidate_area-original_area)/max(original_area,1)<.015 and not bad_small):
            fitted=candidate;selected=float(tolerance)
            agreements=dict(minimum_class_iou=min(per),changed_label_area_fraction=altered/max(original_area,1),
                changed_tissue_area_fraction=abs(candidate_area-original_area)/max(original_area,1))
            break
    names={int(k):v for k,v in report["group_class_names"].items()}
    # The group-name map is keyed by anonymous ID; final labels may be shared.
    class_names={int(report["group_class_ids"][i-1]):name for i,name in names.items() if name}
    mpp=np.asarray(report["native_mpp_xy"])
    geo=output/"mask.geojson"
    with geo.open("w") as f:
        f.write('{"type":"FeatureCollection","coordinate_space":"native_image_pixel_edges",')
        f.write('"origin":"top_left","native_shape_yx":'+json.dumps([h,w])+',')
        f.write('"native_mpp_xy":'+json.dumps(mpp.tolist())+',')
        f.write('"features":[')
        for index,(value,g) in enumerate(zip(source_classes,fitted)):
            if index:f.write(',')
            name=class_names.get(value,f"class_{value}")
            feature=dict(type="Feature",geometry=mapping(g),properties=dict(value=value,name=name,
                classification=dict(name=name),predicted_tissue_id=value,predicted_tissue=name,
                objectType="annotation",area_pixels=float(g.area),
                area_um2=float(g.area*np.prod(mpp)),tissue_identity_source="Neighbor context, native fastPLS"))
            json.dump(feature,f,separators=(",",":"),allow_nan=False)
        f.write('],"compact_boundary_report":')
        json.dump(dict(method="topology-preserving coverage_simplify",selected_tolerance_pixels=selected,
            selected_tolerance_um=selected*float(np.mean(mpp)),**agreements),f,separators=(",",":"))
        f.write('}\n')
    audit=dict(classes=source_classes,feature_count=len(source_classes),
        original_polygon_area_pixels=float(original_area),
        output_polygon_area_pixels=float(sum(g.area for g in fitted)),
        selected_tolerance_pixels=selected,agreements=agreements,
        mask_geojson=str(geo),ome_mask=str(output/"mask.ome.tif"))
    atomic_json(output/"VECTOR_COMPLETE.json",audit)
    return audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--stage",choices=("classify","cleanup","vector","all"),default="all")
    p.add_argument("--vector-core",type=int,default=4096)
    p.add_argument("--fit-core",type=int,default=2048)
    p.add_argument("--vector-workers",type=int,default=3)
    a=p.parse_args()
    if a.stage in ("classify","all"):
        report=classify(a.source,a.output,a.model)
        print(json.dumps(dict(stage="classify",classes=report["group_class_names"])),flush=True)
    if a.stage in ("cleanup","all"):
        audit=cleanup(a.output)
        print(json.dumps(dict(stage="cleanup",**audit)),flush=True)
    if a.stage in ("vector","all"):
        from vectorize_wsi import export
        audit=export(a.output,core=a.vector_core,fit_core=a.fit_core,
                     workers=a.vector_workers)
        print(json.dumps(dict(stage="vector",**audit)),flush=True)
        path=write_ome(a.output)
        print(json.dumps(dict(stage="ome",path=str(path))),flush=True)


if __name__=="__main__":main()
