"""Physically defined feature units; annotations never affect their geometry."""
import math
import numpy as np


def tile_boxes(shape_yx,mpp_xy,width_um=45.):
    h,w=map(int,shape_yx);mx,my=map(float,mpp_xy)
    if min(h,w,mx,my,width_um)<=0:raise ValueError('Positive geometry required')
    xs=np.unique(np.clip(np.rint(np.arange(math.ceil(w*mx/width_um)+1)*width_um/mx),0,w).astype(int))
    ys=np.unique(np.clip(np.rint(np.arange(math.ceil(h*my/width_um)+1)*width_um/my),0,h).astype(int))
    return np.asarray([(x0,y0,x1-x0,y1-y0) for y0,y1 in zip(ys[:-1],ys[1:])
                       for x0,x1 in zip(xs[:-1],xs[1:])],dtype=np.int32)


def pool_tokens(tokens,boxes,patch_px=14):
    """Area-weight token-grid cells, not encoder receptive fields or RGB pixels."""
    means=[];centres=[]
    for x,y,w,h in boxes:
        x0,x1=int(x//patch_px),min(tokens.shape[1],int(math.ceil((x+w)/patch_px)))
        y0,y1=int(y//patch_px),min(tokens.shape[0],int(math.ceil((y+h)/patch_px)))
        ix=np.arange(x0,x1);iy=np.arange(y0,y1)
        wx=np.maximum(0,np.minimum((ix+1)*patch_px,x+w)-np.maximum(ix*patch_px,x)).astype(np.float32)
        wy=np.maximum(0,np.minimum((iy+1)*patch_px,y+h)-np.maximum(iy*patch_px,y)).astype(np.float32)
        weight=wy[:,None]*wx[None,:];weight/=weight.sum()
        block=np.asarray(tokens[y0:y1,x0:x1],dtype=np.float32)
        means.append(np.einsum('ij,ijc->c',weight,block,optimize=False))
        cx=min(tokens.shape[1]-1,int((x+.5*w)//patch_px));cy=min(tokens.shape[0]-1,int((y+.5*h)//patch_px))
        centres.append(np.asarray(tokens[cy,cx],dtype=np.float32))
    return np.asarray(means,dtype=np.float32),np.asarray(centres,dtype=np.float32)


def region_targets(reference,boxes,working_shape_yx,raw_to_role,roles):
    """Exact native-pixel counts; zero and unmapped aggregates remain unlabelled."""
    target=np.zeros((len(boxes),roles),dtype=np.float32)
    coverage=np.zeros(len(boxes),dtype=np.float32)
    max_code=max(int(reference.max()),max(map(int,raw_to_role),default=0))
    lookup=np.zeros(max_code+1,dtype=np.int32)
    for code,role in raw_to_role.items():lookup[int(code)]=role
    sy=reference.shape[0]/working_shape_yx[0];sx=reference.shape[1]/working_shape_yx[1]
    for i,(x,y,w,h) in enumerate(boxes):
        x0,x1=round(x*sx),round((x+w)*sx);y0,y1=round(y*sy),round((y+h)*sy)
        tile=lookup[reference[y0:y1,x0:x1]]
        counts=np.bincount(tile.ravel(),minlength=roles+1)[1:]
        if tile.size:coverage[i]=counts.sum()/tile.size
        if counts.sum():target[i]=counts/counts.sum()
    return target,coverage


def training_rows(fractions,coverage,tissue_fraction,*,pure=False):
    keep=(coverage>=(.90 if pure else .50))&(tissue_fraction>=.50)
    if pure:keep&=fractions.max(1)>=.95
    return keep


def ranked_labels(scores,class_ids,k=5):
    scores=np.asarray(scores);class_ids=np.asarray(class_ids)
    if scores.ndim!=2 or scores.shape[1]!=len(class_ids) or not np.isfinite(scores).all():
        raise ValueError('Finite class scores in declared class order required')
    # Stable ties retain the registered class ordering, never test labels.
    return class_ids[np.argsort(-scores,axis=1,kind='stable')[:,:min(k,len(class_ids))]]
