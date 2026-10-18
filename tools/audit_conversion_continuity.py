"""Check fixed RGB conversion against preserved scientific source on CPU fixtures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from bora.conversion import convert_rgb, select_dimensions, observations


def run(reference_code):
    sys.path.append(str(reference_code))
    from experiments.virchow2_multi.methods import Context, align_variant, run_method
    records=[]
    for seed in [19,101,314159]:
        rng=np.random.default_rng(seed)
        coarse=np.ones((360,360),np.uint8)
        coarse[:180,180:]=2;coarse[180:,:180]=3;coarse[180:,180:]=4
        values=rng.normal(size=(26,26,1280)).astype(np.float32)
        values[13:,:,:16]+=2;values[:,13:,16:32]+=2
        palette=np.array([[242,84,84],[84,130,242],[176,242,84],[242,84,222]],np.float32)
        variants={name:dict(values=rng.normal(size=(n,n,1280)).astype(np.float32),shape_yx=[shape,shape])
            for name,n,shape in [('scale_1',13,180),('scale_2',7,90)]}
        context=Context(np.zeros((360,360,3),np.uint8),coarse,values,palette)
        x,y=observations(values,coarse)
        np.testing.assert_array_equal(select_dimensions(x,y),context.selected)
        aligned={name:align_variant(spec['values'][...,context.selected],spec['shape_yx'],context) for name,spec in variants.items()}
        probabilities=[run_method(i,context,aligned,return_probabilities=True)[-1] for i in [9,19]]
        expected=context.probability_rgb(np.mean(np.stack(probabilities),axis=0,dtype=np.float32))
        actual,_=convert_rgb(values,coarse,palette,variants)
        np.testing.assert_array_equal(actual,expected)
        records.append(dict(seed=seed,selected_coordinates_identical=True,rgb_bitwise_identical=True))
    return dict(ok=True,fixtures=records,encoder_extraction_performed=False,gpu_work_performed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-code',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):report=run(a.reference_code)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
