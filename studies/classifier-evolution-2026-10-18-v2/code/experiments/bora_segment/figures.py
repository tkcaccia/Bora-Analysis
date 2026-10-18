"""Publication figures from frozen reports and existing viewer thumbnails."""
import argparse
from pathlib import Path
import json
import colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
from PIL import Image


def run(reports,controlled,output,examples):
    reports,controlled,output,examples=map(Path,(reports,controlled,output,examples));output.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text());summary=load(reports/'REPORT.json');comparison=load(controlled/'CONTROLLED_FINAL.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    fig,ax=plt.subplots(figsize=(6.6,3.6));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    def box(x,y,w,h,label,colour='#eef3f6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor=colour,edgecolor='#506a78',lw=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.6)
    box(.02,.74,.27,.17,'Calibrated H&E + Virchow2\nfull unselected spatial tokens')
    box(.37,.74,.27,.17,'45 µm physical tile evidence\nanonymous coarse groups')
    box(.72,.74,.26,.17,'Fixed RGB conversion 19\nno fine annotation')
    box(.02,.41,.27,.17,'Bora-AF appearance fields\nH&E-guided regularization')
    box(.37,.41,.27,.17,'Connected native regions\nfreeze boundaries and IDs')
    box(.72,.41,.26,.17,'Pool evidence per region\nor aggregate its tile ranks')
    box(.02,.08,.27,.17,'Training references\nattach after geometry freezes','#f4efe6')
    box(.37,.08,.27,.17,'Native fastPLS CPU head\nfit training groups only')
    box(.72,.08,.26,.17,'One top 1 and top 5\nfor each whole tissue segment')
    for a,b in [((.29,.825),(.37,.825)),((.64,.825),(.72,.825)),((.85,.74),(.85,.65)),((.85,.65),(.15,.65)),((.15,.65),(.15,.58)),((.29,.495),(.37,.495)),((.64,.495),(.72,.495)),((.85,.41),(.85,.33)),((.85,.33),(.51,.33)),((.51,.33),(.51,.25)),((.29,.165),(.37,.165)),((.64,.165),(.72,.165))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#374e5a','lw':.9})
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);fig.savefig(output/'figure1-region-workflow.png');plt.close(fig)
    names=['TILES','GRAPH','FBS-RGB','Bora-AF','Bora-VAF'];labels=['Tile input','Graph','Bilateral','Bora-AF','Bora-VAF']
    fig,axes=plt.subplots(2,1,figsize=(6.6,4.6),layout='constrained',gridspec_kw={'height_ratios':[2,1.3]})
    axes[0].boxplot([[r['methods'][n]['score']['macro_iou'] for r in comparison['cases']] for n in names],tick_labels=labels,
        showmeans=True,meanprops={'marker':'D','markerfacecolor':'#20597b','markeredgecolor':'white','markersize':5},medianprops={'color':'#333'})
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Native pixel macro IoU');axes[0].set_title('A  Controlled refinement of reference-derived tile masks',loc='left',fontsize=9.5,weight='bold')
    stats={r['name']:r for r in comparison['ranking']};times=[stats[n]['mean_seconds'] for n in names]
    axes[1].scatter(times,np.arange(len(names)),s=30,color=['#999','#999','#999','#20597b','#6b8794'])
    for i,t in enumerate(times):axes[1].annotate(f'{t:.3f} s',(t,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
    axes[1].set_yticks(np.arange(len(names)),labels);axes[1].invert_yaxis();axes[1].set_xlim(-.01,max(times)+.14);axes[1].set_xlabel('Mean cached refinement time per ROI (s)')
    axes[1].set_title('B  Two CPU threads; encoder and RGB conversion excluded',loc='left',fontsize=9.5,weight='bold')
    fig.savefig(output/'figure2-controlled-refinement.png');plt.close(fig)
    names=['R01','R02','R03','R04','R05'];labels=['Tile vote','Region mean','Mean + SD','Composition','Neighbor context']
    val={r['family']:r for r in summary['results']['validation']['models']};test={r['family']:r for r in summary['results']['followup_test']['models']}
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.4));y=np.arange(5)
    for ax,key,title in [(axes[0],'balanced_accuracy','A  Segment balanced accuracy'),(axes[1],'mean_native_pixel_iou','B  Native mask IoU')]:
        ax.barh(y-.17,[val[n][key] for n in names],height=.32,color='#b8ccd8',label='Validation')
        ax.barh(y+.17,[test[n][key] for n in names],height=.32,color='#235e7c',label='Reused test')
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title,fontsize=8.8,weight='bold',loc='left');ax.set_xlim(0,.60 if key=='balanced_accuracy' else .35)
        ax.set_xlabel('Balanced accuracy' if key=='balanced_accuracy' else 'Case-mean macro IoU')
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=.23,right=.98,top=.89,bottom=.22,wspace=.27)
    fig.legend(handles=[Patch(facecolor='#b8ccd8',label='Validation'),Patch(facecolor='#235e7c',label='Reused test')],
        loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.60,0))
    fig.savefig(output/'figure3-segment-heads.png');plt.close(fig)
    result=load(reports/'classifiers-followup-test-v1/RESULTS.json');primary=next(m for m in result['ranking'] if m['name']==result['selected_segment_classifier'])
    candidates=sorted([r for r in primary['cases'] if r['dataset']=='tiger'],key=lambda r:r['pixel_score']['macro_iou']);case=candidates[len(candidates)//2]['case']
    fields=[('original','Original H&E'),('reference','Expert tissue roles'),('coarse','Anonymous tile input'),('converted19','Fixed RGB conversion'),('regions','Frozen Bora regions'),(primary['name'],'Whole-segment identity')]
    fig,axes=plt.subplots(2,3,figsize=(6.6,5.1))
    registry=load(controlled/'REGISTRY.json');row=next(r for r in registry['cases'] if r['case']==case);extent=row['sample']['native_shape_yx'][1]*row['sample']['native_mpp_xy'][0]
    for ax,(name,label) in zip(axes.ravel(),fields):
        im=np.asarray(Image.open(examples/case/(name+'.jpg')));ax.imshow(im);ax.axis('off');ax.set_title(label,fontsize=8)
        x0=im.shape[1]*.06;x1=x0+100/extent*im.shape[1];yy=im.shape[0]*.92
        ax.plot([x0,x1],[yy,yy],color='white',lw=4);ax.plot([x0,x1],[yy,yy],color='black',lw=2)
    viewer=load(reports/'viewer-followup-test/data.json');item=next(r for r in viewer['cases'] if r['case']==case)
    prediction=next(r for r in item['models'] if r['name']==primary['name'])
    present=set(row['ontology_raw_to_role'].values())|{r[0] for r in prediction['top5']};present.discard(0)
    handles=[Patch(facecolor=colorsys.hsv_to_rgb(((g-1)*.618034)%1,.65,.95),label=viewer['roles'][g-1].replace('_',' ')) for g in sorted(present)]
    fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.14,wspace=.04,hspace=.17)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=6.8,columnspacing=1.,bbox_to_anchor=(.5,.01))
    fig.savefig(output/'figure4-segment-example.png');plt.close(fig)
    (output/'PROVENANCE.json').write_text(json.dumps(dict(example_case=case,example_selection='Median primary R02 test IoU among16 eligible TIGER ROIs; not selected for best appearance',scale_bars_um=100,primary_model=primary['name'],reports=str(reports),controlled=str(controlled)),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reports','controlled','output','examples'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.reports,a.controlled,a.output,a.examples)


if __name__=='__main__':main()
