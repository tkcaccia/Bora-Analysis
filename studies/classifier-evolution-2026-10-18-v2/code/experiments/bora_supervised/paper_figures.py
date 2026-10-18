"""Scientific figures from frozen reports; no inference or annotation changes."""
import argparse
import json
from pathlib import Path
import colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
from PIL import Image


def run(results,output):
    results,output=Path(results),Path(output);output.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text())
    controlled=load(results/'CONTROLLED_FINAL.json');test=load(results/'test-v1/RESULTS.json');registry=load(results/'REGISTRY.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    fig,ax=plt.subplots(figsize=(6.6,3.35));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#f0f4f6',edgecolor='#466070',linewidth=.9))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.2)
    box(.02,.71,.20,.2,'Original H&E\nphysical pixel spacing')
    box(.02,.39,.20,.2,'Coarse class mask\nfixed evidence RGB')
    box(.31,.53,.26,.22,'Bora-AF\nclass appearance fields\nH&E graph conductance')
    box(.67,.53,.30,.22,'Appearance likelihoods\nwhole-field competition\npost-refinement glass removal')
    box(.02,.06,.29,.17,'Full Virchow2 spatial tokens\n45 µm tiles or H&E superpixels')
    box(.40,.06,.23,.17,'Native fastPLS CPU head\ntrain groups only')
    box(.73,.06,.24,.17,'Ranked tissue options\ntop 1 and top 5')
    for a,b in [((.22,.8),(.31,.7)),((.22,.5),(.31,.6)),((.57,.64),(.67,.64)),((.31,.145),(.40,.145)),((.63,.145),(.73,.145))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#263e4c','lw':1.2})
    ax.text(.5,.96,'Geometric refinement and supervised tissue identification',ha='center',weight='bold',fontsize=10)
    ax.text(.5,.31,'Bora-VAF replaces graph conduction with bending and fitted log-variance fields',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(output/'figure1-workflow.png');plt.close(fig)
    names=['TILES','GRAPH','FBS-RGB','Bora-AF','Bora-VAF'];labels=['Tile mask','Graph\ndiffusion','Bilateral\nconverged','Bora-AF','Bora-VAF']
    fig,axes=plt.subplots(2,1,figsize=(6.6,5.15),gridspec_kw={'height_ratios':[2.3,1.3]},layout='constrained')
    values=[[r['methods'][name]['score']['macro_iou'] for r in controlled['cases']] for name in names]
    axes[0].boxplot(values,tick_labels=labels,showmeans=True,meanprops={'marker':'D','markerfacecolor':'#20597b','markeredgecolor':'white','markersize':5},medianprops={'color':'#333'})
    axes[0].set_ylabel('Native mask IoU');axes[0].set_ylim(0,1);axes[0].set_title('A  Controlled refinement on 39 development ROIs',loc='left',weight='bold',fontsize=10)
    ranking={r['name']:r for r in controlled['ranking']};times=[ranking[n]['mean_seconds'] for n in names]
    axes[1].scatter(times,np.arange(len(names)),c=['#777','#777','#777','#20597b','#667b88'],s=40)
    for i,t in enumerate(times):axes[1].annotate(f'{t:.3f} s',(t,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
    axes[1].set_yticks(np.arange(len(names)),['Tiles','Graph','Bilateral','Bora-AF','Bora-VAF']);axes[1].invert_yaxis();axes[1].set_xlim(-.01,max(times)+.13)
    axes[1].set_xlabel('Mean cached CPU refinement time per ROI (s)');axes[1].set_title('B  Two numerical threads and matched input fields',loc='left',weight='bold',fontsize=10)
    fig.savefig(output/'figure2-refinement.png');plt.close(fig)
    names=['S01-c24','S02-c16','S06-c24','S07-c24','RIDGE-a1','S08-c48','S09-c48'];rank={r['name']:r for r in test['ranking']}
    labels=['Centre PLS','Tile PLS','Tile PLS-LDA','Context PLS-LDA','Ridge','Region PLS','Region fractions PLS']
    fig,axes=plt.subplots(2,1,figsize=(6.6,5.6),layout='constrained')
    y=np.arange(len(names));colours=['#bcc9d1']*7;colours[3]='#c98c40';colours[6]='#20597b'
    v=np.array([rank[n]['mean_native_pixel_macro_iou'] for n in names])*100
    axes[0].barh(y,v,color=colours)
    for i,a in enumerate(v):axes[0].text(a+.5,i,f'{a:.1f}',va='center',fontsize=8)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,42);axes[0].set_xlabel('Mean native-pixel macro IoU (%)')
    axes[0].set_title('A  Supervised segmentation on 62 eligible held-out ROIs',loc='left',fontsize=10,weight='bold')
    axes[1].barh(y-.16,[100*rank[n]['mean_pixel_top1'] for n in names],height=.3,color='#476d85',label='Top 1')
    axes[1].barh(y+.16,[100*rank[n]['mean_pixel_top5'] for n in names],height=.3,color='#bacbd5',label='Top 5')
    axes[1].set_yticks(y,labels);axes[1].invert_yaxis();axes[1].set_xlim(0,105);axes[1].set_xlabel('Mean annotated-pixel coverage (%)');axes[1].legend(loc='lower right',frameon=False)
    axes[1].set_title('B  Tissue identity coverage and geometric overlap differ',loc='left',fontsize=10,weight='bold')
    fig.savefig(output/'figure3-tissue-heads.png');plt.close(fig)
    selected=[]
    for dataset in ['bcss','tiger','puma']:
        rows=sorted([r for r in test['cases'] if r['dataset']==dataset],key=lambda r:r['models']['S09-c48']['macro_iou']);selected.append(rows[len(rows)//2])
    samples={r['case']:r for r in registry['cases']};fig=plt.figure(figsize=(6.6,7.5));grid=fig.add_gridspec(4,4,height_ratios=[1,1,1,1.05],hspace=.13,wspace=.035)
    for i,row in enumerate(selected):
        case=row['case'];sample=samples[case]['sample']
        for j,(key,title) in enumerate([('original','H&E'),('reference','Expert roles'),('RIDGE-a1','Ridge'),('S09-c48','Region fractions')]):
            ax=fig.add_subplot(grid[i,j]);image=np.asarray(Image.open(output/'source_examples'/case/(key+'.jpg')));ax.imshow(image);ax.axis('off')
            if i==0:ax.set_title(title,fontsize=8.2)
            if j==0:ax.text(0,1.02,case,transform=ax.transAxes,fontsize=8,weight='bold')
            length=100/(sample['working_shape_yx'][1]*sample['working_mpp_xy'][0])*image.shape[1]
            ax.plot([image.shape[1]*.06,image.shape[1]*.06+length],[image.shape[0]*.9]*2,color='white',lw=4)
            ax.plot([image.shape[1]*.06,image.shape[1]*.06+length],[image.shape[0]*.9]*2,color='black',lw=2)
            if key=='S09-c48':ax.text(.03,.04,f"IoU {row['models'][key]['macro_iou']:.3f}",transform=ax.transAxes,fontsize=7.5,color='white',bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':2})
    legend_ax=fig.add_subplot(grid[3,:]);legend_ax.axis('off')
    roles=registry['ontology']['roles'];patches=[Patch(facecolor=colorsys.hsv_to_rgb((i*.618034)%1,.65,.95),label=f'{i+1} '+r.replace('_',' ')) for i,r in enumerate(roles)]
    legend_ax.legend(handles=patches,ncol=3,loc='center',frameon=False,fontsize=6.8,columnspacing=1,handlelength=1.1,labelspacing=.55)
    fig.subplots_adjust(left=.03,right=.99,top=.97,bottom=.015);fig.savefig(output/'figure4-example-masks.png');plt.close(fig)
    (output/'FIGURE_PROVENANCE.json').write_text(json.dumps(dict(examples=[r['case'] for r in selected],selection='Median S09 per-image IoU within each of BCSS, TIGER and PUMA; no outcome-based best-image selection',opacity=.42,scale_bars_um=100,source_reports=[str(results/'CONTROLLED_FINAL.json'),str(results/'test-v1/RESULTS.json')]),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.results,a.output)


if __name__=='__main__':main()
