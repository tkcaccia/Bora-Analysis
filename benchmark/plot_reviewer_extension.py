"""Plot observed ROI-level deltas without treating ROIs as independent patients."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/mnt/sata_ssd/Bora-benchmark/results')


def main() -> None:
    lung=[r for r in csv.DictReader((ROOT/'bora-wsss-validation-2026-10-16'/'reviewer_cohort'/'roi_results.csv').open())
          if r['sample_id']!='validation--25']
    breast=list(csv.DictReader((ROOT/'bora-breast-extension-2026-10-16'/'reviewer_extension.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for j,metric in enumerate(('macro','boundary')):
        ax=axes[j]
        for i,tile in enumerate((56,90,216)):
            rows=[r for r in lung if int(r['tile'])==tile]
            if metric=='macro':
                values=[float(r['bora_macro_iou'])-float(r['baseline_macro_iou']) for r in rows]
            else:
                values=[float(r['bora_pair_f1_8'])-float(r['baseline_pair_f1_8']) for r in rows
                        if r['bora_pair_f1_8'] and r['baseline_pair_f1_8']]
            rng=np.random.default_rng(20261016+i)
            ax.scatter(np.full(len(values),i)+rng.uniform(-.16,.16,len(values)),values,
                       s=18,alpha=.55,color='#386c91')
            ax.plot(i,np.median(values),'D',color='#b32f2f',markersize=7)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(range(3),['56','90','216'])
        ax.set_xlabel('Source tile width (pixels)')
        ax.set_ylabel('Bora minus tile baseline')
        ax.set_title('Macro IoU' if metric=='macro' else 'Class pair boundary F1 at 8 pixels')
    fig.suptitle('WSSS4LUAD validation ROIs excluding the pilot ROI; points are ROIs, diamonds are medians',fontsize=10)
    out=ROOT/'bora-wsss-validation-2026-10-16'/'reviewer_cohort'/'delta_distribution.png'
    fig.savefig(out,dpi=200)
    plt.close(fig)
    print(out)

if __name__=='__main__':main()
