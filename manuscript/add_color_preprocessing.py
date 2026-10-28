#!/usr/bin/env python3
"""Add the verified color-preprocessing pilot to an existing CMPB manuscript.

This narrow patch preserves figures, tables, captions and authors. Input and
output paths are supplied by the caller; no private benchmark path is stored.
"""
from pathlib import Path
import argparse,json,hashlib,re
from copy import deepcopy
from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph
from docx.table import _Row
from docx.shared import RGBColor

METHODS = (
 'An optional color-drift correction was evaluated on the pancreas WSI, with the cropped colorectal image and brain WSI used as technical controls. '
 'Bright, weakly colored glass connected to the slide edge was identified on a thumbnail and separated from stained tissue by a safety margin. '
 'Median red, green and blue (RGB) intensities were sampled in a 24 × 24 grid, and a robust linear surface was fitted separately to each channel in slide coordinates. '
 'At least 48 glass cells, coverage of 75% of each slide axis and eight cells in each spatial quadrant were required. '
 'A channel was corrected only when spatial variation of at least 1.5% was predicted, its median absolute prediction error on held-out spatial quadrants was reduced by at least 25% relative to a constant model, and gains within ±5% were sufficient. '
 'Slides with more than 5% clipped glass pixels were excluded from correction.'
)
METHODS_2 = (
 'Each accepted channel was multiplied by the ratio of the median glass intensity to the fitted local glass intensity. '
 'One smooth correction field was applied across the slide; tissue histograms were not used for fitting. '
 'The original image was retained, and a separate losslessly compressed image was exported with unchanged dimensions, pixel spacing and annotation coordinates. '
 'The export was checked against an independent implementation at 15 fixed windows and 12 native-resolution tissue tiles. '
 'Glass lightness was computed from decoded RGB values on the L* scale (0–100), assuming standard RGB (sRGB) encoding. '
 'Dark-field calibration was unavailable, so the fitted plane was treated as an empirical correction of decoded RGB values.'
)
RERUN_METHODS = (
 'For the corrected pancreas reanalysis, the same 6,925 tissue-containing tile locations were retained at approximately 0.5 µm/pixel. '
 'Virchow2 tile and spatial features were extracted anew. Tile vectors were standardized, 50 principal components were computed, and KODAMA was fitted with 100 repetitions and four processor cores. '
 'A graph with 15 neighbors was partitioned into four anonymous groups by Leiden clustering at resolution 0.095. '
 'Multiscale spatial tokens and group-response models were regenerated from the corrected image. '
 'All group responses were refined jointly over the complete slide using the frozen Bora 0.2.2 implementation with the same H&E-guided refinement settings. '
 'Components and enclosed holes smaller than 400 µm² were removed or filled globally before compact diagonal GeoJSON export. '
 'The same nine diagnostic windows were cropped from the saved full-slide mask; no window-specific refinement was performed.'
)

def insert_before(anchor,text,style='Normal'):
    e=OxmlElement('w:p');anchor._p.addprevious(e);p=Paragraph(e,anchor._parent);p.style=style;p.add_run(text);return p

def find(doc,prefix):
    found=[p for p in doc.paragraphs if p.text.startswith(prefix)]
    if len(found)!=1:raise ValueError(f'Expected one anchor {prefix!r}, got {len(found)}')
    return found[0]

def apply(source,output,qc_path,rerun_path=None):
    if output.exists():raise FileExistsError('Use a fresh manuscript output')
    qc=json.loads(qc_path.read_text());assert qc['status']=='COMPLETE' and qc['full_native_derivative_verified']
    p=qc['slides']['pancreas'];assert p['decision']=='APPLY' and p['channels']==['G','B']
    assert qc['slides']['brain']['decision']==qc['slides']['colon']['decision']=='SKIP'
    rerun=None
    if rerun_path:
        rerun=json.loads(rerun_path.read_text())
        assert rerun['status']=='COMPLETE' and rerun['all_features_regenerated_from_corrected_WSI'] and rerun['all_nine_crops_verified']
    doc=Document(source)
    doc.styles['Heading 3'].font.color.rgb=RGBColor(0,0,0)
    background=find(doc,'Background and Objective')
    background.text=background.text.replace('calibrated H&E images',
        'calibrated hematoxylin and eosin (H&E) images')
    abstract_methods=find(doc,'Methods Borders')
    abstract_methods.text=abstract_methods.text.replace(
        'hematoxylin and eosin (H&E) image','H&E image')
    if any(p.text=='Optional color drift correction' for p in doc.paragraphs):raise ValueError('Color preprocessing already described')
    anchor=find(doc,'Virchow2 evidence conversion')
    insert_before(anchor,'Optional color drift correction','Heading 2');insert_before(anchor,METHODS);insert_before(anchor,METHODS_2)
    if rerun:insert_before(anchor,RERUN_METHODS)
    anchor=find(doc,'Discussion')
    insert_before(anchor,'Color drift correction','Heading 2')
    text=(f'The glass-supported correction reduced the absolute upper-to-lower glass lightness difference in the pancreas WSI from {abs(p["before_glass_delta_L"]):.2f} to {abs(p["after_glass_delta_L"]):.2f} units. '
      'Green and blue passed the correction criteria; red remained unchanged. Colorectal and brain controls did not meet all criteria, so the pipeline retained their original pixels. '
      'The lossless export preserved every tested pixel under the independently implemented correction equation.')
    if rerun:text+=' The corrected-image reanalysis produced a complete four-group pancreas mask and diagonal GeoJSON; all nine diagnostic masks matched their full-slide crops exactly. These anonymous groups provide a descriptive result rather than a validated tissue classification.'
    else:text+=' These checks establish technical image consistency, without testing changes in segmentation or tissue classification.'
    insert_before(anchor,text)
    anchor=find(doc,'Native-coordinate polygons support')
    insert_before(anchor,'Color drift can influence both feature extraction and H&E-guided refinement. The glass-only model removes the spatial component supported by blank-slide pixels while leaving tissue composition out of the fit. It cannot resolve all staining or section-thickness variation, and improved glass uniformity does not establish better tissue discrimination. Microscopy shading methods such as BaSiC address related acquisition artifacts [23].')
    availability=find(doc,'The authors provide Bora 0.2.1')
    availability.text=availability.text.replace(
        'The authors provide Bora 0.2.1 on GitHub main at commit bfae24aba03b18f77133a7b9d695ad49f8edf22a (https://github.com/tkcaccia/Bora).',
        'The authors provide the archived Bora 0.2.1 source at commit bfae24aba03b18f77133a7b9d695ad49f8edf22a (https://github.com/tkcaccia/Bora). The current release provides the frozen whole-slide implementation as Bora 0.2.2 at commit def3b76f88c7666417c8e7938f9a86ed72568e08.')
    availability.add_run(' The optional color-drift stage is provided in Bora-Analysis at pipeline/wsi_color_preprocess.py, with commands and validation criteria in docs/WSI_COLOR_PREPROCESSING.md. Corrected images require new Virchow2 features and downstream fits; original-image caches are not reused.')
    data=find(doc,'The team staged four further Visium HD')
    data.add_run(' The color-preprocessing sources are the 10x pancreas release (https://www.10xgenomics.com/datasets/visium-hd-cytassist-11mm-human-pancreas) and brain cancer release (https://www.10xgenomics.com/datasets/visium-hd-cytassist-6p5mm-human-brain-cancer).')
    table=doc.tables[0]
    extra=[('10x pancreas WSI','Pancreas (healthy tissue)','One WSI from one donor in the sample release','One color-correction and whole-slide reanalysis case'),
           ('10x brain WSI','Brain (anaplastic oligodendroglioma, grade III)','One WSI from one donor in the sample release','One color-correction technical control')]
    templates=list(table.rows[-2:])
    for template,values in zip(templates,extra):
        element=deepcopy(template._tr);table._tbl.append(element);row=_Row(element,table)
        for cell,text in zip(row.cells,values):
            props=deepcopy(cell.paragraphs[0].runs[0]._r.rPr)
            cell.text=text
            if props is not None:cell.paragraphs[0].runs[0]._r.insert(0,props)
    refs=[p for p in doc.paragraphs if re.match(r'^\[\d+\]',p.text)];assert len(refs)==22
    e=OxmlElement('w:p');refs[-1]._p.addnext(e);ref=Paragraph(e,refs[-1]._parent);ref.style=refs[-1].style;ref.add_run('[23] Peng T, Thorn K, Schroeder T, et al. A BaSiC tool for background and shading correction of optical microscopy images. Nat Commun. 2017;8:14836. https://doi.org/10.1038/ncomms14836')
    # Preserve the existing first mention of TIFF in the classifier description;
    # no additional acronym is introduced by the preprocessing paragraphs.
    assert len(doc.tables)==6
    output.parent.mkdir(parents=True,exist_ok=True);doc.save(output)
    return dict(status='COMPLETE',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_qc_sha256=hashlib.sha256(qc_path.read_bytes()).hexdigest(),rerun_included=bool(rerun),new_reference=23,figures_and_tables_preserved=True)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--qc',type=Path,required=True);a.add_argument('--rerun-receipt',type=Path);s=a.parse_args();print(json.dumps(apply(s.input,s.output,s.qc,s.rerun_receipt),indent=2))
if __name__=='__main__':main()
