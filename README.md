

## Table of contents
* [Description](#Description)
* [Cell Cycle Database Web server](#CellCycle-DB-Web-server)
* [Usage](#usage)
* [Input files](#input-files)
* [Output files](#output-files)
* [License](#license)
* [Reference](#reference)

## Description
A proteomic and phosphoproteomic analysis of the human cell cycle in hTERT-RPE-1 cells using deep quantitative mass spectrometry by isobaric labelling. 
Through analysing non-transformed cells, and improving the temporal resolution and coverage of key cell cycle regulators, 
we present a dataset of cell cycle-dependent protein and phosphorylation site oscillation that offers a foundational reference for investigating cell cycle regulation.

<p align="center">
  <img src="./docs/img/cell_cycle_git.jpg" width="90%" height="100%" title="Cell Cycle Analysis">
</p>

## Cell Cycle Database Web server
The data produced and collected in this study is available in the Cell Cycle Database (CCdb) resource. The database holds the 
cell cycle abundance changes for proteins and phosphorylation sites for the Time Course, Mitotic Exit and Serum Starvation datasets, 
the respective statistical metrics and the cell cycle dependency status. 
The data is additionally enriched with external information on protein, phosphorylation site, mRNA abundance and cell cycle dependency from publicly available datasets. 
Protein level information on stability and degrons is also provided. For phosphorylation sites, annotations on the accessibility, 
domain and motif overlap, and presence in other publicly available phosphoproteomic datasets are annotated. 
The CCdb links cell cycle-dependent abundance dynamics to functional changes illuminating the biological role of oscillating proteins 
and phosphorylation events and is available at https://slim.icr.ac.uk/cell_cycle/.
A detailed description of CCdb usage and output is provided on the help page (https://slim.icr.ac.uk/cell_cycle/blog?blog_id=ccdb_help)

## Usage

### Requirements
* Docker

### Build docker image
To build a Docker image from the provided Dockerfile run the following steps:
```shell
git clone git@github.com:ifigenia-t/cell-cycle-analysis.git
cd cell-cycle-analysis
docker build -t python-r-latest .
```
### Run the Time Course analyser
```sh
docker run -it --rm -v $(pwd):/app/cell python-r-latest cell_cycle_timecourse_analyser.py
```
### Run the Mitotic Exit analyser
```sh
docker run -it --rm -v $(pwd):/app/cell python-r-latest cell_cycle_mitotic_exit_analyser.py
```

## Input files
.tdt files that contain the raw Mass Spec abundances changes for proteins and phosphorylation sites for the Time Course, Mitotic Exit and Serum Starvation datasets, 

## Output files

The pipeline will produce 2 files:
- `TimeCourse_Full_info.json`: .JSON file containing the results of the Time Course experiment analysis.
- `Mitotic_Exit_Full_Info.json`: .JSON file containing the results of the Mitotic Exit and Serum Starvation experiments analysis.

The combined data includes cell cycle abundance changes for proteins and phosphorylation sites across the Time Course, Mitotic Exit, and Serum Starvation datasets, 
along with their respective statistical metrics and cell cycle dependency status. 
Both the Proteome and Phosphoproteome analyses are included in each file.

#### Output file example format

```
{
   "Q09666":	
      gene_name:  "AHNAK",
      protein_name:  "Neuroblast differentiati…ssociated protein AHNAK",
      protein_abundances:  {
          "raw": {…},
          "normalised": {…},
          "imputed": {…}
                        },
      phosphorylation_abundances:  {…},
      confidence:  {…},
      protein_info:  {…},
      metrics:  {…}
.......
}
```

## License
This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.

## Reference
If you find the pipeline or the data useful in your research, we ask that you cite our paper:

*"Camilla Rega, Ifigenia Tsitsa, Theodoros I. Roumeliotis, Izabella Krystkowiak, Maria Portillo, Lu Yu, Julia Vorhauser, Jonathon Pines, Joerg Mansfeld, Jyoti Choudhary, Norman E. Davey **High resolution profiling of cell cycle-dependent protein and phosphorylation abundance changes in non-transformed cells**."*

 [DOI: 10.1101/2024.06.20.599917](https://doi.org/10.1101/2024.06.20.599917)
