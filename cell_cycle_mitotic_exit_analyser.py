import logging
import re, os, sys
import json
import statistics, math
import traceback
import requests
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

# Import R libraries
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

utils = importr('utils')
base = importr('base')
stats = importr('stats')
utils.chooseCRANmirror(ind=1)
utils.install_packages("BiocManager")
biocManager = importr("BiocManager")
biocManager.install('limma')
limma = importr('limma')

logger = logging.getLogger(__name__)

import utilities_basicReader
from static_mapping import time_points, data_files, data_files_datakeys, time_points_mapping, replicates_timepoints

with open("./data/" + data_files["cached_index_protein_names"]) as outfile:
    index_protein_names = json.load(outfile)

####-------------------------------------------------------------####
# General Functions
def calcColumnMedian(data_source, total_abundance, phospho = False):
    """
    Calculates the median per column for each replicate.
    """
    logger.info("Calculate per Column Median for " + data_source)

    rep_abundance_timepoint = {}
    rep_median = {}

    rep_sample_time_map = time_points_mapping[data_source]

    for accession in total_abundance:
        if phospho == False:
            protein_abundance = total_abundance[accession]
            for data_key in data_files_datakeys[data_source]:
                if data_key in protein_abundance:
                    if data_key not in rep_abundance_timepoint:
                            rep_abundance_timepoint[data_key] = []
                    rep_abundance_timepoint[data_key].append(protein_abundance[data_key])

            for rep_sample_name in rep_abundance_timepoint:
                timepoint = [timepoint for timepoint, sample_name in rep_sample_time_map.items() if sample_name == rep_sample_name][0]
                rep_median[timepoint] = statistics.median(rep_abundance_timepoint[rep_sample_name])
        else:
            for phospho_site in total_abundance[accession]:
                for modificantion in total_abundance[accession][phospho_site]['peptide_abundances']:
                    peptide_abundance = total_abundance[accession][phospho_site]['peptide_abundances'][modificantion]["abundance"]

                    for timepoint in time_points_mapping[data_source]:
                        if timepoint not in rep_abundance_timepoint:
                            rep_abundance_timepoint[timepoint] = []
                        for replicate in peptide_abundance:
                            if timepoint in peptide_abundance[replicate]:
                                rep_abundance_timepoint[timepoint].append(peptide_abundance[replicate][timepoint])

            for rep_sample_name in rep_abundance_timepoint:
                rep_median[rep_sample_name] = statistics.median(rep_abundance_timepoint[rep_sample_name])
    
    return rep_median

def getProteinInfo(uniprot_accession):
    """
    Fetches protein and gene name from the UniProt REST API for a given accession.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.json"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve data from UniProt. Status code: {response.status_code}")

    data = response.json()

    protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
    gene_name = (
        data.get("genes", [{}])[0].get("geneName", {}).get("value", "Unknown")
        if data.get("genes") else "Unknown"
    )

    return gene_name, protein_name

def addProteinAnnotations(uniprot_accession):
    protein_info = {}

    if uniprot_accession in index_protein_names:
        for annotation in index_protein_names[uniprot_accession]:
            if annotation in ['protein_name', 'gene_name','phospho']:
                continue
            elif annotation in ['basic_localisation', 'localisation_keyword']:
                basic_localisation = index_protein_names[uniprot_accession]["basic_localisation"]
                localisation_keyword = index_protein_names[uniprot_accession]["localisation_keyword"]
                protein_info["localisation_info"] = {"basic_localisation": basic_localisation,
                        "localisation_keyword": localisation_keyword}
            else:
                protein_info[annotation] = index_protein_names[uniprot_accession][annotation]    
        
    return protein_info

def renameTimepoints(data_source, abundance):
    """
    Renames sample names to the corresponding Time Point.
    """
    abundance = abundance.copy()
    timepoint_abundances = {}
    rep_sample_time_map = time_points_mapping[data_source]

    for sample_time_point in data_files_datakeys[data_source]:
        if sample_time_point in abundance:
            timepoint = [timepoint for timepoint, sample_name in rep_sample_time_map.items() if sample_name == sample_time_point][0]
            timepoint_abundances[timepoint] = abundance[sample_time_point]

    return timepoint_abundances

def calculateAverageRepAbundance(input_protein_data):
    """
    Calculates the average abundance for each timepoint for one protein between the three replicates.
    And returns a dictionary that stores the new abundances and their respective timepoints.

    Input: abundances = {
        "abundance_rep_1" : {"Palbo arrest_R1":56, "DMA arrest_R1":67, "DMA release_R1":45, "Serum starvation arrest_R1":56, "Serum starvation release_R1":12},
        "abundance_rep_2" : {"Palbo arrest_R2":78, "DMA arrest_R2":56, "DMA release_R2":34, "Serum starvation arrest_R2":12, "Serum starvation release_R2":56},
        "abundance_rep_3" : {"Palbo arrest_R3":89, "DMA arrest_R3":45, "DMA release_R3":34, "Serum starvation arrest_R3":34, "Serum starvation release_R3":87}
        }  
    Output: {'Palbo arrest': 74.333, 'DMA arrest': 56.0, 'DMA release': 37.667, 'Serum starvation arrest': 34.0, 'Serum starvation release': 51.667}
    """   
    timepoints = ["Palbo arrest", "DMA arrest", "DMA release", "Serum starvation arrest", "Serum starvation release"]

    average_abundance = {}
    average_abundance_reps = {}

    for timepoint in timepoints:
        average_abundance_reps[timepoint] = {}
        for replicate in input_protein_data:
            rep_number = replicate.split("_",2)[2]
            rep_timepoint = timepoint + "_R" + rep_number
            if rep_timepoint in input_protein_data[replicate]:
                rep_timepoint_value = input_protein_data[replicate][rep_timepoint]
                average_abundance_reps[timepoint][rep_timepoint] = rep_timepoint_value

    for timepoint in average_abundance_reps:
        if len(average_abundance_reps[timepoint]) == 0:
            continue
        else:
            average_abundance[timepoint] = round(sum(list(average_abundance_reps[timepoint].values()))/len(average_abundance_reps[timepoint]),3)

    return average_abundance

def findAbundance(abundance):
    """
    Takes the raw abundances from the input file, renames the timepoints and splits the abundances 
    in 3 replicates. 
    """
    abundance_reps = {}

    abundances = renameTimepoints("Mitotic_Exit_Proteome", abundance)

    for replicate in replicates_timepoints:
        abundance_reps[replicate] = {}
        replicate_abundance = {}
        for timepoint in replicates_timepoints[replicate]:
            if timepoint in abundances:
                replicate_abundance[timepoint] = abundances[timepoint]
        abundance_reps[replicate] = replicate_abundance

    return abundance_reps

def getPhosphoCCD(mitotic_exit_dataset):
    """
    Defines and adds the CCD for each phosphorylation site
    """
    group = 'DMA_release-DMA_arrest'

    for accession in mitotic_exit_dataset:
        if len(mitotic_exit_dataset[accession]['phosphorylation_abundances']) != 0:
            for site in mitotic_exit_dataset[accession]['phosphorylation_abundances']:
                if site.find("-") == -1:
                    if 'limma' in mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]:
                        if group in mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['limma']: 
                            # Oscillating Site
                            if abs(mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['limma'][group]['logFC']) >= 1 and mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['limma'][group]['adj_P_Val'] <= 0.01:
                                # Oscilllating Protein
                                if len(mitotic_exit_dataset[accession]['protein_abundances']['raw']) != 0 :
                                    if abs(mitotic_exit_dataset[accession]['protein_abundances']['limma'][group]['logFC']) >= abs(np.log2(0.7)) and mitotic_exit_dataset[accession]['protein_abundances']['limma'][group]['adj_P_Val'] <= 0.01:
                                        mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]["Cell Cycle Dependency"] = "CCD PhosphoSite - CCD Protein"
                                    else:
                                        mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]["Cell Cycle Dependency"] = "CCD PhosphoSite"
                                else:
                                    mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]["Cell Cycle Dependency"] = "CCD PhosphoSite - No Protein info"
                            else:
                                mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]["Cell Cycle Dependency"] = "Rest"
                else:
                    mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]["Cell Cycle Dependency"] = "-"
    return mitotic_exit_dataset

####-------------------------------------------------------------####
# Normalisation Functions
def firstLevelNormalisation(rep_median_col, abundance):
    """
    First level/sample normalisation the raw abundances of each protein. 
    Normalise raw abundances by diving with column median.
    """
    normalised_abundance = {}
    for time_point in abundance:
        normalised_abundance[time_point] = abundance[time_point] / rep_median_col[time_point]

    return normalised_abundance

def calclog2PalboNormalisation(abundances):
    """
    Log2 transforms the abundance data and normalises each replicate on Palbo Arrest abundance.
    """
    normalised_abundance = {}

    for palbo_rep in ["Palbo arrest_R1","Palbo arrest_R2","Palbo arrest_R3"]:
        if palbo_rep in list(abundances.keys()):
            palbo = abundances[palbo_rep]
            for timepoint, abundance in abundances.items():
                normalised_abundance[timepoint] = round(math.log2(abundance/palbo), 4)

    return normalised_abundance

def calcMean100Normalisation(abundances):
    """
    For the scaling: Divide by the mean of all samples row-wise (i.e. per protein)
    and multiply by 100 so that row mean = 100.
    """
    normalised_abundace = {}

    # Find the mean of all samples per protein
    abundance_values = []
    for replicate in abundances:
        for timepoint in abundances[replicate]:
            abundance_values.append(abundances[replicate][timepoint])
    abundance_mean = sum(abundance_values)/len(abundance_values)

    # Scale
    for replicate in abundances:
        if replicate not in normalised_abundace:
            normalised_abundace[replicate] = {}
        for timepoint in abundances[replicate]:
            normalised_abundace[replicate][timepoint] = (abundances[replicate][timepoint]/abundance_mean) * 100

    return normalised_abundace

def normaliseData(abundance, zero_min=True):
    """
    Normalise the abundance values and store them in the abuncance_normalised dictionary
    min-max method
    """
    norm_abundance = {}
    abundance_data = []

    for k,v in abundance.items():
        if k.find("Palbo") == -1:
            abundance_data.append(v)

    min_value = min(abundance_data)
    if zero_min == True:
        min_value = 0
    max_value = max(abundance_data)

    for time_point in abundance:
        if time_point.find("Palbo") == -1:
            abundance_value = abundance[time_point]
            try:
                abundance_normalised = (abundance_value - min_value) / ( 
                    max_value - min_value
                )
            except:
                abundance_normalised = 0.5

            norm_abundance[time_point] = round(abundance_normalised, 4)

    return norm_abundance

####-------------------------------------------------------------####
# Regression
def calciScore(abundance_phospho, abundance_protein):
    """
    log2(release/arrest)phospho - log2(release/arrest)protein
    """
    i_score = {}

    groups_list = {
        "DMA_arrest-DMA_release": ['DMA arrest', 'DMA release'],
        "Serum_starvation_arrest-Serum_starvation_release":["Serum starvation arrest", "Serum starvation release"],
        "DMA_Release-Serum_starvation_release":["DMA release", "Serum starvation release"],
        "DMA_Release-Palbo_Arrest":["DMA release", "Palbo arrest"],
        "Serum_starvation_release-PalboArrest":["Serum starvation release", "Palbo arrest"],
    }

    for group in groups_list:
        timepoints = groups_list[group]
        if timepoints[0] in abundance_phospho and timepoints[0] in abundance_protein and timepoints[1] in abundance_phospho and  timepoints[1] in abundance_protein:
            phospho_ratio = round(math.log2(abundance_phospho[timepoints[1]] / abundance_phospho[timepoints[0]]), 4)
            protein_ratio = round(math.log2(abundance_protein[timepoints[1]] / abundance_protein[timepoints[0]]), 4)
            i_score[group] = phospho_ratio - protein_ratio

    return i_score

def calStableRegressionColumn(mitotic_exit_phospho, mitotic_exit_dataset):
    """
    Mitotic Exit Phospho DMA Arrest/Release Regression on the Stable Sites
    Normalise the phospho abundance for DMA Arrest/Release on the log2 sum stable sites abundance per column.
    Calculates and adds all the Regressed Phospho Abundance for each phosphosite.
    Phospho = Dependent = Y
    Stable Sites = Independent = X
    Linear Model => y = ax + b
    Residuals = Y - Y_predict 
    """
    phospho_stable_data = {}
    phospho_score_distribution = []
    phospho_data = {}

    # Define stable phosphorylation sites on Time Course Dataset
    with open ("TimeCourse_Full_info.json") as json_file:
        time_course = json.loads(json_file.read())

    # Stable Site Distribution
    # We trim the lower and upper bounds of the stable sites distribution to end up with a smaller and more well define "stable sites" set
    for accession in time_course:
        if len(time_course[accession]['phosphorylation_abundances']) != 0:
            for site in time_course[accession]['phosphorylation_abundances']:
                # Only consider certain sites
                if site.find("-") == -1:
                    # Stable Site
                    if time_course[accession]['phosphorylation_abundances'][site]['metrics']['0-max']['standard_deviation'] <= 0.05 and time_course[accession]['phosphorylation_abundances'][site]['metrics']['log2_mean']['ANOVA']['q_value'] > 0.01:
                        if accession in mitotic_exit_phospho and site in mitotic_exit_phospho[accession]:
                            row = []
                            for sample in ["DMA arrest_R", "DMA release_R"]:
                                for replicate in ['1','2','3']:
                                    row.append(mitotic_exit_phospho[accession][site]['position_abundances']['normalised']['log2_sum']['abundance_rep_' + replicate][sample + replicate])

                            phospho_score_distribution.append(sum(row[:3])/sum(row[3:]))

    phospho_score_distribution.sort()
    phospho_score_distribution_upper = phospho_score_distribution[-int(len(phospho_score_distribution)/10)]
    phospho_score_distribution_lower = phospho_score_distribution[int(len(phospho_score_distribution)/10)]

    for accession in mitotic_exit_phospho:
        for site in mitotic_exit_phospho[accession]:
            # Only consider certain sites
            if site.find("-") == -1:
                try:
                    # Stable Sites
                    if accession in time_course and site in time_course[accession]['phosphorylation_abundances'] and time_course[accession]['phosphorylation_abundances'][site]['metrics']['0-max']['standard_deviation'] <= 0.05 and time_course[accession]['phosphorylation_abundances'][site]['metrics']['log2_mean']['ANOVA']['q_value'] > 0.01:
                        row = []
                        
                        for sample in ["DMA arrest_R", "DMA release_R"]:
                            for replicate in ['1','2','3']:
                                row.append(mitotic_exit_phospho[accession][site]['position_abundances']['normalised']['log2_sum']["abundance_rep_" + replicate][sample + replicate])

                        if sum(row[:3])/sum(row[3:]) < phospho_score_distribution_lower or sum(row[:3])/sum(row[3:]) > phospho_score_distribution_upper:
                            pass
                        else:
                            phospho_stable_data[accession + ' ' + str(site)] = row
                    else:
                        row = []
                        for sample in ["DMA arrest_R", "DMA release_R"]:
                            for replicate in ['1','2','3']:
                                row.append(mitotic_exit_phospho[accession][site]['position_abundances']['normalised']['log2_sum']["abundance_rep_" + replicate][sample + replicate])
                            
                        phospho_data[accession + ' ' + str(site)] = row
                except:
                        logger.error(accession + ' ' + str(site))

    phospho_data_df = pd.DataFrame.from_dict(phospho_data, orient='index', columns=["DMA arrest_R1", "DMA arrest_R2", "DMA arrest_R3", "DMA release_R1", "DMA release_R2", "DMA release_R3"])
    phospho_stable_data_df = pd.DataFrame.from_dict(phospho_stable_data, orient='index', columns=[ "DMA arrest_R1", "DMA arrest_R2", "DMA arrest_R3", "DMA release_R1", "DMA release_R2","DMA release_R3"])

    for replicate in ['1','2','3']:
        arrest = "DMA arrest_R" + replicate
        release = "DMA release_R"  + replicate
        
        # We create the linear model on the Stable Sites 
        lin = LinearRegression()
        lin.fit(phospho_stable_data_df[[arrest]], phospho_stable_data_df[release])
        # Stable Corrected
        phospho_stable_data_df[arrest] = phospho_stable_data_df[arrest]*lin.coef_ + lin.intercept_ 
        # Rest of all the sites - All Corrected
        phospho_data_df[arrest] = phospho_data_df[arrest]*lin.coef_ + lin.intercept_ 

    # Add the new col_regression to self.mitotic_exit 
    for dataset in [phospho_data_df, phospho_stable_data_df]:
        for i, row in dataset.iterrows():
            accession = i.split(" ")[0]
            site = i.split(" ")[1]
            abundance_dic = row.to_dict()
            corrected_abundance = {}
            for replicate in ['1','2','3']:
                corrected_abundance["abundance_rep_" + replicate] = {}
                for timepoint in abundance_dic:
                    if replicate in timepoint:
                        corrected_abundance["abundance_rep_" + replicate][timepoint] = abundance_dic[timepoint]

            mitotic_exit_dataset[accession]["phosphorylation_abundances"][site]["position_abundances"]["normalised"]["col_regression"] = corrected_abundance

    return mitotic_exit_dataset

####-------------------------------------------------------------####
# Statistical Functions
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    logger.info("Calculate Benjamini-Hochberg p-value correction")
    p = np.asarray(p, dtype=float)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))

    return q[by_orig]

def getLIMMAstats(mitotic_exit_dataset, phospho = False, force = False):
    """
    Performs LIMMA analysis on the normalised palbo ratio data by comparing the goups in the group list
    to determine the significance of the differentially expressed proteins from quantitative mitotic exit
    proteomics dataset.
    In short, fold changes and p-values were calculated for every protein on the log2(palbo ratio) 
    by fitting a linear model using the lmFit function. The p-values were smoothed using empirical Bayes
    with the function eBayes. The resulting p-values were corrected using the Benjamini-Hochberg p-value 
    correction for multiple hypothesis testing.

    Input: The mitotic_exit dictionary
    Output: The mitotic_exit dictionary with the added limma section in "protein_abundances"
    """
    logger.info("limma Analysis")

    groups_list = [
        {"DMA_release":["DMA release_R1","DMA release_R2","DMA release_R3"], "DMA_arrest": ['DMA arrest_R1', 'DMA arrest_R2', 'DMA arrest_R3']},
        {"Serum_starvation_release":["Serum starvation release_R1","Serum starvation release_R2","Serum starvation release_R3"], "Serum_starvation_arrest":["Serum starvation arrest_R1","Serum starvation arrest_R2", "Serum starvation arrest_R3"]},
        {"Serum_starvation_arrest":["Serum starvation arrest_R1","Serum starvation arrest_R2", "Serum starvation arrest_R3"], "Palbo_Arrest":["Palbo arrest_R1","Palbo arrest_R2","Palbo arrest_R3"]},
        {"DMA_Release":["DMA release_R1","DMA release_R2","DMA release_R3"], "Serum_starvation_release":["Serum starvation release_R1","Serum starvation release_R2","Serum starvation release_R3"]},
        {"DMA_Release":["DMA release_R1","DMA release_R2","DMA release_R3"], "Palbo_Arrest":["Palbo arrest_R1","Palbo arrest_R2","Palbo arrest_R3"]},
        {"Serum_starvation_release":["Serum starvation release_R1","Serum starvation release_R2","Serum starvation release_R3"], "Palbo_Arrest":["Palbo arrest_R1","Palbo arrest_R2","Palbo arrest_R3"]},
    ]
    # Create a dataframe with the desired data
    replicate_names = ['abundance_rep_1','abundance_rep_2','abundance_rep_3']
    for groups in groups_list:  
        design_list = [] 
        data_column_list = [] 
        cc_data_list = []
        group_name = ""
        # Name of the Comparison
        for group in groups:
            if group_name == "":
                group_name = group + "-"
            else:
                group_name += group

            for subgroup in groups[group]:
                design_list.append([subgroup,group])
                data_column_list.append(subgroup)                
        
        if phospho == False:
            normalisation_method = 'log2_sum'
            for accession in list(mitotic_exit_dataset.keys()):
                if len(mitotic_exit_dataset[accession]['protein_abundances']['raw']) != 0:
                    cc_data_list_row = [accession]
                    for group in groups:
                        for timepoint in groups[group]:
                            for replicate_name in replicate_names:
                                if replicate_name in mitotic_exit_dataset[accession]["protein_abundances"]["normalised"][normalisation_method]:
                                    if timepoint in mitotic_exit_dataset[accession]["protein_abundances"]["normalised"][normalisation_method][replicate_name]:
                                        cc_data_list_row.append(mitotic_exit_dataset[accession]["protein_abundances"]["normalised"][normalisation_method][replicate_name][timepoint])
                    cc_data_list.append(cc_data_list_row)
        # Phospho = True        
        else:
            for accession in list(mitotic_exit_dataset.keys()):
                for site in mitotic_exit_dataset[accession]['phosphorylation_abundances']:
                    phospho_key = accession + "_" + site
                    cc_data_list_row = [phospho_key]

                    for group in groups:
                        for timepoint in groups[group]:
                            if timepoint.find('DMA') != -1:
                                normalisation_method = 'col_regression' 
                            else:
                                normalisation_method = 'log2_sum' 

                            for replicate_name in replicate_names:
                                if replicate_name in mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']["normalised"][normalisation_method]:
                                    if timepoint in mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']["normalised"][normalisation_method][replicate_name]:
                                        cc_data_list_row.append(mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']["normalised"][normalisation_method][replicate_name][timepoint])
                    cc_data_list.append(cc_data_list_row)

        data_df = pd.DataFrame(cc_data_list,columns = ['Name'] + data_column_list) 
        data_df = data_df.set_index('Name') 
        design_df = pd.DataFrame(design_list,columns = ['Sample Identifier','TimePoint'])

        # Convert data and design pandas dataframes to R dataframes
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data = ro.conversion.py2rpy(data_df)
            r_design = ro.conversion.py2rpy(design_df)
            genes = ro.StrVector([ str(index) for index in data_df.index.tolist()])

        # Create a model matrix using design's TimePoint column using the R formula "~0 + f" to get all the unique factors as columns
        f = base.factor(r_design.rx2('TimePoint'), levels=base.unique(r_design.rx2('TimePoint')))
        form = Formula('~0 + f')
        form.environment['f'] = f
        r_design = stats.model_matrix(form)
        r_design.colnames = base.levels(f)
        # Fitting the linear model
        fit = limma.lmFit(r_data, r_design)
        # Make a contrasts matrix with the 1st and the last unique values
        contrast_matrix = limma.makeContrasts(f"{r_design.colnames[0]}-{r_design.colnames[-1]}", levels=r_design)
        # Fit the contrasts matrix to the lmFit data & calculate the bayesian fit
        fit2 = limma.contrasts_fit(fit, contrast_matrix)
        fit2 = limma.eBayes(fit2)
        # topTreat the bayesian fit using the contrasts and add the genelist
        r_output = limma.topTreat(fit2, coef=1, genelist=genes, number=np.inf)
        # Append in mitotic_exit_dataset dictionary
        p_value = list(r_output.rx2('P.Value'))
        p_value_adj = list(r_output.rx2('adj.P.Val'))
        logFC = list(r_output.rx2('logFC'))
        names = list(r_output.rx2('ID'))

        for i in range(0,len(names)):
            if math.isnan(logFC[i]) == True:
                    logFC[i] = 0
                    p_value[i] = 0
                    p_value_adj[i] = 0

            if phospho == True:
                accession = names[i].split("_")[0]
                if names[i].find("[") != -1:
                    site = names[i].split("_")[1].split("[")[1].strip("]")
                else:
                    site = names[i].split("_")[1]
                mitotic_exit_dic = mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]

            else:
                accession = names[i]
                mitotic_exit_dic = mitotic_exit_dataset[accession]['protein_abundances']

            if 'limma' not in mitotic_exit_dic:
                mitotic_exit_dic['limma'] = {}

            mitotic_exit_dic['limma'][group_name] = {
                    'logFC': logFC[i], 
                    'p_value': p_value[i], 
                    'adj_P_Val': p_value_adj[i]
                }
            
    return mitotic_exit_dataset

def tTeststats(mitotic_exit_dataset, phospho = False):
    """
    Performs Two-Sample T-Test betwen Release and Arrest
    """
    # create input df
    col_regress = {}

    if phospho == True:
        for accession in mitotic_exit_dataset:
            if len(mitotic_exit_dataset[accession]['phosphorylation_abundances']) != 0:
                for site in mitotic_exit_dataset[accession]['phosphorylation_abundances']:
                    if len(mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']['normalised']['col_regression']) != 0:
                        site_key = accession + "_" + site
                        col_regress[site_key] = {}
                        for rep in mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']['normalised']['col_regression']:
                            for timepoint in mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']['normalised']['col_regression'][rep]:
                                if timepoint in ['DMA release_R1', 'DMA release_R2', 'DMA release_R3']:
                                    col_regress[site_key][timepoint] = mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']['normalised']['col_regression'][rep][timepoint]
                                elif timepoint in ['DMA arrest_R1', 'DMA arrest_R2','DMA arrest_R3']:
                                    col_regress[site_key][timepoint] = mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]['position_abundances']['normalised']['col_regression'][rep][timepoint]
    else:
        for accession in mitotic_exit_dataset:
            if len(mitotic_exit_dataset[accession]['protein_abundances']['raw']) != 0:
                col_regress[accession] = {}
                for rep in mitotic_exit_dataset[accession]['protein_abundances']['normalised']['log2_sum']:
                    if rep != 'abundance_average':
                        for timepoint in mitotic_exit_dataset[accession]['protein_abundances']['normalised']['log2_sum'][rep]:
                            if timepoint in ['DMA release_R1', 'DMA release_R2', 'DMA release_R3']:
                                col_regress[accession][timepoint] = mitotic_exit_dataset[accession]['protein_abundances']['normalised']['log2_sum'][rep][timepoint]
                            elif timepoint in ['DMA arrest_R1', 'DMA arrest_R2','DMA arrest_R3']:
                                col_regress[accession][timepoint] = mitotic_exit_dataset[accession]['protein_abundances']['normalised']['log2_sum'][rep][timepoint]

    col_regress_df = pd.DataFrame(col_regress).T

    # Perform Two-Sample T-Test
    for index, row in col_regress_df.iterrows():
        arrest = [row['DMA arrest_R1'], row['DMA arrest_R2'], row['DMA arrest_R3']]
        release = [row['DMA release_R1'], row['DMA release_R2'], row['DMA release_R3']]

        col_regress_df.loc[[index],['fold_change']] = np.mean(release) - np.mean(arrest)

        t_stat, p_value = ttest_ind(release, arrest)
        col_regress_df.loc[[index],['t_statistic']] = t_stat
        col_regress_df.loc[[index],['p_value']] = p_value

    # Correct the p_values
    col_regress_df['q_value'] = p_adjust_bh(col_regress_df['p_value'])

    # Append in mitotic_exit_dataset dictionary        
    t_statistic = list(col_regress_df['t_statistic'])
    p_value = list(col_regress_df['p_value'])
    p_value_adj = list(col_regress_df['q_value'])
    logFC = list(col_regress_df['fold_change'])
    names = list(col_regress_df.index)

    for i in range(0,len(names)):
        if math.isnan(logFC[i]) == True:
                logFC[i] = 0
                p_value[i] = 0
                p_value_adj[i] = 0
                t_statistic[i] = 0 

        if phospho == True:
            accession = names[i].split("_")[0]
            if names[i].find("[") != -1:
                site = names[i].split("_")[1].split("[")[1].strip("]")
            else:
                site = names[i].split("_")[1]
            mitotic_exit_dic = mitotic_exit_dataset[accession]['phosphorylation_abundances'][site]
        else:
            accession = names[i]
            mitotic_exit_dic = mitotic_exit_dataset[accession]['protein_abundances']

        if 't_test' not in mitotic_exit_dic:
            mitotic_exit_dic['t_test'] = {}

        mitotic_exit_dic['t_test']['DMA_release-DMA_arrest'] = {
                't_statistic': t_statistic[i],
                'fold_change': logFC[i], 
                'p_value': p_value[i], 
                'q_value': p_value_adj[i]
            }
    return mitotic_exit_dataset

####-------------------------------------------------------------####
# PhosphoProteomics Parsing Functions
def findPositionInMasterProtein(data_point):
    """
    Finds the start and stop positions of the phosphopeptide in the master protein.
    Input: "P16402 [35-47]; P10412 [34-46]; P16403 [34-46]"
    Output: {'P16402': {'start': '35', 'end': '47'}, 'P10412': {'start': '34', 'end': '46'}, 'P16403': {'start': '34', 'end': '46'}}
    """
    position_in_master_protein = data_point
    protein_position_info = {}

    positions = position_in_master_protein.split(";")        
    regexes = {"accession": r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"}
    uniprot_accession_pattern = re.compile(regexes["accession"])
    accessions = [match.group() for match in uniprot_accession_pattern.finditer(position_in_master_protein)]
    
    for accession in accessions:
        protein_position_info[accession] = []

        for position in positions:
            position_info = position.strip(" ").split(" ")
            if len(accessions) != 1:
                if position_info[0] != accession:
                    continue
            for index, item in enumerate(position_info):
                position_info[index] = item.strip("[").strip("]")
                if len(position_info) > 1:
                    position_ranges = position_info[1].split("-")
                else:
                    position_ranges = position_info[0].split("-")

            protein_position_info[accession].append({
                "start": position_ranges[0],
                "end": position_ranges[1],
            })

    return protein_position_info

def findSite(sites):
    """
    Gets the phosphorylation sites and returns a dictionary with all the information.
    Possible inputs: P0DJD0 [1207-1239]; P49792 [2198-2230] | P27816 1xPhospho [S280] | A0A0B4J2F2 1xPhospho [S575] |
    O15085 [1452-1473] | Q9NYF8 2xPhospho [S290 S/Y] | Q9P206 2xPhospho [S971 S979] | P24928 [1909-1922]; [1923-1936]
    """
    likelihood = None
    final_sites = {}
    for site in sites:
        site = site.split("; ")
        for p_site in site:
            p_site = p_site.split(" ")
            for pp_site in p_site:
                pp_site = pp_site.strip("[").strip("]")
                temp_site_dic = {}
                temp_site_dic[pp_site] = {}
                # [S575]
                if pp_site.find("-") == -1 and pp_site.find("/") == -1:
                    pp_site_full = pp_site.split("(")
                    pp_site = pp_site_full[0]
                    if len(pp_site_full) > 1:
                        likelihood = pp_site_full[1].strip(")")
                    aa = pp_site[0]
                    position = pp_site[1::]
                    if position == "":
                        position = pp_site
                    uniprot_position = {"start": "-", "end": "-"}
                    final_sites[pp_site] = {
                        "aa": aa,
                        "position": position,
                        "phosphosite": pp_site,
                        "uniprot_position": uniprot_position,
                        "likelihood": likelihood,
                    }
                # [S/Y]
                elif pp_site.find("/") != -1:
                    aa = pp_site
                    position = pp_site
                    uniprot_position = {"start": "-", "end": "-"}
                    final_sites[pp_site] = {
                        "aa": aa,
                        "position": position,
                        "phosphosite": pp_site,
                        "uniprot_position": uniprot_position,
                        "likelihood": likelihood,
                    }
                # [1452-1473]
                elif pp_site.find("-") != -1:
                    aa = "-"
                    position = pp_site
                    site_range = pp_site.split("-")
                    uniprot_position = {
                        "start": site_range[0],
                        "end": site_range[1],
                    }
                    final_sites[pp_site] = {
                        "aa": aa,
                        "position": position,
                        "phosphosite": pp_site,
                        "uniprot_position": uniprot_position,
                        "likelihood": likelihood,
                    }

    return final_sites

def findModificationPosition(data_point):
    """
    Creates and returns a dictionary that contain information about the positions in the master protein
    and the modification events.
    Example input : ["P20700 1xPhospho [S393]; Q03252 1xPhospho [S407]", "A0A0B4J2F2 1xPhospho [S575]", "O15085 [1452-1473]"
                                    "Q9NYF8 2xPhospho [S290 S/Y]", "Q9P206 2xPhospho [S971 S979]"], 
                                    "Q9P2E9 1xPhospho [T225(100)]; 1xPhospho [T245(100)]; 1xPhospho [T255(100)]"

    Example output: modification_info = {'Q9P206': {'S971': {'event': '2xPhospho', 'aa': 'S', 'Position': '971',
            'likelihood': None, 'phosphosite': 'S971', 'Uniprot_Position': {'start': '-', 'end': '-'}},
            'S979': {'event': '2xPhospho', 'aa': 'S', 'Position': '979', 'likelihood': None, 'phosphosite': 'S979', ...}}}}
    """
    modification = data_point["Modifications in Master Proteins"]
    positions = findPositionInMasterProtein(data_point["Positions in Master Proteins"])
    
    modification_info = {}
    regexes = {
        "accession": r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}",
        "ptm": r"\dxPhospho",
        "site": r"\[\S+\s*\S*\]",
    }
    modifications = modification.split("; ")
    accession = ""
    position_count = 0

    # modifications = "Q07820 [137-176]" or empty
    if len(modifications) == 0 or modifications == ['']:
        if data_point["Positions in Master Proteins"] != '':
            modification = data_point["Positions in Master Proteins"]
            modifications = [modification]
            
    for event in modifications:
        try:
            # Uniprot_accession
            uniprot_accession_pattern = re.compile(regexes["accession"])
            accessions = [
                match.group() for match in uniprot_accession_pattern.finditer(event)
            ]
            if len(accessions) > 0:
                accession = accessions[0]
            if accession not in modification_info:
                modification_info[accession] = {}
            # Site
            site_pattern = re.compile(regexes["site"])
            sites = [match.group() for match in site_pattern.finditer(event)]
            if len(sites) == 0:
                # input Q9Y608 2xPhospho [S328(100); S340(99.2)]
                # After split: ['Q9Y608 2xPhospho [S328(100)', 'S340(99.2)]']
                site_pattern = re.compile(r"\[\S+\s*\S*")
                sites = [match.group() for match in site_pattern.finditer(event)]
                # 'S340(99.2)]'
                if len(sites) == 0:
                    sites = [event]

            sites = findSite(sites)
            
            for p_site, info in sites.items():
                # Add the peptide uniprot position on site name of the no specific phosphosites
                # e.g. T/S --> T/S[123-130]
                if p_site.find("/") != -1 or len(p_site) == 1:
                    range_name = (
                        "["
                        + str(positions[accession][position_count]["start"])
                        + "-"
                        + str(positions[accession][position_count]["end"])
                        + "]"
                    )
                    p_site = p_site + range_name

                    if (
                        info["phosphosite"].find("/") != -1
                        or len(info["phosphosite"]) == 1
                    ):
                        info["phosphosite"] = info["phosphosite"] + range_name
                        info["position"] = range_name.strip("[").strip("]")

                # Add info in modification_info dictionary
                if p_site not in modification_info[accession]:
                    modification_info[accession][p_site] = {}
                modification_info[accession][p_site]["phosphosite"] = p_site
                modification_info[accession][p_site]["aa"] = info["aa"]
                modification_info[accession][p_site]["position"] = info["position"]
                modification_info[accession][p_site]["likelihood"] = info[
                    "likelihood"
                ]
                if len(modifications) == 1:
                    modification_info[accession][p_site][
                            "uniprot_position"
                        ] = positions[accession][position_count]
                else:
                    # P46013 2xPhospho [T1531(100); S1533(100)]; 2xPhospho [T2380(100); S2382(100)]
                    if len(positions[accession]) > 1:
                        # when we have a positon
                        if info["position"].find("-") == -1:
                            if (float(info["position"]) >= float(positions[accession][position_count]['start']) 
                                and float(info["position"]) <= float(positions[accession][position_count]['end'])): 

                                modification_info[accession][p_site][
                                        "uniprot_position"
                                    ] = positions[accession][position_count]
                            else:
                                position_count += 1
                                modification_info[accession][p_site][
                                        "uniprot_position"
                                    ] = positions[accession][position_count]
                        else:
                            modification_info[accession][p_site][
                                    "uniprot_position"
                                ] = positions[accession][position_count]
                            position_count += 1
                    else:
                        modification_info[accession][p_site][
                                    "uniprot_position"
                                ] = positions[accession][0]
                    
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(modification)
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)

    return modification_info

def combinePhosphoAbundances(abundance_dic):
    """
    Combine the abundances for a phosphosite from different modifications.
    """
    for accession in abundance_dic:
        for site in abundance_dic[accession]:
            raw = abundance_dic[accession][site]['position_abundances']['raw']
            for modification in abundance_dic[accession][site]['peptide_abundances']:
                for replicate in abundance_dic[accession][site]['peptide_abundances'][modification]["abundance"]:
                    if replicate not in raw:
                        raw[replicate] = {}
                    abundance = abundance_dic[accession][site]['peptide_abundances'][modification]["abundance"][replicate]
                    for k, v in abundance.items():
                        cur_ab = 0
                        if k in raw[replicate]:
                            cur_ab += raw[replicate][k]
                        raw[replicate][k] = (cur_ab + abundance[k])

    return abundance_dic

####-------------------------------------------------------------####
# Kinases Prediction
def getProteinSeq(uniprot_accession):
    """
    Fetches the protein sequence for the given UniProt accession by querying UniProt directly.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.fasta"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.splitlines()
            sequence = ''.join(lines[1:]) 
            return sequence
        else:
            return f"Error: Unable to fetch data for accession {uniprot_accession}. HTTP Status: {response.status_code}"
    except requests.RequestException as e:
        return f"Error: Unable to connect to UniProt. {str(e)}"

def getPeptideSequence(uniprot_accession, phosphosite):
    """
    Creates a peptide sequence which is a substring of the original protein sequence.
    +5/-5
    amino acids from the phosphorylation site.
    """
    sequence = getProteinSeq(uniprot_accession)

    # Start counting from 1
    position = int(phosphosite[1::]) -1
    sequence_len = len(sequence)
    
    start = 0
    end = sequence_len-1

    # start of protein sequence
    if position < 5:
        start = 0
        end = position + 6
    # middle of protein sequence
    elif sequence_len > position + 6:
        start = position - 5
        end = position + 6
    # end of protein sequence
    elif  sequence_len < position + 6 or sequence_len == position + 6:
        start = position - 5
        end = sequence_len

    peptide_sequence = sequence[start:end]

    return peptide_sequence

def getPeptideAligned(uniprot_accession, phosphosite):
    """
    Checks if the phosphosite is centered in the peptide sequence and it aligns it if not.
    """
    aa = phosphosite[0]
    position = int(phosphosite[1::]) -1
    site =  str(position+1)

    if 'phospho' in index_protein_names[uniprot_accession]:
        if site in index_protein_names[uniprot_accession]['phospho']:
            if 'peptide_seq' in index_protein_names[uniprot_accession]['phospho'][site]:
                peptide_seq = index_protein_names[uniprot_accession]['phospho'][site]['peptide_seq']
            else:
                peptide_seq = getPeptideSequence(uniprot_accession, phosphosite)

    phospho_alignment = ""

    #Discard invalid inputs
    if len(peptide_seq) == 0:           
        peptide_seq = index_protein_names[uniprot_accession]['phospho'][site]['Peptide']
        phospho_alignment = peptide_seq[5:16]
    # Middle of the protein sequence
    elif len(peptide_seq) == 11:
        if aa == peptide_seq[5]:
            phospho_alignment = peptide_seq
        else:
            # Site not in the middle of seq
            peptide_seq_new = index_protein_names[uniprot_accession]['phospho'][site]['Peptide']
            phospho_alignment = peptide_seq_new[5:16]
            if  uniprot_accession == "P62861" and phosphosite == 'S5':
                phospho_alignment = "-KVHGSLARAG"                

    # Missing Positions
    elif len(peptide_seq) < 11:
        # beginning of the protein sequence
        if position < 5:
            increase = 5 - position
            peptide_seq = ("-" * increase) + peptide_seq
            phospho_alignment = peptide_seq
        else:
            # end of the protein sequence
            increase = 11 - len(peptide_seq)
            peptide_seq = peptide_seq + ("-" * increase)
            if aa != peptide_seq[5]:
                peptide_seq = peptide_seq[:5] + aa + peptide_seq[5 + 1:]
            phospho_alignment = peptide_seq

    return phospho_alignment

def getConsensusKinasePred(uniprot_accession, phosphosite):
    """
    Predicts the kinase most likely to phosphorylate a phosphorylation site 
    based on the consensus approach.
    """
    phospho_kinases_class = {}
    peptide_seq = getPeptideAligned(uniprot_accession, phosphosite)

    motifs = [
        {"motif": "plk", "pattern": ".{3}[DNE].[ST][FGAVLIMW][GAVLIPFMW].{3}"},
        {"motif": "cdk", "pattern": ".{3}..[ST]P.[RK]."},
        {"motif": "aurora", "pattern": ".{3}R.[ST][GAVLIFMW].{4}"},
        {"motif": "stp_consensus", "pattern": ".{5}[ST]P.{4}"},
        {"motif": "stq_consensus", "pattern": ".{5}[ST]Q.{4}"},
        {"motif": "krxst_consensus", "pattern": ".{3}[KR].[ST].{5}"},
        {"motif": "krxxst_consensus", "pattern": ".{2}[KR].{2}[ST].{5}"},
    ]

    total_not_matched = 0
    matches = 0
    motif_matches = []
    for m in motifs:
        motif = m["motif"]
        pattern = m["pattern"]
        res = re.match(pattern, peptide_seq)
        if res:
            matches+=1
            motif_matches.append(motif)
        phospho_kinases_class = {"accession":uniprot_accession,"site":phosphosite, "peptide_seq": peptide_seq, "kinase_motif_match": motif_matches}
    if matches == 0:
        total_not_matched+=1
        phospho_kinases_class = {"accession":uniprot_accession, "site":phosphosite, "peptide_seq": peptide_seq, "kinase_motif_match": ["-"]}

    return phospho_kinases_class

####-------------------------------------------------------------####
# Parse Data Functions

def findContaminants():
    """
    Create a uniprot accession list of contaminant proteins
    """
    theos_contaminants_list = ["P04745","P13645", "P35527", "P04264", "P35908", "Q15323", "Q14532", "O76011", "Q92764", "O76013", "O76014", "O76015", "O76009", "Q14525", "Q14533", "Q9NSB4", "P78385", "Q9NSB2", "P78386", "O43790", "O77727", "P02534", "P25690", "P02539", "P15241", "P25691", "P02444", "P02445", "P02443", "P02441", "Q02958", "P02438", "P02439", "P02440", "P08131", "P26372", "O82803","P15252",]
    contaminants = []

    data_source = "TimeCourse_Proteomics_rep_1"
   
    data_points = utilities_basicReader.readTableFile(
        "./data/" + data_files[data_source],
        byColumn=False,
        stripQuotes=True,
    )

    for data_point in data_points:
        uniprot_accession = data_points[data_point]["Accession"]
        contaminant = data_points[data_point]["Contaminant"]
        if (
            uniprot_accession == ""
            or uniprot_accession == "Accession"
            or contaminant == "Yes"
            or contaminant == "TRUE"
            or uniprot_accession in theos_contaminants_list
        ):
            if uniprot_accession not in contaminants:
                contaminants.append(uniprot_accession)

    return contaminants

def MitoticExitProteomicsParser(data_source):
    """
    Reads and parses the proteomics data from the Mitotic Exit replicates experiment for every protein in the dataset.
    """
    logger.info("Loading abundance data from Mitotic Exit Proteomics")
    
    cellCycleDataset = {}

    data_points = utilities_basicReader.readTableFile(
       "./data/" + data_files[data_source],
        byColumn=False,
        stripQuotes=True)
    
    contaminants = findContaminants()

    for data_point in data_points:
        try:
            uniprot_accession = data_points[data_point]["Accession"]
            protein_description = data_points[data_point]["Description"]
            master_protein = data_points[data_point]["Master"]
            
            if protein_description.find("GN=") != -1:
                # gene_name = re.findall("GN=\S*", protein_description)[0].split('=')[1]
                gene_name = re.findall(r"GN=\S*", protein_description)[0].split('=')[1]
            else: 
                gene_name = ""
            
            # uniprot_accession == "Q8WZ71" because it doesn't have Palbo abundance so we can't normalise it
            if (uniprot_accession == "" or uniprot_accession in contaminants or uniprot_accession == "Q8WZ71"):
                continue
            if (master_protein != "IsMasterProtein"):
                continue

            if uniprot_accession not in cellCycleDataset:
                cellCycleDataset[uniprot_accession] = {"gene_name": gene_name, "protein_description": protein_description}

            for datakey in data_files_datakeys[
                data_source]:
                value = data_points[data_point][datakey]
                if value == "":
                    continue
                cellCycleDataset[uniprot_accession][datakey] = float(value)

            if len(cellCycleDataset[uniprot_accession]) == 2:
                del cellCycleDataset[uniprot_accession]
            else:
                cellCycleDataset[uniprot_accession]["confidence"] = {
                    "exp_q-value_combined": data_points[data_point]["Exp q-value Combined"],
                    "peptides": data_points[data_point]["Number of Peptides"],
                    "PSMs": data_points[data_point]["Number of PSMs"],
                    "unique_peptides": data_points[data_point]["Number of Unique Peptides"],
                    "protein_FDR_confidence_combined": data_points[data_point]["Protein FDR Confidence Combined"],
                    "coverage": data_points[data_point]["Coverage in Percent"],
                    "aas": data_points[data_point]["Number of AAs"],
                    "MW": data_points[data_point]["MW in kDa"],
                }

        except Exception as e:
            print(uniprot_accession)
            print(e)

    logger.info("Added cross-reference - " + data_source)
    print("-----> {}".format(data_source),len(cellCycleDataset),)

    return cellCycleDataset

def MitoticExitProteomeInfo(data_source):

    mitotic_exit_dataset = {}

    mitotic_exit_proteome = MitoticExitProteomicsParser(data_source)
    col_medians_proteome = calcColumnMedian("Mitotic_Exit_Proteome", mitotic_exit_proteome)
    col_sum_proteome = {'Palbo arrest_R1': 21525334.699999996, 'DMA arrest_R1': 18530184.09999999, 'DMA release_R1': 21250729.899999883, 'Serum starvation arrest_R1': 19749752.899999995, 'Serum starvation release_R1': 19141061.69999996, 'Palbo arrest_R2': 20629306.400000114, 'DMA arrest_R2': 20678593.99999991, 'DMA release_R2': 19131276.099999942, 'Serum starvation arrest_R2': 19897247.20000014, 'Serum starvation release_R2': 20345632.8999999, 'Palbo arrest_R3': 19646135.899999883, 'DMA arrest_R3': 20153276.599999942, 'DMA release_R3': 18446561.599999912, 'Serum starvation arrest_R3': 17881599.10000001, 'Serum starvation release_R3': 18693852.399999972}

    for uniprot_accession in mitotic_exit_proteome:
        try:
            if uniprot_accession in index_protein_names:
                protein_name = index_protein_names[uniprot_accession]["protein_name"]
            else:
                protein_name = mitotic_exit_proteome[uniprot_accession]["gene_name"]
                
            mitotic_exit_dataset[uniprot_accession] = {
                "gene_name": mitotic_exit_proteome[uniprot_accession]["gene_name"],
                "protein_name": protein_name,
                "protein_abundances": {
                    "raw": {},
                    "normalised": {"median": {}, "log2_median": {}, "sum":{}, "log2_sum":{}, "log2_palbo": {}, "mean100_scale":{}, "0-max":{}},
                },
                "phosphorylation_abundances": {},
                "confidence": mitotic_exit_proteome[uniprot_accession]["confidence"],
                "protein_info": addProteinAnnotations(uniprot_accession),
            }
            
            abundances = renameTimepoints("Mitotic_Exit_Proteome", mitotic_exit_proteome[uniprot_accession])
            
            replicates_timepoints = {
                "abundance_rep_1" : ["Palbo arrest_R1", "DMA arrest_R1", "DMA release_R1", "Serum starvation arrest_R1", "Serum starvation release_R1"],
                "abundance_rep_2" : ["Palbo arrest_R2", "DMA arrest_R2", "DMA release_R2", "Serum starvation arrest_R2", "Serum starvation release_R2"],
                "abundance_rep_3" : ["Palbo arrest_R3", "DMA arrest_R3", "DMA release_R3", "Serum starvation arrest_R3", "Serum starvation release_R3"]
                }     
                       
            for replicate in replicates_timepoints:
                replicate_abundance = {}
                for timepoint in replicates_timepoints[replicate]:
                    if timepoint in abundances:
                        replicate_abundance[timepoint] = abundances[timepoint]

                # Raw Abundances
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["raw"][replicate] = replicate_abundance
                # First Level Normalisation - Normalise raw abundances by diving with column median
                median_norm_rep_abundance = firstLevelNormalisation(col_medians_proteome, replicate_abundance)
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["median"][replicate] = median_norm_rep_abundance
                # First Level Normalisation - Normalise raw abundances by diving with column sum
                sum_norm_rep_abundance = firstLevelNormalisation(col_sum_proteome, replicate_abundance)
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["sum"][replicate] = sum_norm_rep_abundance
                # 0-max
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["0-max"][replicate] = normaliseData(median_norm_rep_abundance, zero_min=True)
            
                # Log2 - Median Normalisation
                log2_normed_abundance = {}
                for time_point in median_norm_rep_abundance:
                    log2_normed_abundance[time_point] =  round(math.log2(median_norm_rep_abundance[time_point]), 4)
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_median"][replicate] = log2_normed_abundance
                # Log2 - Sum Normalisation
                log2_normed_abundance = {}
                for time_point in sum_norm_rep_abundance:
                    log2_normed_abundance[time_point] =  round(math.log2(sum_norm_rep_abundance[time_point]), 4)
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_sum"][replicate] = log2_normed_abundance

                # Log2 - Palbo Normalisation
                palbo_norm_rep_abundance = calclog2PalboNormalisation(median_norm_rep_abundance)
                mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_palbo"][replicate] = palbo_norm_rep_abundance
            # Mean100_scale Normalisation
            mean_100_scale_rep_abundance = calcMean100Normalisation(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["median"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["mean100_scale"] = mean_100_scale_rep_abundance
            
            # Averages
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["raw"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["raw"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["median"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["median"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_median"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_median"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["sum"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["sum"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_sum"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_sum"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_palbo"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["log2_palbo"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["mean100_scale"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["mean100_scale"])
            mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["0-max"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_dataset[uniprot_accession]["protein_abundances"]["normalised"]["0-max"])

        except Exception as e:
            print(uniprot_accession)
            print(traceback.format_exc())

    return mitotic_exit_dataset
    
def MitoticExitPhosphoParser(data_source):
    """
    Reads and parses the phosphoproteomics data from the Mitotic Exit replicates experiment for every protein in the dataset.
    """
    logger.info("Loading abundance data from Mitotic Exit PhosphoProteomics")
    cellCycleDataset = {}

    data_points = utilities_basicReader.readTableFile(
        "./data/" + data_files[data_source],
        byColumn=False,
        stripQuotes=True,
    )
    contaminants = findContaminants()

    for data_point in data_points:
        try:
            modification_info = findModificationPosition(data_points[data_point])
            for uniprot_accession_key in modification_info:
                if uniprot_accession_key == "" or uniprot_accession_key in contaminants:
                    continue
                
                # Abundances
                phospho_abundance = {}
                for datakey in data_files_datakeys["Mitotic_Exit_Proteome"]:
                    value = data_points[data_point][datakey]
                    if value == "":
                        continue
                    phospho_abundance[datakey] = float(value)
                # Phosphopeptides with empty abundances
                if len(phospho_abundance) == 0:
                    continue

                if uniprot_accession_key not in cellCycleDataset:
                    cellCycleDataset[uniprot_accession_key] = {}

                for phosphosite in modification_info[uniprot_accession_key]:
                    mod_key = modification_info[uniprot_accession_key][phosphosite][
                        "position"
                    ]
                    modification = data_points[data_point]["Modifications"]
                    phosphosite_info = modification_info[uniprot_accession_key][phosphosite]

                    if mod_key not in cellCycleDataset[uniprot_accession_key]:
                        cellCycleDataset[uniprot_accession_key][mod_key] = {
                            "phosphorylation_site": phosphosite_info["phosphosite"],
                            "peptide_abundances": {},
                            "position_abundances": {
                                "raw": {},
                                "normalised": {"median": {},"log2_median": {}, "sum": {},"log2_sum": {}, "log2_palbo": {}, "mean100_scale":{}, "col_regression": {}, "0-max":{}}
                            },
                            "confidence": {},
                        }

                    cellCycleDataset[uniprot_accession_key][mod_key]["confidence"]  = {
                        "Sequence": data_points[data_point]["Annotated Sequence"],
                        "no_Protein_groups": data_points[data_point]["# Protein Groups"],
                        "no_proteins": data_points[data_point]["# Proteins"],
                        "no_isoforms": data_points[data_point]["# Isoforms"],
                        "PSMs": data_points[data_point]["# PSMs"],
                        "no_missed_cleavages": data_points[data_point]["# Missed Cleavages"],
                        "Percolator_PEP": data_points[data_point]["Percolator PEP (by Search Engine): Sequest HT"],
                        "Percolator_q_value": data_points[data_point]["Percolator q-Value (by Search Engine): Sequest HT"],
                        "XCorr": data_points[data_point]["XCorr (by Search Engine): Sequest HT"],
                        "MH+": data_points[data_point]["Theo. MH+ [Da]"],
                    }

                    if modification not in cellCycleDataset[uniprot_accession_key][mod_key]["peptide_abundances"]:
                        cellCycleDataset[uniprot_accession_key][mod_key]["peptide_abundances"][modification] = {
                            "abundance": findAbundance(phospho_abundance),
                            "uniprot_position": phosphosite_info["uniprot_position"],
                            "aa": phosphosite_info["aa"],
                            "phosphosite": phosphosite_info["phosphosite"],
                            "likelihood": phosphosite_info["likelihood"],
                            "modification": modification,
                        }      
                
        except Exception as e:
            print("this is a problem")
            print(uniprot_accession_key)
            print(mod_key)
            print(modification_info)
            print(traceback.format_exc())


    logger.info("Added cross-reference - " + data_source)
    print("-----> {}".format(data_source),len(cellCycleDataset))

    return cellCycleDataset

def MitoticExitPhosphoInfo(data_source):
    col_sum_phospho = {'Palbo arrest_R1': 7876571.999999955, 'DMA arrest_R1': 16379401.400000028, 'DMA release_R1': 6325827.099999989, 'Serum starvation arrest_R1': 5476054.000000004, 'Serum starvation release_R1': 7077658.3, 'Palbo arrest_R2': 7542343.89999998, 'DMA arrest_R2': 17150636.999999892, 'DMA release_R2': 6133582.500000004, 'Serum starvation arrest_R2': 5952412.099999985, 'Serum starvation release_R2': 5892527.200000035, 'Palbo arrest_R3': 7503985.500000016, 'DMA arrest_R3': 18279908.59999989, 'DMA release_R3': 5960757.999999989, 'Serum starvation arrest_R3': 5855794.700000015, 'Serum starvation release_R3': 9645558.499999937}
    
    mitotic_exit_phospho = MitoticExitPhosphoParser(data_source)
    mitotic_exit_phospho = combinePhosphoAbundances(mitotic_exit_phospho)

    col_medians_phospho = calcColumnMedian("Mitotic_Exit_Proteome", mitotic_exit_phospho, phospho = True)

    for uniprot_accession in mitotic_exit_phospho:
        for phosphosite in mitotic_exit_phospho[uniprot_accession]:
            # Normalisation Methods
            for replicate in mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']['raw']:
                replicate_abundance = mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']['raw'][replicate]
                # First Level Normalisation - Normalise raw abundances by diving with column median
                median_norm_rep_abundance = firstLevelNormalisation(col_medians_phospho, replicate_abundance)
                mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["median"][replicate] = median_norm_rep_abundance
                # First Level Normalisation - Normalise raw abundances by diving with column sum
                sum_norm_rep_abundance = firstLevelNormalisation(col_sum_phospho, replicate_abundance)
                mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["sum"][replicate] = sum_norm_rep_abundance
                # 0-max
                mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["0-max"][replicate] = normaliseData(median_norm_rep_abundance, zero_min=True)
                # Log2 - Median Normalisation
                log2_normed_abundance = {}
                for time_point in median_norm_rep_abundance:
                    log2_normed_abundance[time_point] =  round(math.log2(median_norm_rep_abundance[time_point]), 4)
                mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_median"][replicate] = log2_normed_abundance
                # Log2 - Sum Normalisation
                log2_normed_abundance = {}
                for time_point in sum_norm_rep_abundance:
                    log2_normed_abundance[time_point] =  round(math.log2(sum_norm_rep_abundance[time_point]), 4)
                mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_sum"][replicate] = log2_normed_abundance
                # Log2 - Palbo Normalisation
                palbo_norm_rep_abundance = calclog2PalboNormalisation(median_norm_rep_abundance)
                mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_palbo"][replicate] = palbo_norm_rep_abundance
            # Mean100_scale Normalisation
            mean_100_scale_rep_abundance = calcMean100Normalisation(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["median"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["mean100_scale"] = mean_100_scale_rep_abundance

            # Averages
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["raw"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["raw"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["median"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["median"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_median"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_median"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["sum"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["sum"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_sum"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_sum"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_palbo"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["log2_palbo"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["mean100_scale"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["mean100_scale"])
            mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["0-max"]["abundance_average"] = calculateAverageRepAbundance(mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["0-max"])
            
            # Kinase Consensus Prediction
            if phosphosite.find("-") == -1:
                # Add PepTools Phospho annotations
                if 'phospho' in index_protein_names[uniprot_accession]:
                    if phosphosite in index_protein_names[uniprot_accession]['phospho']:
                        mitotic_exit_phospho[uniprot_accession][phosphosite]['PepTools_annotations'] = index_protein_names[uniprot_accession]['phospho'][phosphosite]
                mitotic_exit_phospho[uniprot_accession][phosphosite]['kinase_prediction'] = {}
                phospho_kinases_class = getConsensusKinasePred(uniprot_accession, mitotic_exit_phospho[uniprot_accession][phosphosite]['phosphorylation_site'])
                mitotic_exit_phospho[uniprot_accession][phosphosite]['kinase_prediction']['peptide_seq'] = phospho_kinases_class['peptide_seq']
                mitotic_exit_phospho[uniprot_accession][phosphosite]['kinase_prediction']['consenus_motif_match'] = phospho_kinases_class['kinase_motif_match']
                

    return mitotic_exit_phospho
 
####-------------------------------------------------------------####
def createMitoticExitSet():
    """
    Creates a json file that stores all the info for the Mitotic Exit Proteome/PhosphoProteme experiment.
    It combines and stores the 3 proteomics and the 3 phosphoproteomics replicates along with the normalised and imputed values of the protein/phosphoprotein abundances.
    """
    logger.info("Combine Proteomics and Phosphoproteomics info")
    
    mitotic_exit_dataset = MitoticExitProteomeInfo("Mitotic_Exit_Proteome")
    mitotic_exit_phospho = MitoticExitPhosphoInfo("Mitotic_Exit_Phospho")

    for uniprot_accession in mitotic_exit_phospho:
        # We have both protein and phospho information -iScore
        if uniprot_accession in mitotic_exit_dataset:
            if len(mitotic_exit_dataset[uniprot_accession]['protein_abundances']['raw']) != 0:
                for phosphosite in mitotic_exit_phospho[uniprot_accession]:
                    abundance_protein = mitotic_exit_dataset[uniprot_accession]['protein_abundances']["normalised"]["median"]["abundance_average"]
                    abundance_phospho = mitotic_exit_phospho[uniprot_accession][phosphosite]['position_abundances']["normalised"]["median"]["abundance_average"]
                    mitotic_exit_phospho[uniprot_accession][phosphosite]['iScore'] = calciScore(abundance_phospho, abundance_protein)

            # Add info in the dictionary
            mitotic_exit_dataset[uniprot_accession]['phosphorylation_abundances'] = mitotic_exit_phospho[uniprot_accession]
        else:
            if uniprot_accession in index_protein_names:
                protein_name = index_protein_names[uniprot_accession]["protein_name"]
                gene_name = index_protein_names[uniprot_accession]["gene_name"]
            else:
                gene_name, protein_name = getProteinInfo(uniprot_accession)

            mitotic_exit_dataset[uniprot_accession] = {
                "gene_name": gene_name,
                "protein_name": protein_name,
                "protein_abundances": {
                    "raw": {},
                    "normalised": {"median": {}, "log2_palbo": {}, "mean100_scale":{}, "col_regression": {}},
                },
                "phosphorylation_abundances": mitotic_exit_phospho[uniprot_accession],
                "confidence": "",
                "protein_info": addProteinAnnotations(uniprot_accession)
            }

    # Column Regression
    calStableRegressionColumn(mitotic_exit_phospho, mitotic_exit_dataset)

    # LIMMA Differentially Expressed Proteins    
    getLIMMAstats(mitotic_exit_dataset, phospho = False, force = True)
    getLIMMAstats(mitotic_exit_dataset,phospho = True, force = True)
    tTeststats(mitotic_exit_dataset,phospho = False)
    tTeststats(mitotic_exit_dataset,phospho = True)

    # Phospho CCD
    getPhosphoCCD(mitotic_exit_dataset)

    with open('Mitotic_Exit_Full_Info.json', "w") as outfile:
        json.dump(mitotic_exit_dataset, outfile)

    return mitotic_exit_dataset
 
####-------------------------------------------------------------####

def main():
    createMitoticExitSet()

if __name__ == "__main__":
    main()