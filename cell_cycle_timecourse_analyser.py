import re
import sys,os
import logging
import json
import statistics, math
import requests
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import moment, f_oneway
from sklearn.metrics import r2_score
from sklearn import linear_model

import utilities_basicReader
from static_mapping import time_points, data_files, data_files_datakeys, time_points_mapping

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

with open("./data/" + data_files["cached_index_protein_names"]) as outfile:
    index_protein_names = json.load(outfile)

####-------------------------------------------------------------####
# General Functions

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

def getProteinLocalisation(uniprot_accession):
    """
    Fetches localisation information for a given UniProt accession by querying UniProt directly.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.json"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            localisation_data = {
                "basic_localisation": [],
                "localisation_keyword": []
            }
            
            if "comments" in data:
                for comment in data["comments"]:
                    if comment["type"] == "SUBCELLULAR_LOCATION":
                        locations = comment.get("subcellularLocations", [])
                        for location in locations:
                            if "location" in location:
                                term = location["location"].get("value", "")
                                if term:
                                    localisation_data["basic_localisation"].append(term)
                            
                            if "keywords" in location:
                                for keyword in location["keywords"]:
                                    keyword_value = keyword.get("value", "")
                                    if keyword_value:
                                        localisation_data["localisation_keyword"].append(keyword_value)
            
            return localisation_data
        else:
            return f"Error: Unable to fetch data for accession {uniprot_accession}. HTTP Status: {response.status_code}"
    except requests.RequestException as e:
        return f"Error: Unable to connect to UniProt. {str(e)}"

def addProteinAnnotations(combined_time_course_info, index_protein_names):
    for uniprot_accession in combined_time_course_info:
        combined_time_course_info[uniprot_accession]["protein_info"] = {}
        if uniprot_accession in index_protein_names:
            basic_localisation = index_protein_names[uniprot_accession]["basic_localisation"]
            localisation_keyword = index_protein_names[uniprot_accession]["localisation_keyword"]
        else:
            basic_localisation, localisation_keyword = getProteinLocalisation(uniprot_accession)

        combined_time_course_info[uniprot_accession]["protein_info"]["localisation_info"] = {"basic_localisation": basic_localisation,
                "localisation_keyword": localisation_keyword}

        if "halflife_mean" in index_protein_names[uniprot_accession]:
            combined_time_course_info[uniprot_accession]["protein_info"]["halflife_mean"] = index_protein_names[uniprot_accession]["halflife_mean"]     
            combined_time_course_info[uniprot_accession]["protein_info"]['halflife_std'] = index_protein_names[uniprot_accession]["halflife_std"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['halflife_min'] = index_protein_names[uniprot_accession]["halflife_min"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['halflife_count'] = index_protein_names[uniprot_accession]["halflife_count"]     
            combined_time_course_info[uniprot_accession]["protein_info"]['relative_abundance_8h_count'] = index_protein_names[uniprot_accession]["relative_abundance_8h_count"]     
            combined_time_course_info[uniprot_accession]["protein_info"]['relative_abundance_8h_mean'] = index_protein_names[uniprot_accession]["relative_abundance_8h_mean"]     
            combined_time_course_info[uniprot_accession]["protein_info"]['relative_abundance_8h_std'] = index_protein_names[uniprot_accession]["relative_abundance_8h_std"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['mean_gene_effect'] = index_protein_names[uniprot_accession]["mean_gene_effect"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['in_DRIVE_cancer_proteins'] = index_protein_names[uniprot_accession]["in_DRIVE_cancer_proteins"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['in_CGC_cancer_proteins'] = index_protein_names[uniprot_accession]["in_CGC_cancer_proteins"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['role_in_cancer'] = index_protein_names[uniprot_accession]["role_in_cancer"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['tier'] = index_protein_names[uniprot_accession][ "tier"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['cell_cycle'] = index_protein_names[uniprot_accession]["cell_cycle"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['mitotic_cell_cycle'] = index_protein_names[uniprot_accession]["mitotic_cell_cycle"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['kinetochore'] = index_protein_names[uniprot_accession][ "kinetochore"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['spindle'] = index_protein_names[uniprot_accession]["spindle"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['centriole'] = index_protein_names[uniprot_accession]["centriole"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['replication_fork'] = index_protein_names[uniprot_accession][ "replication fork"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['G0_to_G1_transition'] = index_protein_names[uniprot_accession]["G0_to_G1_transition"]      	
            combined_time_course_info[uniprot_accession]["protein_info"]['G1/S_transition'] = index_protein_names[uniprot_accession]["G1/S_transition"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['G2/M_transition'] = index_protein_names[uniprot_accession]["G2/M_transition"]      	
            combined_time_course_info[uniprot_accession]["protein_info"]['S_phase'] = index_protein_names[uniprot_accession]["S_phase"]      
            combined_time_course_info[uniprot_accession]["protein_info"]['transcription_factor'] = index_protein_names[uniprot_accession]["transcription_factor"]     
            combined_time_course_info[uniprot_accession]["protein_info"]['kinase_domain_containing']	 = index_protein_names[uniprot_accession]["kinase_domain_containing"]     
            combined_time_course_info[uniprot_accession]["protein_info"]["is_E3"] = index_protein_names[uniprot_accession]["is_E3"]
            combined_time_course_info[uniprot_accession]["protein_info"]["APC_complex"] = index_protein_names[uniprot_accession]["APC_complex"]
            combined_time_course_info[uniprot_accession]["protein_info"]["dna_replication_machinery"] = index_protein_names[uniprot_accession]["dna_replication_machinery"]
        
    return combined_time_course_info

def renameTimepoints(data_source, abundance):
    """
    Renames sample names to the corresponding time point.
    """
    abundance = abundance.copy()
    timepoint_abundances = {}
    rep_sample_time_map = time_points_mapping[data_source]

    for sample_time_point in data_files_datakeys[data_source]:
        if sample_time_point in abundance:
            timepoint = [timepoint for timepoint, sample_name in rep_sample_time_map.items() if sample_name == sample_time_point][0]
            timepoint_abundances[timepoint] = abundance[sample_time_point]

    return timepoint_abundances

def calculateAverageRepAbundance(input_protein_data, imputed=False, phospho=False, norm=False, phospho_oscillation=False):
    """
    Calculates the average abundance for each timepoint for one protein between the three phospho replicates.
    And returns a dictionary that stores the new abundances and their respective timepoints.
    """
    input_protein_data = input_protein_data.copy()
    abundance_average = {}

    abundance_rep_1 = {}
    abundance_rep_1 = {}

    if phospho_oscillation == True:
        if "protein_oscillation_abundance_rep_1" in input_protein_data:
            abundance_rep_1 = input_protein_data[
                "protein_oscillation_abundance_rep_1"]
            abundance_rep_2 = input_protein_data[
                "protein_oscillation_abundance_rep_2"]
    else:
        abundance_rep_1 = input_protein_data["abundance_rep_1"]
        abundance_rep_2 = input_protein_data["abundance_rep_2"]


    if imputed == False:
        for timepoint in time_points:
            if (timepoint in abundance_rep_1 and timepoint in abundance_rep_2):
                average_abundance = (
                    abundance_rep_1[timepoint]
                    + abundance_rep_1[timepoint]) / 2
                abundance_average[timepoint] = round(average_abundance, 4)
            elif timepoint in abundance_rep_1 and timepoint not in abundance_rep_2:
                abundance_average[timepoint] = round(abundance_rep_1[timepoint], 4)
            elif timepoint in abundance_rep_2 and timepoint not in abundance_rep_1:
                abundance_average[timepoint] = round(abundance_rep_2[timepoint], 4)
            # timepoint not in any replicate
            elif timepoint not in abundance_rep_1 and timepoint not in abundance_rep_2:
                continue


    if imputed == True:
        for timepoint in time_points:
            abundance_average[timepoint] = {}
            # timepoint present in all replicates
            if (timepoint in abundance_rep_1 and timepoint in abundance_rep_2):
                if (
                    abundance_rep_1[timepoint]["status"] == "imputed"
                    and abundance_rep_2[timepoint]["status"] == "imputed"
                ):
                    average_abundance = (
                        abundance_rep_1[timepoint]["value"]
                        + abundance_rep_2[timepoint]["value"]
                    ) / 2
                    abundance_average[timepoint] = {
                        "value": round(average_abundance, 4),
                        "status": "imputed",
                    }
                elif (
                    abundance_rep_1[timepoint]["status"] == "experimental"
                    and abundance_rep_2[timepoint]["status"] == "imputed"
                ):
                    average_abundance = (
                        abundance_rep_1[timepoint]["value"]
                        + abundance_rep_2[timepoint]["value"]
                    ) / 2
                    abundance_average[timepoint] = {
                        "value": round(average_abundance, 4),
                        "status": "experimental and imputed",
                    }
                else:
                    average_abundance = (
                        abundance_rep_1[timepoint]["value"]
                        + abundance_rep_2[timepoint]["value"]
                    ) / 2
                    abundance_average[timepoint] = {
                        "value": round(average_abundance, 4),
                        "status": "experimental",
                    }
            # timepoint present only in replicate 1
            elif (
                timepoint in abundance_rep_1
                and timepoint not in abundance_rep_2
            ):
                abundance_average[timepoint] = {
                    "value": round(abundance_rep_1[timepoint]["value"], 4),
                    "status": abundance_rep_1[timepoint]["status"],
                }
            # timepoint present only in replicate 2
            elif (
                timepoint in abundance_rep_2
                and timepoint not in abundance_rep_1
            ):
                average_abundance = (
                    abundance_rep_2[timepoint]["value"]
                ) 
                abundance_average[timepoint] = {
                    "value": round(average_abundance, 4),
                    "status": abundance_rep_2[timepoint]["status"],
                }

    return abundance_average

def findContaminants():
    """
    Create a uniprot accession list of contaminant proteins
    """
    contaminants = []
    for data_source in ["TimeCourse_Proteomics_rep_1", "TimeCourse_Proteomics_rep_2"]:
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
            ):
                if uniprot_accession not in contaminants:
                    contaminants.append(uniprot_accession)
    return contaminants

####-------------------------------------------------------------####
# Metrics Functions 

def polyfit(x, y, degree):
    """
    Calculate the r-squared for polynomial curve fitting.
    """
    r_squared = 0
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    # calculate r-squared
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssres = np.sum((y - yhat) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    r_squared = ssres / sstot

    return 1 - round(r_squared, 2)

def calcCurveFoldChange(norm_abundances, uniprot_accession):
    """
    Calculates the curve_fold_change and curve peaks for the three or two replicates normalised abundance for each protein.
    """

    timepoint_map = {
        "Palbo": 0,
        "Late G1_1": 1,
        "G1/S": 2,
        "S": 3,
        "S/G2": 4,
        "G2_2": 5,
        "G2/M_1": 6,
        "M/Early G1": 7,
    }
    curves_all = {}

    # if we have info for the protein in at least 2 replicates
    if len(norm_abundances) >= 3:
        reps = norm_abundances
        curves_all[uniprot_accession] = {}
        x = []
        for rep in reps:
            if rep == "abundance_average" or rep == "abundance_rep_1":
                continue
            for timepoint in norm_abundances[rep]:
                x.append(timepoint_map[timepoint])
        x.sort()
        y = []
        for timepoint in timepoint_map:
            for rep in reps:
                if rep == "abundance_average" or rep == "abundance_rep_1":
                    continue
                if timepoint in norm_abundances[rep]:
                    y.append(norm_abundances[rep][timepoint])

        p = np.poly1d(np.polyfit(x, y, 2))
        curve_abundances = p(x)

        # find the timepoint peak of the curve
        curve_index = x[list(curve_abundances).index(max(curve_abundances))]
        for time_point, index in timepoint_map.items():
            if index == curve_index:
                curve_peak = time_point

        # Calculate the fold change from the curve
        curve_fold_change = max(curve_abundances) / max(0.05, min(curve_abundances))

    return curve_fold_change, curve_peak

def calcResidualsR2All(norm_abundances):
    """
    Calculate the residuals and the R squared for all the abundances from all the replicates for a protein.
    """

    timepoint_map = {
        "Palbo": 0,
        "Late G1_1": 1,
        "G1/S": 2,
        "S": 3,
        "S/G2": 4,
        "G2_2": 5,
        "G2/M_1": 6,
        "M/Early G1": 7,
    }
    # if we have info for the protein in at least 2 replicates
    if len(norm_abundances) >= 3:
        reps = norm_abundances
        x = []
        for rep in reps:
            if rep == "abundance_average" or rep == "abundance_rep_1":
                continue
            for timepoint in norm_abundances[rep]:
                x.append(timepoint_map[timepoint])
        x.sort()
        y = []
        for timepoint in timepoint_map:
            for rep in reps:
                if rep == "abundance_average" or rep == "abundance_rep_1":
                    continue
                if timepoint in norm_abundances[rep]:
                    y.append(norm_abundances[rep][timepoint])

        p = np.poly1d(np.polyfit(x, y, 2))
        curve_abundances = p(x)
        residuals_all = np.polyfit(x, y, 2, full=True)[1][0]
        r_squared_all = round(r2_score(y, curve_abundances), 2)

        return residuals_all, r_squared_all

def calcAbundanceMetrics(norm_abundances, uniprot_accession):
    """
    Calculates the moments and peaks for the average for the three replicates normalised abundance for each protein.
    """
    metrics = {}
    norm_abundance_average = norm_abundances["abundance_average"]
    norm_abundance_average_list = list(norm_abundance_average.values())

    norm_method = '0-max'

    abundance = []
    for timepoint in time_points:
        for rep in norm_abundances:
            if rep != "abundance_average":
                if timepoint in norm_abundances[rep]:
                    abundance.append(norm_abundances[rep][timepoint])
    
    std = statistics.stdev(abundance)

    try:
        residuals_array = np.polyfit(
            range(0, len(norm_abundance_average)),
            norm_abundance_average_list,
            2,
            full=True,
        )[1]

        if len(residuals_array) == 0:
            # eg Q9HBL0 {'G2_2': 0.4496, 'G2/M_1': 0.7425, 'M/Early G1': 1.0}
            residuals = 5
        else:
            residuals = np.polyfit(
            range(0, len(norm_abundance_average)),
            norm_abundance_average_list,
            2,
            full=True,)[1][0]

        r_squared = polyfit(
            range(0, len(norm_abundance_average)), norm_abundance_average_list, 2
        )
        max_fold_change = max(norm_abundance_average.values()) - min(
            norm_abundance_average.values()
        )
        metrics = {
            "variance": moment(norm_abundance_average_list, moment=2),
            "skewness": moment(norm_abundance_average_list, moment=3),
            "kurtosis": moment(norm_abundance_average_list, moment=4),
            "peak": max(norm_abundance_average, key=norm_abundance_average.get),
            "max_fold_change": max_fold_change,
            "residuals": residuals,
            "R_squared": r_squared,
        }
        # if we have info for the protein in at least 2 replicates
        if len(norm_abundances) == 3:
            curve_fold_change, curve_peak = calcCurveFoldChange(
                norm_abundances, uniprot_accession
            )
            residuals_all, r_squared_all = calcResidualsR2All(norm_abundances)
            metrics = {
                "standard_deviation": std, 
                "variance_average": round(moment(norm_abundance_average_list, moment=2),2),
                "skewness_average": moment(norm_abundance_average_list, moment=3),
                "kurtosis_average": moment(norm_abundance_average_list, moment=4),
                "peak_average": max(norm_abundance_average, key=norm_abundance_average.get),
                "max_fold_change_average": max_fold_change,
                "residuals_average": residuals,
                "R_squared_average": r_squared,
                "residuals_all": residuals_all,
                "R_squared_all": r_squared_all,
                "curve_fold_change": curve_fold_change,
                "curve_peak": curve_peak,
            }

    except Exception as e:
        print(uniprot_accession)
        print(norm_abundance_average)
        print(norm_abundance_average_list)
        print(range(0, len(norm_abundance_average)))
        print(traceback.format_exc())

    return metrics

####-------------------------------------------------------------####
# Statistics

def createAbundanceDf(combined_time_course_info, norm_method, raw = False, phospho = False, phospho_ab = False, phospho_reg = False):
    """Transforms the abundance data into a dataframe and groups the columns by timepoint"""

    time_course_abundance = {}

    for accession in combined_time_course_info:
        if phospho == True:
            for site in combined_time_course_info[accession]["phosphorylation_abundances"]:
                protein_abundances_all = combined_time_course_info[accession]["phosphorylation_abundances"][site]["position_abundances"]
                site_key = accession + "_" + combined_time_course_info[accession]['phosphorylation_abundances'][site]['phosphorylation_site']
                if len(protein_abundances_all) != 0:
                    if raw == True:
                        protein_abundances = protein_abundances_all["raw"]
                    else:
                        protein_abundances = protein_abundances_all["normalised"][norm_method]
                        if phospho_ab == True:
                            if "protein_oscillation_abundances" in combined_time_course_info[accession]["phosphorylation_abundances"][site]:
                                protein_abundances = combined_time_course_info[accession]["phosphorylation_abundances"][site]["protein_oscillation_abundances"][norm_method]
                            else:
                                continue
                        if phospho_reg == True:
                            if "phospho_regression" in combined_time_course_info[accession]["phosphorylation_abundances"][site]:
                                protein_abundances = combined_time_course_info[accession]["phosphorylation_abundances"][site]["phospho_regression"][norm_method]
                            else:
                                continue
                        # for abundance_rep in protein_abundances:
                        for abundance_rep in ['abundance_rep_1','abundance_rep_2']:
                            if site_key not in time_course_abundance:
                                time_course_abundance[site_key] = {}
                            for timepoint in protein_abundances[abundance_rep]:
                                if phospho_ab == False:
                                    rep = "_".join(abundance_rep.split("_", 1)[1].split("_")[:2])
                                    rep_timepoint = rep + "_" + timepoint
                                else:
                                    rep = "_".join(abundance_rep.split("_", 1)[1].split("_")[:2])
                                    rep_timepoint = rep + "_" + timepoint
                                time_course_abundance[site_key][rep_timepoint] = protein_abundances[abundance_rep][timepoint]
        else:
            protein_abundances_all = combined_time_course_info[accession]["protein_abundances"]
            if len(protein_abundances_all["raw"]) != 0:
                if raw == True:
                    protein_abundances = protein_abundances_all["raw"]
                else:
                    protein_abundances = protein_abundances_all["normalised"][norm_method]
                    for abundance_rep in protein_abundances:
                        if abundance_rep == "abundance_average":
                            continue
                        if accession not in time_course_abundance:
                            time_course_abundance[accession] = {}
                        for timepoint in protein_abundances[abundance_rep]:
                            rep = "_".join(abundance_rep.split("_", 1)[1].split("_")[:2])
                            rep_timepoint = rep + "_" + timepoint
                            time_course_abundance[accession][rep_timepoint] = protein_abundances[abundance_rep][timepoint]

    time_course_abundance_df = pd.DataFrame(time_course_abundance)
    time_course_abundance_df = time_course_abundance_df.T

    # Group by time point
    new_cols = ['rep_1_Palbo', 'rep_2_Palbo',
    'rep_1_Late G1_1', 'rep_2_Late G1_1',
    'rep_1_G1/S', 'rep_2_G1/S',
    'rep_1_S', 'rep_2_S',
    'rep_1_S/G2', 'rep_2_S/G2',
    'rep_1_G2_2', 'rep_2_G2_2',
    'rep_1_G2/M_1', 'rep_2_G2/M_1',
    'rep_1_M/Early G1', 'rep_2_M/Early G1']

    time_course_abundance_df = time_course_abundance_df[new_cols]

    return time_course_abundance_df

def tp(timepoint,input_protein_data):
    """
    Groups by time point for the replicates
    """
    if "abundance_rep_ 1" in input_protein_data or "abundance_rep_2" in input_protein_data:
        rep_1 = "abundance_rep_1"
        rep_2 = "abundance_rep_2"

    res = []
    try:
        if timepoint in input_protein_data[rep_1]:
            rep_1 = float(input_protein_data[rep_1][timepoint])
            res.append(rep_1)

        if timepoint in input_protein_data[rep_2]:
            rep_2 = float(input_protein_data[rep_2][timepoint])
            res.append(rep_2)

        res = [x for x in res if x == x]
    
    except Exception as e:
        print(input_protein_data)
        print(traceback.format_exc())

    return res

def calcANOVA(input_protein_data):
    """
    Groups by time point and performs ANOVA between all time point groups.
    """
    try:
        timepoints_1 = [tp('Palbo',input_protein_data),tp('Late G1_1', input_protein_data), tp('G1/S', input_protein_data),
            tp('S', input_protein_data),tp('S/G2', input_protein_data),tp('G2_2', input_protein_data),
            tp('G2/M_1',input_protein_data),tp('M/Early G1', input_protein_data)]

        timepoints = [x for x in timepoints_1 if x != []]
        
        # Protein info in at least 2 reps:
        if len(timepoints) != 0:
            one_way_anova = f_oneway(*timepoints)
            f_statistic = one_way_anova[0]
            p_value = one_way_anova[1]
            if np.isnan(p_value):
                p_value = 1
        else:
            f_statistic = 1
            p_value = 1

    except Exception as e:
        print(input_protein_data)
        print(timepoints)
        print(traceback.format_exc())

    return p_value, f_statistic

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    logger.info("Calculate Benjamini-Hochberg p-value correction")
    p = np.asarray(p, dtype=float)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))

    return q[by_orig]

def calcFisherG(combined_time_course_info, norm_method, raw = False, phospho = False, phospho_ab = False, phospho_reg = False):
    """
    Performs a periodicity test using the 'Fisher' method.
    it takes a vector containing just numeric abundance values as input
    where each vector is a different protein and each value corresponds to a different timepoint, ordered from low to high
    the output is a dictionary containing the fisher g-statistic, pvalues and periodic frequencies for each protein.
    """
    # R part for Fisher G-Statistic
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    import rpy2.robjects.packages as rpackages
    # R vector of strings
    from rpy2.robjects.vectors import StrVector

    logger.info("Calculate Fisher G Statistics")
    # import R's utility package
    utils = rpackages.importr('utils')
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    packnames = ('perARMA', 'quantreg')
    # Selectively install what needs to be install.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    not_in_cran = ("ptest")
    not_in_crac_names_to_install = [x for x in not_in_cran if not rpackages.isinstalled(x)]
    if len(not_in_crac_names_to_install) > 0:
        ro.r('install.packages("https://cran.r-project.org/src/contrib/Archive/ptest/ptest_1.0-8.tar.gz", repos = NULL, type = "source")')

    time_course_fisher = createAbundanceDf(combined_time_course_info, norm_method, raw, phospho, phospho_ab, phospho_reg)
    time_course_fisher = time_course_fisher.dropna()

    for index,row in time_course_fisher.iterrows():
        row_z = row.tolist()
        row_z = [str(i) for i in row_z]
        ro.r('''
            library(ptest)
            z <- c(''' + ','.join(row_z) + ''')
            # p_value <- ptestg(z,method="Fisher")$pvalue
            # # freq <- ptestg(z,method="Fisher")$freq
            # # g_stat <- ptestg(z,method="Fisher")$$obsStat
            ptestg_res <- ptestg(z,method="Fisher")
        '''
        )
        ptestg_res = ro.globalenv['ptestg_res']
        g_stat = ptestg_res[0]
        p_value = ptestg_res[1]
        freq = ptestg_res[2]

        time_course_fisher.loc[[index],['G_statistic']] = g_stat
        time_course_fisher.loc[[index],['p_value']] = p_value
        time_course_fisher.loc[[index],['frequency']] = freq

    # Add q-value columns 
    q_value = p_adjust_bh(time_course_fisher['p_value'])

    time_course_fisher['q_value'] = q_value
    
    # Turn df into a dictionary
    cols = time_course_fisher.columns
    fisher_cols = ['G_statistic','p_value', 'frequency', 'q_value']
    ab_col = [x for x in cols if x not in fisher_cols]
    time_course_fisher = time_course_fisher.drop(columns=ab_col)
    time_course_fisher_dict = time_course_fisher.to_dict('index')

    return time_course_fisher_dict

def addPhosphoMetrics(time_course_phospho):
    """
    Add all metrics for each phosphosite.
    """
    for uniprot_accession in time_course_phospho:
        for site in time_course_phospho[uniprot_accession]["phosphorylation_abundances"]:
            norm_abundances = time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["position_abundances"]["normalised"]

            time_course_phospho[uniprot_accession]["phosphorylation_abundances"][
                site
            ]["metrics"] = {
                "log2_mean": calcAbundanceMetrics(
                    norm_abundances["log2_mean"], uniprot_accession),
                "0-max": calcAbundanceMetrics(
                    norm_abundances["0-max"], uniprot_accession
                )
            }
            # ANOVA
            p_value, f_statistic = calcANOVA(norm_abundances["log2_mean"])
            time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["ANOVA"] = {}
            time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["ANOVA"]["p_value"] = p_value
            time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["ANOVA"]["F_statistics"] = f_statistic
    
    # Fisher G Statistic
    time_course_fisher_dict = calcFisherG(time_course_phospho, "log2_mean", raw = False, phospho = True)

    # Corrected q values - Phospho
    # 1) Create a dataframe with the desired regression info
    phospho_anova_info = {}
    for uniprot_accession in time_course_phospho:
        for site in time_course_phospho[uniprot_accession]["phosphorylation_abundances"]:
            phospho_key = uniprot_accession + "_" + time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]['phosphorylation_site']
            if phospho_key not in phospho_anova_info:
                phospho_anova_info[phospho_key] = {}
            phospho_anova_info[phospho_key]['p_value'] = time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["ANOVA"]["p_value"]
    
    phospho_anova_info_df = pd.DataFrame(phospho_anova_info)
    phospho_anova_info_df = phospho_anova_info_df.T
    # 2) Regression ANOVA q values
    phospho_anova_info_df['q_value'] = p_adjust_bh(phospho_anova_info_df['p_value'])
    # 3) Turn dataframe into a dictionary
    phospho_anova_info = phospho_anova_info_df.to_dict('index')
    # 4) Add Regression info in time_course_phospho dictionary
    for uniprot_accession in time_course_phospho:
        for site in time_course_phospho[uniprot_accession]["phosphorylation_abundances"]:
            site_key = uniprot_accession + "_" + time_course_phospho[uniprot_accession]['phosphorylation_abundances'][site]['phosphorylation_site']
            # ANOVA q values
            if site_key in phospho_anova_info:
                time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["ANOVA"]["q_value"] = phospho_anova_info[site_key]['q_value']
            else:
                time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["ANOVA"]["q_value"] = 1
            # Fisher
            if site_key in time_course_fisher_dict:
                time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["Fisher_G"] = time_course_fisher_dict[site_key]
            else:
                time_course_phospho[uniprot_accession]["phosphorylation_abundances"][site]["metrics"]["log2_mean"]["Fisher_G"] = {'G_statistic': 1, 'p_value': 1, 'frequency': 1, 'q_value': 1}

    return time_course_phospho

####-------------------------------------------------------------####
# Normalisation Methods

def calcColumnMedian(proteome_sample, data_source, phospho={}):
    """
    Calculates the median per column for each replicate.
    """
    logger.info("Calculate per Column Median for " + data_source)

    rep_abundance_timepoint = {}
    rep_median = {}

    if len(phospho) == 0:
        rep_sample_time_map = time_points_mapping[data_source]

        for accession in proteome_sample["data"]:
            protein_abundance = proteome_sample["data"][accession]

            for data_key in data_files_datakeys[data_source]:
                if data_key in protein_abundance:
                    if data_key not in rep_abundance_timepoint:
                            rep_abundance_timepoint[data_key] = []
                    rep_abundance_timepoint[data_key].append(protein_abundance[data_key])

            for rep_sample_name in rep_abundance_timepoint:
                timepoint = [timepoint for timepoint, sample_name in rep_sample_time_map.items() if sample_name == rep_sample_name][0]
                rep_median[timepoint] = statistics.median(rep_abundance_timepoint[rep_sample_name])
    
    else:
        for accession in phospho:
            for site in phospho[accession]["phosphorylation_abundances"]:
                for rep in phospho[accession]["phosphorylation_abundances"][site]["position_abundances"]["raw"]:
                    if rep not in rep_median:
                        rep_median[rep] = {}
                    if rep not in rep_abundance_timepoint:
                        rep_abundance_timepoint[rep] = {}

                    phosphosite_abundance = phospho[accession]["phosphorylation_abundances"][site]["position_abundances"]["raw"][rep]
            
                    for timepoint in phosphosite_abundance:
                        if timepoint not in rep_abundance_timepoint[rep]:
                            rep_abundance_timepoint[rep][timepoint] = []
                        rep_abundance_timepoint[rep][timepoint].append(phosphosite_abundance[timepoint])

        for rep in rep_abundance_timepoint:
            for timepoint in rep_abundance_timepoint[rep]:
                rep_median[rep][timepoint] = statistics.median(rep_abundance_timepoint[rep][timepoint])

    return rep_median

def firstLevelNormalisationProteomics(rep_median_col, abundance):
        """
        First level/sample normalisation the raw abundances of each protein. 
        Normalise raw abundances by diving with column median.
        """
        normalised_abundance = {}
        for time_point in rep_median_col:
            if time_point in abundance: 
                normalised_abundance[time_point] = abundance[time_point] / rep_median_col[time_point]

        return normalised_abundance

def calclog2PalboNormalisation(abundances):
        """
        Log2 transforms the abundance data and normalises each replicate on Palbo Arrest abundance.
        """
        normalised_abundance = {}

        for replicate in abundances:
            normalised_abundance[replicate] = {}
            if "Palbo" in abundances[replicate]:
                palbo = abundances[replicate]["Palbo"]
                for timepoint, abundance in abundances[replicate].items():
                    normalised_abundance[replicate][timepoint] = round(math.log2(abundance/ palbo), 4)

        return normalised_abundance

def calclog2RelativeAbundance(input_protein_data):
    """
    Normalisation method:
    First stage: log2 transform to achieve relative scaling
    Second stage: Subtract the mean to center at zero.
    """
    log2_mean_norm = {}
    abundance_rep_1 = {}
    abundance_rep_2 = {}

    if "abundance_rep_1" in input_protein_data:
        abundance_rep_1 = input_protein_data["abundance_rep_1"]
        log2_mean_norm["rep_1"] = {}
    if "abundance_rep_2" in input_protein_data:
        abundance_rep_2 = input_protein_data["abundance_rep_2"]
        log2_mean_norm["rep_2"] = {}

    if len(abundance_rep_1) != 0:

        # log2 Transform
        for timepoint, abundance in abundance_rep_1.items():
            log2_mean_norm["rep_1"][timepoint] = math.log2(abundance)
        for timepoint, abundance in abundance_rep_2.items():
            log2_mean_norm["rep_2"][timepoint] = math.log2(abundance)

        # Calculate the mean grouped by experiment
        mean_rep_1_2 = (
            sum(log2_mean_norm["rep_1"].values())
            + sum(log2_mean_norm["rep_2"].values())
        ) / (len(log2_mean_norm["rep_1"]) + len(log2_mean_norm["rep_2"]))

        # Subtract the mean from the abundances
        for timepoint, abundance in log2_mean_norm["rep_1"].items():
            log2_mean_norm["rep_1"][timepoint] = round(
                (abundance - mean_rep_1_2), 4
            )
        for timepoint, abundance in log2_mean_norm["rep_2"].items():
            log2_mean_norm["rep_2"][timepoint] = round(
                (abundance - mean_rep_1_2), 4
            )

    return log2_mean_norm

def normaliseData(abundance, zero_min=False):
    """
    Normalise the abundance values and store them in the abuncance_normalised dictionary
    min-max method
    """

    norm_abundance = {}

    abundance_data = list(abundance.values())

    min_value = min(abundance_data)
    if zero_min == True:
        min_value = 0
    max_value = max(abundance_data)

    for time_point in abundance:
        abundance_value = abundance[time_point]
        try:
            abundance_normalised = (abundance_value - min_value) / ( 
                max_value - min_value
            )
        except:
            abundance_normalised = 0.5

        norm_abundance[time_point] = round(abundance_normalised, 4)

    return norm_abundance

def imputeData(normalised_abundace):
    """
    Imputes missing time_points and abundance data. Fits info when there is no info available
    """

    abundance_imputed = {}
    abundance_normalised = []

    for time_point_iter in range(0, len(time_points)):
        time_point = time_points[time_point_iter]

        last_time_point_data = [None, None]
        next_time_point_data = [None, None]

        if time_point in normalised_abundace:
            #  Timepoint was tested and returned data
            time_point_value = "%1.2f" % normalised_abundace[time_point]
            time_point_value_status = "experimental"

        else:
            ## Timepoint was not tested
            for i in range(1, len(time_points)):
                if last_time_point_data[0] == None:
                    # Look backwards through the cell cycle
                    if (
                        time_points[time_point_iter - i]
                        in normalised_abundace
                    ):
                        # Comparison timepoint was tested but did not return data
                        last_time_point_abundance = 0
                        last_time_point_data = [
                            time_points[time_point_iter - i],
                            i,
                            last_time_point_abundance,]

                    if (
                        time_points[time_point_iter - i]
                        in normalised_abundace
                    ):
                        # Comparison timepoint was tested and returned data
                        last_time_point_abundance = normalised_abundace[
                            time_points[time_point_iter - i]
                        ]
                        last_time_point_data = [
                            time_points[time_point_iter - i],
                            i,
                            last_time_point_abundance,]

                # if we're in the last timepoint (M --> G1)
                next_time_point_modulo = time_points[
                    (time_point_iter + i) % len(time_points)
                ]

                if next_time_point_data[0] == None:
                    # Looks forward in the cell cycle
                    if next_time_point_modulo in normalised_abundace:
                        # Comparison timepoint was tested but did not return data
                        next_time_point_abundance = 0
                        next_time_point_data = [
                            time_points[time_point_iter - i],
                            i,
                            next_time_point_abundance,
                        ]

                    if next_time_point_modulo in normalised_abundace:
                        # Comparison timepoint was tested and returned data
                        next_time_point_abundance = normalised_abundace[
                            time_points[
                                (time_point_iter + i) % len(time_points)
                            ]
                        ]
                        next_time_point_data = [
                            time_points[
                                (time_point_iter + i) % len(time_points)
                            ],
                            i,
                            next_time_point_abundance,
                        ]

            last_time_point_distance = last_time_point_data[1]
            next_time_point_distance = next_time_point_data[1]
            last_time_point_abundance = last_time_point_data[2]
            next_time_point_abundance = next_time_point_data[2]

            # imputation algorithm
            step_height = (
                last_time_point_abundance - next_time_point_abundance
            ) / (last_time_point_distance + next_time_point_distance)
            time_point_value = (
                next_time_point_distance * step_height + next_time_point_abundance
            )

            time_point_value_status = "imputed"

        ###
        abundance_normalised.append(str(time_point_value))
        abundance_imputed[time_point] = {
            "value": round(float(time_point_value), 4),
            "status": time_point_value_status,
        }

    return abundance_imputed   

def firstLevelNormalisationPhospho(abundance, rep_median):
    """
    First level/sample normalisation the raw phospho abundances of each protein. Sum+Median
    """
    normalised_abundance = {}

    for rep in abundance:
        if rep not in normalised_abundance:
            normalised_abundance[rep] = {}
        for time_point in abundance[rep]:
            if time_point in rep_median[rep]:
                normalised_abundance[rep][time_point] = (abundance[rep][time_point] / rep_median[rep][time_point])
                    
    return normalised_abundance

####-------------------------------------------------------------####
# Kinases Prediction

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
    
####-------------------------------------------------------------####
# Parse Phospho Data

def findPositionInMasterProtein(data_point):
        """
        Finds the positions of the peptide in the master protein.
        Input: "P16402 [35-47]; P10412 [34-46]; P16403 [34-46]"
        Output: {'P16402': {'start': '35', 'end': '47'}, 'P10412': {'start': '34', 'end': '46'}, 'P16403': {'start': '34', 'end': '46'}}
        """
        position_in_master_protein = data_point
        protein_position_info = {}

        positions = position_in_master_protein.split(";")
        for position in positions:
            position = position.strip(" ")
            position_info = position.split(" ")
            for index, item in enumerate(position_info):
                position_info[index] = item.strip("[").strip("]")
                if len(position_info) > 1:
                    position_ranges = position_info[1].split("-")
                else:
                    position_ranges = position_info[0].split("-")
                protein_position_info[position_info[0]] = {
                    "start": position_ranges[0],
                    "end": position_ranges[1],
                }
        return protein_position_info

def findSite(sites):
    """
    Gets the phosphorylation sites for rep1 (merge file) and returns a dictionary with all the information.
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
                    # print("-----> [Sx]")
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
                    # print("-----> [S/Y]")
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
                    # print("-----> [x-x]")
                    aa = "-"
                    position = pp_site
                    site_range = pp_site.split("-")
                    # uniprot_position = {"start": "-", "end": "-"}
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

def findModificationPositionRep1(data_point):
    """
    Creates and returns a dictionary that contain information about the positions in the master protein
    and the modification events.
    Example input : ["P20700 1xPhospho [S393]; Q03252 1xPhospho [S407]", "A0A0B4J2F2 1xPhospho [S575]", "O15085 [1452-1473]"
                                    "Q9NYF8 2xPhospho [S290 S/Y]", "Q9P206 2xPhospho [S971 S979]"]

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

    if len(modifications) == 0 or modifications == ['']:
        # modifications = "Q07820 [137-176]"
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
                if len(sites) == 0:
                    # 'S340(99.2)]'
                    sites = [event]

            sites = findSite(sites)
            # Add the peptide uniprot position on site name of the no specific phosphosites
            # e.g. T/S --> T/S[123-130]
            for p_site, info in sites.items():
                if p_site.find("/") != -1 or len(p_site) == 1:
                    range_name = (
                        "["
                        + str(positions[accession]["start"])
                        + "-"
                        + str(positions[accession]["end"])
                        + "]"
                    )
                    p_site = p_site + range_name
                if (
                    info["phosphosite"].find("/") != -1
                    or len(info["phosphosite"]) == 1
                ):
                    info["phosphosite"] = info["phosphosite"] + range_name
                    info["position"] = info["position"] + range_name

                if p_site not in modification_info[accession]:
                    modification_info[accession][p_site] = {}
                modification_info[accession][p_site]["phosphosite"] = p_site
                modification_info[accession][p_site]["aa"] = info["aa"]
                modification_info[accession][p_site]["position"] = info["position"]
                modification_info[accession][p_site][
                    "uniprot_position"
                ] = positions[accession]
                modification_info[accession][p_site]["likelihood"] = info[
                    "likelihood"
                ]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(modification)
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)

    return modification_info

def findPhosphoAbundance(data_source, data_point, uniprot_accession_key):
    """
    Creates and returns a raw phospho abundance dictionary for one protein.
    Data_points: Directly reading from a file.
    """
    ab = {}
    abundance = {uniprot_accession_key: {}}
    for datakey in data_files_datakeys[data_source]:
        value = data_point[datakey]
        if value == "":
            continue
        ab[datakey] = float(value)

    for time_point in time_points:
        col_name = time_points_mapping[data_source][time_point]
        if col_name in ab:
            abundance[uniprot_accession_key][time_point] = ab[col_name]

    return abundance

def calculateProteinOscillationAbundances(combined_time_course_info, norm_method, uniprot_accession, phosphosite):
    """
    # Protein oscillation normalised abundances to the phospho part of the JSON too.
    # It would just be the normalised phopho abundances minus the normalised protein abundances per replicate.
    # And then the average of them.
    """
    phospho_oscillations = {}

    # Replicate 1

    # protein exists
    if "abundance_rep_1" in combined_time_course_info[uniprot_accession]["protein_abundances"]["normalised"][norm_method]:
        # site exists
        if ("abundance_rep_1" in combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                phosphosite]["position_abundances"]["normalised"][norm_method]):
            
            protein_oscillation_abundances_rep_1 = {}
            protein_normed_abundance_rep_1 = combined_time_course_info[uniprot_accession][
                "protein_abundances"]["normalised"][norm_method]["abundance_rep_1"]
            
            phosho_normed_abundance_rep_1 = combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"][phosphosite]["position_abundances"]["normalised"][norm_method][
                "abundance_rep_1"]
            
            for time_point in protein_normed_abundance_rep_1:
                if time_point in phosho_normed_abundance_rep_1:
                    protein_oscillation_abundances_rep_1[time_point] = (
                        phosho_normed_abundance_rep_1[time_point]
                        - protein_normed_abundance_rep_1[time_point]
                    )
            phospho_oscillations["rep_1"] = protein_oscillation_abundances_rep_1

    # Replicate 2
    if "abundance_rep_2" in combined_time_course_info[uniprot_accession]["protein_abundances"]["normalised"][norm_method]:
        if ("abundance_rep_2" in combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                phosphosite]["position_abundances"]["normalised"][norm_method]):
            
            protein_oscillation_abundances_rep_2 = {}
            protein_normed_abundance_rep_2 = combined_time_course_info[uniprot_accession][
                "protein_abundances"]["normalised"][norm_method]["abundance_rep_2"]
            phosho_normed_abundance_rep_2 = combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"][phosphosite]["position_abundances"]["normalised"][norm_method][
                "abundance_rep_2"]
            for time_point in protein_normed_abundance_rep_2:
                if time_point in phosho_normed_abundance_rep_2:
                    protein_oscillation_abundances_rep_2[time_point] = (
                        phosho_normed_abundance_rep_2[time_point]
                        - protein_normed_abundance_rep_2[time_point]
                        )
            phospho_oscillations["rep_2"] = protein_oscillation_abundances_rep_2
    
    return phospho_oscillations

####-------------------------------------------------------------####
# Parsing the data

def TimeCourseProteomicsParser(data_source):
    """
    Reads and parses the proteomics data from the TimeCourse replicates experiment for every protein in the dataset.
    """
    
    logger.info("Loading abundance data from TimeCourse Proteomics")
    cellCycleDataset = {"data": {}}

    data_points = utilities_basicReader.readTableFile(
        "./data/" + data_files[data_source],
        byColumn=False,
        stripQuotes=True,
    )
    contaminants = []
    for data_point in data_points:
        try:
            uniprot_accession = data_points[data_point]["Accession"]
            gene = data_points[data_point]["Gene Symbol"]
            contaminant = data_points[data_point]["Contaminant"]

            if (
                uniprot_accession == ""
                or uniprot_accession == "Accession"
                or contaminant == "Yes"
                or contaminant == "TRUE"
            ):
                contaminants.append(uniprot_accession)
                continue

            if uniprot_accession not in cellCycleDataset["data"]:
                cellCycleDataset["data"][uniprot_accession] = {"Gene Symbol": gene}
            for datakey in data_files_datakeys[
                data_source
            ]:
                value = data_points[data_point][datakey]
                if value == "":
                    continue
                cellCycleDataset["data"][uniprot_accession][datakey] = float(value)

            if len(cellCycleDataset["data"][uniprot_accession]) == 1:
                del cellCycleDataset["data"][uniprot_accession]
            else:
                cellCycleDataset["data"][uniprot_accession]["confidence"] = {
                    "coverage": data_points[data_point]["Coverage [%]"],
                    "protein_groups": data_points[data_point]["# Protein Groups"],
                    "peptides": data_points[data_point]["# Peptides"],
                    "PSMs": data_points[data_point]["# PSMs"],
                    "unique_peptides": data_points[data_point]["# Unique Peptides"],
                    "aas": data_points[data_point]["# AAs"],
                    "MW": data_points[data_point]["MW [kDa]"],
                    "protein_FDR_confidence": data_points[data_point]['Protein FDR Confidence: Combined'],
                    "Exp._q-value": data_points[data_point]['Exp. q-value: Combined'],
                    "Sum_PEP_Score": data_points[data_point]['Sum PEP Score'],
                    "Score_Sequest_HT": data_points[data_point]['Score Sequest HT'],
                    "calc._pI": data_points[data_point]['calc. pI'],
                }

        except Exception as e:
            print("Error {}: in {}".format(e, uniprot_accession))

    logger.info("Added cross-reference - " + data_source)
    print(len(cellCycleDataset["data"]))

    return cellCycleDataset

def createTimeCourseProteome():
    """
    Creates the main TimeCourse Proteome experiment info along with the normalised 
    values of the protein abundances.
    """

    time_course_proteome = {}

    proteome_rep_1 = TimeCourseProteomicsParser("TimeCourse_Proteomics_rep_1")
    rep1_col_median = calcColumnMedian(proteome_rep_1, "TimeCourse_Proteomics_rep_1")
    proteome_rep_2 = TimeCourseProteomicsParser("TimeCourse_Proteomics_rep_2")
    rep1_col_median = calcColumnMedian(proteome_rep_2, "TimeCourse_Proteomics_rep_2")

    logger.info("Combine Proteomics and Phosphoproteomics info")

    for uniprot_accession in proteome_rep_1['data']:
        try:
            if uniprot_accession in index_protein_names:
                gene_name = index_protein_names[uniprot_accession]["gene_name"]
                protein_name = index_protein_names[uniprot_accession][
                    "protein_name"]
            else:
                gene_name, protein_name = getProteinInfo(uniprot_accession)

            time_course_proteome[uniprot_accession] = {
                "gene_name": gene_name,
                "protein_name": protein_name,
                "protein_abundances": {
                    "raw": {},
                    "normalised": {"median": {}, "log2_mean": {}, "0-max": {}, "min-max": {}, "log2_palbo":{}},
                    "imputed": {},
                
                },
                "phosphorylation_abundances": {},
                "confidence": {},
                "protein_info":{},
            }

            abundance_rep_1 = proteome_rep_1["data"][uniprot_accession]
            abundance_rep_2 = proteome_rep_2["data"][uniprot_accession]

            if len(abundance_rep_1) != 1:
                abundance_rep_1 = renameTimepoints("TimeCourse_Proteomics_rep_1", abundance_rep_1)
                abundance_rep_2 = renameTimepoints("TimeCourse_Proteomics_rep_2", abundance_rep_2)
                
                # Raw Abundances
                raw_abundances = time_course_proteome[uniprot_accession]["protein_abundances"]["raw"]
                
                raw_abundances["abundance_rep_1"] = abundance_rep_1
                raw_abundances["abundance_rep_2"]  = abundance_rep_2

                abundance_average = calculateAverageRepAbundance(raw_abundances, imputed=False)
                raw_abundances["abundance_average"] = abundance_average

                # First Level Normalisation - Normalise raw abundances by diving with column median
                norm_abundances = time_course_proteome[uniprot_accession]["protein_abundances"]["normalised"]

                first_norm_abundance_rep_1_2 = {}
                first_norm_abundance_rep_1_2["abundance_rep_1"] = firstLevelNormalisationProteomics(rep1_col_median, abundance_rep_1)
                norm_abundances["median"]["abundance_rep_1"] =  first_norm_abundance_rep_1_2["abundance_rep_1"]
                first_norm_abundance_rep_1_2["abundance_rep_2"] = firstLevelNormalisationProteomics(rep1_col_median, abundance_rep_2)
                norm_abundances["median"]["abundance_rep_2"] =  first_norm_abundance_rep_1_2["abundance_rep_2"]

                norm_abundances[
                    "median"]["abundance_average"] = calculateAverageRepAbundance(
                        norm_abundances["median"], imputed=False)

                # Log2 - Palbo Normalisation
                norm_abundances["log2_palbo"] = calclog2PalboNormalisation(first_norm_abundance_rep_1_2)
                norm_abundances[
                    "log2_palbo"]["abundance_average"] = calculateAverageRepAbundance(
                    norm_abundances["log2_palbo"], imputed=False)

                # Log2 - substract row mean for every batch seperately
                log2_norm_abundance_reps = calclog2RelativeAbundance(first_norm_abundance_rep_1_2)
                norm_abundance_rep_1 = log2_norm_abundance_reps["rep_1"]
                norm_abundance_rep_2 = log2_norm_abundance_reps["rep_2"]

                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "normalised"]["log2_mean"]["abundance_rep_1"] = norm_abundance_rep_1
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "normalised"]["log2_mean"]["abundance_rep_2"] = norm_abundance_rep_2 
                
                norm_abundances[
                    "log2_mean"]["abundance_average"] = calculateAverageRepAbundance(
                    norm_abundances["log2_mean"], imputed=False)
                
                # Normalised - Second Level
                # min-max
                normed_abundance_rep_1 = normaliseData(norm_abundance_rep_1)
                normed_abundance_rep_2 = normaliseData(norm_abundance_rep_2)
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "normalised"]["min-max"]["abundance_rep_1"] = normed_abundance_rep_1
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "normalised"]["min-max"]["abundance_rep_2"] = normed_abundance_rep_2
                
                norm_abundances[
                    "min-max"]["abundance_average"] = calculateAverageRepAbundance(
                    norm_abundances["min-max"], imputed=False)

                # 0-max
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "normalised"]["0-max"]["abundance_rep_1"] = normaliseData(
                    first_norm_abundance_rep_1_2["abundance_rep_1"], zero_min=True
                )
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "normalised"]["0-max"]["abundance_rep_2"] = normaliseData(
                    first_norm_abundance_rep_1_2["abundance_rep_2"], zero_min=True
                )

                norm_abundances[
                    "0-max"]["abundance_average"] = calculateAverageRepAbundance(
                    norm_abundances["0-max"], imputed=False)
    
                # Imputed
                imputed_abundance_rep_1 = imputeData(normed_abundance_rep_1)
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "imputed"]["abundance_rep_1"] = imputed_abundance_rep_1
                imputed_abundance_rep_2 = imputeData(normed_abundance_rep_2)
                time_course_proteome[uniprot_accession]["protein_abundances"][
                    "imputed"]["abundance_rep_2"] = imputed_abundance_rep_2
                
                imputed_abundances = time_course_proteome[uniprot_accession]["protein_abundances"]["imputed"]
                imputed_abundance_average = calculateAverageRepAbundance(imputed_abundances, imputed=True)
                time_course_proteome[uniprot_accession]["protein_abundances"]["imputed"][
                    "abundance_average"] = imputed_abundance_average
                
                # Metrics
                time_course_proteome[uniprot_accession]["metrics"] = {
                    "log2_mean": calcAbundanceMetrics(
                        norm_abundances["log2_mean"], uniprot_accession ),
                    "0-max": calcAbundanceMetrics(
                        norm_abundances["0-max"], uniprot_accession
                    )}

                # ANOVA
                p_value, f_statistic = calcANOVA(norm_abundances["log2_mean"])
                time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["ANOVA"] = {}
                time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["ANOVA"]["p_value"] = p_value
                time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["ANOVA"]["F_statistics"] = f_statistic

                # Confidence
                time_course_proteome[uniprot_accession]["confidence"] = proteome_rep_1["data"][
                    uniprot_accession]["confidence"]

        except Exception as e:
            print(uniprot_accession)
            print(traceback.format_exc())

    # Fisher G Statistic
    time_course_fisher_dict = calcFisherG(time_course_proteome, "log2_mean", raw = False)

    # 1) Create a dataframe with the desired anova info
    prot_anova_info = {}
    for uniprot_accession in time_course_proteome:
        if uniprot_accession not in prot_anova_info:
            prot_anova_info[uniprot_accession] = {}
        prot_anova_info[uniprot_accession]['p_value'] = time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["ANOVA"]["p_value"]

    prot_anova_info_df = pd.DataFrame(prot_anova_info)
    prot_anova_info_df = prot_anova_info_df.T

    # 2) Protein ANOVA q values
    prot_anova_info_df['q_value'] = p_adjust_bh(prot_anova_info_df['p_value'])
    # 3) Turn dataframe into a dictionary
    prot_anova_info = prot_anova_info_df.to_dict('index')
    # 4) Add Protein ANOVA info in time_course_proteome dictionary
    for uniprot_accession in time_course_proteome:
        # ANOVA q values
        if uniprot_accession in prot_anova_info:
            time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["ANOVA"]["q_value"] = prot_anova_info[uniprot_accession]['q_value']
        else:
            time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["ANOVA"]["q_value"] = 1
        # Fisher
        if uniprot_accession in time_course_fisher_dict:
            time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["Fisher_G"] = time_course_fisher_dict[uniprot_accession]
        else:
            time_course_proteome[uniprot_accession]["metrics"]["log2_mean"]["Fisher_G"] = {'G_statistic': 1, 'p_value': 1, 'frequency': 1, 'q_value': 1}

    # Add Protein Annotations
    time_course_proteome = addProteinAnnotations(time_course_proteome, index_protein_names)

    return time_course_proteome

def parseTimeCoursePhosphoProteomics():
    """
    Parses the phosphoproteomics data from all the TimeCourse replicates for every protein in the dataset.
    """
    logger.info("Parsing phospho abundance data from all TimeCourse Phosphoproteomics Replicates")

    contaminants = findContaminants()
    phospho_rep = {}

    data_points_2 = utilities_basicReader.readTableFile(
        "./data/" + data_files["TimeCourse_Phosphoproteomics_rep_1"],
        byColumn=False,stripQuotes=True,)
    
    for key, data_point_2 in data_points_2.items():
        if key == 0:
            continue
        try:
            modification_info = findModificationPositionRep1(data_point_2)
            for uniprot_accession_key in modification_info:
                if (
                    uniprot_accession_key == ""
                    or uniprot_accession_key in contaminants
                ):
                    continue

                abundance_rep_1 = findPhosphoAbundance(
                    "TimeCourse_Phosphoproteomics_rep_1",
                    data_point_2,uniprot_accession_key,)
        
                abundance_rep_2 = findPhosphoAbundance(
                    "TimeCourse_Phosphoproteomics_rep_2",
                    data_point_2, uniprot_accession_key)

                if (
                    len(abundance_rep_1[uniprot_accession_key]) == 0
                    and len(abundance_rep_2[uniprot_accession_key]) == 0
                ):
                    continue
                
                if uniprot_accession_key not in phospho_rep:
                    phospho_rep[uniprot_accession_key] = {
                        "phosphorylation_abundances": {}
                    }

                for phosphosite in modification_info[uniprot_accession_key]:
                    mod_key = modification_info[uniprot_accession_key][phosphosite][
                        "position"
                    ]
                    modification = data_point_2["Modifications"]
                    phosphosite_info = modification_info[uniprot_accession_key][
                        phosphosite
                    ]

                    if (
                        mod_key
                        not in phospho_rep[uniprot_accession_key][
                            "phosphorylation_abundances"
                        ]
                    ):
                        phospho_rep[uniprot_accession_key][
                            "phosphorylation_abundances"
                        ][mod_key] = {
                            "phosphorylation site": phosphosite_info["phosphosite"],
                            "peptide_abundances": {"rep_1":{}, "rep_2":{}},
                            "position_abundances": {
                                "raw": {},
                                "normalised": {"log2_mean": {}, "min-max": {}, "0-max": {}},
                                "imputed": {},
                            },
                            "confidence": {},
                        }

                    phospho_rep[uniprot_accession_key][
                        "phosphorylation_abundances"
                    ][mod_key]["confidence"]["CR07"] = {
                        "protein_groups": data_point_2["Number of Protein Groups"],
                        "no_proteins": data_point_2["Number of Proteins"],
                        "no_isoforms": data_point_2["Number of Isoforms"],
                        "PSMs": data_point_2["Number of PSMs"],
                        "missed_cleavages": data_point_2[
                            "Number of Missed Cleavages"
                        ],
                        "MHplus": data_point_2["Theo MHplus in Da"],
                        "q_value_HT": data_point_2[
                            "Percolator q-Value by Search Engine Sequest HT"
                        ],
                        "PEP_Sequest": data_point_2[
                            "Percolator PEP by Search Engine Sequest HT"
                        ],
                        "XCorr_HT": data_point_2[
                            "XCorr by Search Engine Sequest HT"
                        ],
                        "confidence_Sequest_HT": data_point_2[
                            "Confidence by Search Engine Sequest HT"
                        ],
                    }

                    
                    if (modification not in phospho_rep[uniprot_accession_key][
                            "phosphorylation_abundances"
                        ][mod_key]["peptide_abundances"]["rep_1"]):

                        phospho_rep[uniprot_accession_key][
                            "phosphorylation_abundances"
                        ][mod_key]["peptide_abundances"]["rep_1"][modification] = {
                            "abundance_rep_1": abundance_rep_1[
                                uniprot_accession_key
                            ],
                            "sequence": data_point_2["Annotated Sequence"],
                            "uniprot_position": phosphosite_info[
                                "uniprot_position"
                            ],
                            "aa": phosphosite_info["aa"],
                            "phosphosite": phosphosite_info["phosphosite"],
                            "likelihood": phosphosite_info["likelihood"],
                            "modification": modification,
                        }

                    if (modification not in phospho_rep[uniprot_accession_key][
                            "phosphorylation_abundances"
                        ][mod_key]["peptide_abundances"]["rep_2"]):
                        phospho_rep[uniprot_accession_key][
                            "phosphorylation_abundances"
                        ][mod_key]["peptide_abundances"]["rep_2"][modification] = {
                            "abundance_rep_2": abundance_rep_2[
                                uniprot_accession_key
                            ],
                            "sequence": data_point_2["Annotated Sequence"],
                            "uniprot_position": phosphosite_info[
                                "uniprot_position"
                            ],
                            "aa": phosphosite_info["aa"],
                            "phosphosite": phosphosite_info["phosphosite"],
                            "likelihood": phosphosite_info["likelihood"],
                            "modification": modification,
                        }

        except Exception as e:
            print("this is a problem")
            print(uniprot_accession_key, mod_key)
            print(modification_info)
            print(traceback.format_exc())
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    return phospho_rep

####-------------------------------------------------------------####
# Combining the data

def addProteinOscillations(combined_time_course_info):
    """
    Calculates and adds all the Protein Oscillations and their metrics for each phosphosite.
    """
    logger.info("Adding Protein Oscillation Normalised Abundances")

    for uniprot_accession in combined_time_course_info:
        # If we have info both in protein and in phospho level
        if len(combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"]) != 0 and len(combined_time_course_info[uniprot_accession][
                "protein_abundances"]["raw"]) != 0:
            # Add the Protein Oscillation Normalised Abundances
            for phosphosite in combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"
            ]:
                combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite
                ]["protein_oscillation_abundances"] = {"0-max":{}, "log2_mean":{}}
                
                for norm_method in ["0-max", "log2_mean"]:
                    protein_oscillation_abundances = combined_time_course_info[
                    uniprot_accession]["phosphorylation_abundances"][phosphosite][
                    "protein_oscillation_abundances"][norm_method]

                    phospho_oscillations = calculateProteinOscillationAbundances(
                        combined_time_course_info, norm_method ,uniprot_accession, phosphosite
                    )
                    for rep in ["rep_1", "rep_2"]:
                        if rep in phospho_oscillations:
                            key = "abundance_" + rep
                            protein_oscillation_abundances[key] = phospho_oscillations[rep]

                    protein_oscillation_abundances["abundance_average"] = calculateAverageRepAbundance(
                        protein_oscillation_abundances,imputed=False,phospho_oscillation=False)
                    
                # if we have info in Protein Oscillation Normalised Abundances
                if (
                    len(
                        combined_time_course_info[
                    uniprot_accession]["phosphorylation_abundances"][phosphosite][
                    "protein_oscillation_abundances"]["log2_mean"])
                    > 1
                ):
                    for norm_method in ["0-max", "log2_mean"]:
                        protein_oscillation_abundances = combined_time_course_info[
                        uniprot_accession]["phosphorylation_abundances"][phosphosite][
                    "protein_oscillation_abundances"][norm_method]

                        # Metrics
                        combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                        phosphosite]["protein_oscillation_abundances"][norm_method]["metrics"] = calcAbundanceMetrics(
                                protein_oscillation_abundances, uniprot_accession)

                    # ANOVA
                    norm_abundances = combined_time_course_info[
                        uniprot_accession]["phosphorylation_abundances"][phosphosite][
                        "protein_oscillation_abundances"]["log2_mean"]

                    p_value, f_statistic = calcANOVA(norm_abundances)
                    combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                        phosphosite]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["ANOVA"] = {}
                    combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                        phosphosite]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["ANOVA"]["p_value"] = p_value
                    combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                        phosphosite]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["ANOVA"]["F_statistics"] = f_statistic

    # Fisher G Statistic - Phospho
    time_course_fisher_dict = calcFisherG(combined_time_course_info, "log2_mean", raw = False,  phospho = True, phospho_ab = True)

    # Corrected q values - Phospho
    # 1) Create a dataframe with the desired Protein-Phospho info
    prot_phospho_info = {}
    for uniprot_accession in combined_time_course_info:
        if len(combined_time_course_info[uniprot_accession]['phosphorylation_abundances']) != 0:
            for site in combined_time_course_info[uniprot_accession]['phosphorylation_abundances']:
                if 'protein_oscillation_abundances' in combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]:
                    phospho_key = uniprot_accession + "_" + combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]['phosphorylation_site']
                    if phospho_key not in prot_phospho_info:
                        prot_phospho_info[phospho_key] = {}
                    prot_phospho_info[phospho_key]['p_value'] = combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["ANOVA"]["p_value"]

    prot_phospho_info_df = pd.DataFrame(prot_phospho_info)
    prot_phospho_info_df = prot_phospho_info_df.T
    # 2) Protein-Phospho ANOVA q values
    prot_phospho_info_df['q_value'] = p_adjust_bh(prot_phospho_info_df['p_value'])
    # 3) Turn dataframe into a dictionary
    prot_phospho_info = prot_phospho_info_df.to_dict('index')
    # 4) Add Protein-Phospho info in combined_time_course_info dictionary
    for uniprot_accession in combined_time_course_info:
        if len(combined_time_course_info[uniprot_accession]["phosphorylation_abundances"]) != 0 and len(combined_time_course_info[uniprot_accession]["protein_abundances"]["raw"]) != 0:
            for site in combined_time_course_info[uniprot_accession]["phosphorylation_abundances"]:
                if "protein_oscillation_abundances" in combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][site]:
                    site_key = uniprot_accession + "_" + combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]['phosphorylation_site']
                    # ANOVA q values
                    if site_key in prot_phospho_info:
                        combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["ANOVA"]["q_value"] = prot_phospho_info[site_key]['q_value']
                    else:
                        combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["ANOVA"]["q_value"] = 1
                    # Fisher
                    if site_key in time_course_fisher_dict:
                        combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["Fisher_G"] = time_course_fisher_dict[site_key]
                    else:
                        combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]["protein_oscillation_abundances"]["log2_mean"]["metrics"]["Fisher_G"] = {'G_statistic': 1, 'p_value': 1, 'frequency': 1, 'q_value': 1}

    return combined_time_course_info

def addPhosphoRegression(combined_time_course_info):
    """
    Normalise the phospho abundance on the protein abundance
    Calculates and adds all the Regressed Phospho Abundance and their metrics for each phosphosite.
    Phospho = Dependent = Y
    Protein = Independent = X
    Linear Model => y = ax + b
    Residuals = Y - Y_predict 
    """
    logger.info("Adding Phospho Normalised on Protein Abundances - Regression")

    for uniprot_accession in combined_time_course_info:
        # If we have info both in protein and in phospho level
        if len(combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"]) != 0 and len(combined_time_course_info[uniprot_accession][
                "protein_abundances"]["raw"]) != 0:
            
            for phosphosite in combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"
            ]:  
                rep1_phosho = combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite]["position_abundances"]["normalised"]["log2_mean"]['abundance_rep_1']
                rep2_phosho = combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite]["position_abundances"]["normalised"]["log2_mean"]['abundance_rep_2']
                rep1_prot = combined_time_course_info[uniprot_accession]["protein_abundances"]["normalised"]["log2_mean"]['abundance_rep_1']
                rep2_prot = combined_time_course_info[uniprot_accession]["protein_abundances"]["normalised"]["log2_mean"]['abundance_rep_2']
                
                # If we don't have missing values in the protein and phospho abundances
                if len(rep1_phosho) != 8 or len(rep1_prot) != 8 or len(rep2_phosho) != 8 or len(rep2_prot) != 8:
                    continue
                
                # Add the Regressed Phospho Normalised Abundances
                combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite]["phospho_regression"] = {"0-max":{}, "log2_mean":{}}

                # converting dictionary values to list for both replicates
                phospho_Y_list = list(rep1_phosho.values()) + list(rep2_phosho.values())
                protein_X_list = list(rep1_prot.values()) + list(rep2_prot.values())
                # converting list to array
                prot_x = np.asarray(protein_X_list).reshape(len(protein_X_list), 1)
                phospho_y = np.asarray(phospho_Y_list).reshape(len(phospho_Y_list), 1)
                # Fit the linear model
                model = linear_model.LinearRegression().fit(prot_x,phospho_y)
                # Predict new Phospho Values
                y_pred = model.predict(prot_x)
                # Calculate Residuals
                residuals = (phospho_y - y_pred)
                # Create new regressed phospho abundances dictionaries
                replicates = ["abundance_rep_1", "abundance_rep_2"]
                res_dic = {}   
                for replicate in replicates:
                    res_dic[replicate] = {}
                    if replicate == "abundance_rep_1":
                        for index, value in enumerate(residuals[0:8]):
                            key = time_points[index]
                            res_dic[replicate][key] = value[0]
                    if replicate == "abundance_rep_2":
                        for index, value in enumerate(residuals[8::]):
                            key = time_points[index]
                            res_dic[replicate][key] = value[0]

                phospho_regression = combined_time_course_info[
                    uniprot_accession]["phosphorylation_abundances"][phosphosite][
                    "phospho_regression"]
                
                phospho_regression["log2_mean"] = res_dic

                phospho_regression["log2_mean"][
                        "abundance_average"
                    ] = calculateAverageRepAbundance(
                        phospho_regression["log2_mean"],
                        imputed=False,
                        phospho_oscillation=False,
                    )

                # Add metrics
                # Calculate the protein - phospho vector correlation
                phospho_regression["0-max"]['metrics'] = {}
                phospho_regression["0-max"]['metrics']["protein-phosho-correlation"] = stats.pearsonr(protein_X_list, phospho_Y_list)[0] # [0] to get the correlation coefficient, [1] = p-value
                # curve fold change phosphorylation/curve fold change protein for  0-max
                phospho_regression["0-max"]['metrics']["phosho-protein-cfc_ratio"] = combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite]['metrics']['0-max']['curve_fold_change'] / combined_time_course_info[uniprot_accession]['metrics']['0-max']['curve_fold_change']

                # ANOVA
                p_value, f_statistic = calcANOVA(phospho_regression["log2_mean"])
                phospho_regression["log2_mean"]['metrics'] = {}
                phospho_regression["log2_mean"]["metrics"]["ANOVA"] = {}
                combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite]["phospho_regression"]["log2_mean"]["metrics"]["ANOVA"]["p_value"] = p_value
                combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                    phosphosite]["phospho_regression"]["log2_mean"]["metrics"]["ANOVA"]["F_statistics"] = f_statistic

    # Fisher G Statistic - Phospho
    time_course_fisher_dict = calcFisherG(combined_time_course_info, "log2_mean", raw = False,  phospho = True, phospho_ab = False, phospho_reg = True)
    # Corrected q values - Phospho
    # 1) Create a dataframe with the desired regression info
    regression_info = {}
    for uniprot_accession in combined_time_course_info:
        if len(combined_time_course_info[uniprot_accession]['phosphorylation_abundances']) != 0:
            for site in combined_time_course_info[uniprot_accession]['phosphorylation_abundances']:
                if 'phospho_regression' in combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]:
                    phospho_key = uniprot_accession + "_" + combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]['phosphorylation_site']
                    if phospho_key not in regression_info:
                        regression_info[phospho_key] = {}
                    regression_info[phospho_key]['p_value'] = combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][site]['phospho_regression']['log2_mean']['metrics']['ANOVA']['p_value']

    regression_info_df = pd.DataFrame(regression_info)
    regression_info_df = regression_info_df.T
    # 2) Regression ANOVA q values
    regression_info_df['q_value'] = p_adjust_bh(regression_info_df['p_value'])
    # 3) Turn dataframe into a dictionary
    regression_info = regression_info_df.to_dict('index')
    # 4) Add Regression info in combined_time_course_info dictionary
    for uniprot_accession in combined_time_course_info:
        if len(combined_time_course_info[uniprot_accession]["phosphorylation_abundances"]) != 0 and len(combined_time_course_info[uniprot_accession]["protein_abundances"]["raw"]) != 0:
            for phosphosite in combined_time_course_info[uniprot_accession]["phosphorylation_abundances"]:
                if "phospho_regression" in combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][phosphosite]:
                    site_key = uniprot_accession + "_" + combined_time_course_info[uniprot_accession]['phosphorylation_abundances'][phosphosite]['phosphorylation_site']
                    # ANOVA q values
                    if site_key in regression_info:
                        combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][phosphosite]["phospho_regression"]["log2_mean"]["metrics"]["ANOVA"]["q_value"] = regression_info[site_key]['q_value']
                    else:
                        combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][phosphosite]["phospho_regression"]["log2_mean"]["metrics"]["ANOVA"]["q_value"] = 1
                    # Fisher
                    phospho_regression["log2_mean"]["metrics"]["Fisher_G"] = {}
                    if site_key in time_course_fisher_dict:
                        combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][phosphosite]["phospho_regression"]["log2_mean"]["metrics"]["Fisher_G"] = time_course_fisher_dict[site_key]
                    else:
                        combined_time_course_info[uniprot_accession]["phosphorylation_abundances"][
                                    phosphosite]["phospho_regression"]["log2_mean"]["metrics"]["Fisher_G"] = {'G_statistic': 1, 'p_value': 1, 'frequency': 1, 'q_value': 1}

    return combined_time_course_info

def combineTimeCoursePhosphoProteomicsReps():
    """
    Combines the phosphoproteomics data from all the TimeCourse replicates for every protein in the dataset in a final time_course_phospho_full dictionary.
    """
    time_course_phospho_full = {}
    time_course_phospho_reps = parseTimeCoursePhosphoProteomics()

    logger.info("Combining phospho abundance data from all TimeCourse Phosphoproteomics Replicates")

    # Raw
    for uniprot_accession in time_course_phospho_reps:
        if uniprot_accession not in time_course_phospho_full:
            time_course_phospho_full[uniprot_accession] = {"phosphorylation_abundances": {}}
            
        for mod_key in time_course_phospho_reps[uniprot_accession]["phosphorylation_abundances"]:
            if (
                mod_key
                not in time_course_phospho_full[uniprot_accession][
                    "phosphorylation_abundances"
                ]
            ):
                try:
                    mod = time_course_phospho_reps[uniprot_accession][
                        "phosphorylation_abundances"
                    ][mod_key]
                    phosphosite = mod["phosphorylation site"]
                    time_course_phospho_full[uniprot_accession][
                        "phosphorylation_abundances"
                    ][mod_key] = {
                        "phosphorylation_site": phosphosite,
                        "peptide_abundances": mod["peptide_abundances"],
                        "position_abundances": {
                            "raw": {'abundance_rep_1':{}, 'abundance_rep_2':{}},
                            "normalised": {"log2_mean": {}, "min-max": {}, "0-max": {}},
                            "imputed": {},
                        },
                        "confidence": time_course_phospho_reps[uniprot_accession][
                            "phosphorylation_abundances"
                        ][mod_key]["confidence"],
                    }

                    raw = time_course_phospho_full[uniprot_accession][
                        "phosphorylation_abundances"
                    ][mod_key]["position_abundances"]["raw"]

                    # Combine the abundances for a phosphosite from different modifications
                    for modification in mod["peptide_abundances"]["rep_1"]:
                        abundance_rep_1 = mod["peptide_abundances"]["rep_1"][modification]["abundance_rep_1"]
                        abundance_rep_2 = mod["peptide_abundances"]["rep_2"][modification]["abundance_rep_2"]

                        for k, v in abundance_rep_1.items():
                            cur_ab = 0
                            if k in raw["abundance_rep_1"]:
                                cur_ab += raw["abundance_rep_1"][k]

                            raw["abundance_rep_1"][k] = (cur_ab + abundance_rep_1[k])

                        for k, v in abundance_rep_2.items():
                            cur_ab = 0
                            if k in raw["abundance_rep_2"]:
                                cur_ab += raw["abundance_rep_2"][k]

                            raw["abundance_rep_2"][k] = (cur_ab + abundance_rep_2[k])

                except Exception as e:
                        print(e)
                        print(uniprot_accession, mod_key)

    # Normalisations
    rep_timepoint_median = calcColumnMedian(time_course_phospho_full,"TimeCourse_Phosphoproteomics_rep_1", phospho=time_course_phospho_full)

    for uniprot_accession in time_course_phospho_full:
        for mod_key in time_course_phospho_full[uniprot_accession][
            "phosphorylation_abundances"
        ]:
            try:
                raw = time_course_phospho_full[uniprot_accession][
                        "phosphorylation_abundances"][mod_key]["position_abundances"]["raw"]
                
                normalised = time_course_phospho_full[uniprot_accession][
                        "phosphorylation_abundances"][mod_key]["position_abundances"]["normalised"]
                
                imputed = time_course_phospho_full[uniprot_accession][
                        "phosphorylation_abundances"][mod_key]["position_abundances"]["imputed"]

                # First Level Normalisation - Normalise raw abundances by diving with column median
                first_norm_abundance_rep_1_2 = firstLevelNormalisationPhospho(raw, rep_timepoint_median)

                normalised["median"] = {}
                normalised["median"]["abundance_rep_1"] = first_norm_abundance_rep_1_2["abundance_rep_1"]
                normalised["median"]["abundance_rep_2"] = first_norm_abundance_rep_1_2["abundance_rep_2"]

                # Log2 - Palbo Normalisation
                normalised["log2_palbo"] = calclog2PalboNormalisation(first_norm_abundance_rep_1_2)

                # Log2 - substract row mean for every batch seperately
                log2_norm_abundance_reps = calclog2RelativeAbundance(first_norm_abundance_rep_1_2)
                norm_abundance_rep_1 = log2_norm_abundance_reps["rep_1"]
                norm_abundance_rep_2 = log2_norm_abundance_reps["rep_2"]
            
                normalised["log2_mean"]["abundance_rep_1"] = norm_abundance_rep_1
                normalised["log2_mean"]["abundance_rep_2"] = norm_abundance_rep_2
                
                # min-max
                normalised["min-max"]["abundance_rep_1"] = normaliseData(first_norm_abundance_rep_1_2["abundance_rep_1"])
                normalised["min-max"]["abundance_rep_2"] = normaliseData(first_norm_abundance_rep_1_2["abundance_rep_2"])

                # 0-max
                normalised["0-max"]["abundance_rep_1"] = normaliseData(first_norm_abundance_rep_1_2["abundance_rep_1"], zero_min=True)
                normalised["0-max"]["abundance_rep_2"] = normaliseData(first_norm_abundance_rep_1_2["abundance_rep_2"], zero_min=True)

                # Imputed

                imputed["abundance_rep_1"] = imputeData(normalised["min-max"]["abundance_rep_1"])
                
                imputed["abundance_rep_2"] = imputeData(normalised["min-max"]["abundance_rep_2"])

                # Averages
                raw["abundance_average"] = calculateAverageRepAbundance(raw, phospho=True)

                normalised["min-max"]["abundance_average"] = calculateAverageRepAbundance(
                    normalised["min-max"], phospho=True, norm=True)
                
                normalised["0-max"]["abundance_average"] = calculateAverageRepAbundance(
                    normalised["0-max"], phospho=True, norm=True)
                
                normalised["log2_mean"]["abundance_average"] = calculateAverageRepAbundance(
                    normalised["log2_mean"], phospho=True, norm=True)
                
                normalised["median"]["abundance_average"] = calculateAverageRepAbundance(normalised["median"], phospho=False, norm=True)
                
                normalised["log2_palbo"]["abundance_average"] = calculateAverageRepAbundance(normalised["log2_palbo"], phospho=True, norm=True)
                
                imputed["abundance_average"] = calculateAverageRepAbundance(imputed, imputed=True, phospho=True)

            except Exception as e:
                print("this is a problem")
                print(uniprot_accession)
                print(traceback.format_exc())
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

    # Metrics
    time_course_phospho_full = addPhosphoMetrics(time_course_phospho_full)

    # Kinase Consensus Prediction
    for uniprot_accession in time_course_phospho_full:
        for phosphosite in time_course_phospho_full[uniprot_accession]["phosphorylation_abundances"]:
            # only for certain phosphorylation sites
            if phosphosite.find("-") == -1:
                # Add PepTools Phospho annotations
                if 'phospho' in index_protein_names[uniprot_accession]:
                    if phosphosite in index_protein_names[uniprot_accession]['phospho']:
                        time_course_phospho_full[uniprot_accession]["phosphorylation_abundances"][phosphosite]['PepTools_annotations'] = index_protein_names[uniprot_accession]['phospho'][phosphosite]
                time_course_phospho_full[uniprot_accession]["phosphorylation_abundances"][phosphosite]['kinase_prediction'] = {}
                phospho_kinases_class = getConsensusKinasePred(uniprot_accession, time_course_phospho_full[uniprot_accession]["phosphorylation_abundances"][phosphosite]['phosphorylation_site'])
                time_course_phospho_full[uniprot_accession]["phosphorylation_abundances"][phosphosite]['kinase_prediction']['peptide_seq'] = phospho_kinases_class['peptide_seq']
                time_course_phospho_full[uniprot_accession]["phosphorylation_abundances"][phosphosite]['kinase_prediction']['consenus_motif_match'] = phospho_kinases_class['kinase_motif_match']

    return time_course_phospho_full

def createCombinedProteomeandPhosphoFile():
    """
    Creates a file that stores all the info for the TimeCourse Proteome and PhosphoProteme experiment.
    It combines and stores the 2 proteomics and the 2 phosphoproteomics replicates along with the normalised 
    values of the protein/phosphoprotein abundances.
    """
    # Add the Protein Data
    combined_time_course_info = createTimeCourseProteome()

    # Add the Phospho Data
    time_course_phospho = combineTimeCoursePhosphoProteomicsReps()

    for uniprot_accession in time_course_phospho:
        if uniprot_accession in combined_time_course_info:
            combined_time_course_info[uniprot_accession][
                "phosphorylation_abundances"
            ] = time_course_phospho[uniprot_accession]["phosphorylation_abundances"]
        else:
            if uniprot_accession in index_protein_names:
                gene_name = index_protein_names[uniprot_accession]["gene_name"]
                protein_name = index_protein_names[uniprot_accession][
                    "protein_name"
                ]
            else:
                gene_name, protein_name = getProteinInfo(uniprot_accession)
            combined_time_course_info[uniprot_accession] = {
                "gene_name": gene_name,
                "protein_name": protein_name,
                "protein_abundances": {"raw": {}, "normalised": {}, "imputed": {}},
                "phosphorylation_abundances": time_course_phospho[uniprot_accession][
                    "phosphorylation_abundances"
                ],
            }

    # Add the Protein Oscillation Normalised Abundances        
    combined_time_course_info = addProteinOscillations(combined_time_course_info)

    # Add the Regressed Phospho Normalised Abundances        
    combined_time_course_info = addPhosphoRegression(combined_time_course_info)
            
    print(len(combined_time_course_info))

    with open("TimeCourse_Full_info.json", "w") as outfile:
        json.dump(combined_time_course_info, outfile)    

    return combined_time_course_info



def main():
    createCombinedProteomeandPhosphoFile()

if __name__ == "__main__":
    main()