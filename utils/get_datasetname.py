#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import os

def get_datasetname(datapath):
    """Maps Openml task number to dataset name

    Parameters
    ----------
    datapath : path to file where dataset resides

    Returns
    -------
    dict mapping from openml task to dataset name

    """

    task_no = os.path.basename(datapath)
    dataset_dict = {"75103": "sylva_agnostic", "2117": "adult", "2122": "kropt",
                    "236": "letter", "262":"pendigits", "75093":"jm1",
                    "75097": "amazon_employee_access", "75098": "mnist_784",
                    "75101": "higgs", "75105": "KDDCup09_appetency",
                    "75106": "KDDCup09_churn", "75107": "KDDCup09_upselling",
                    "75110": "chess", "75112": "MagicTelescope",
                    "75113": "sylva_prior", "75181":"tamilnadu-electricity",
                    "75191":"vehicle_sensIT", "75205":"ohscal.wc",
                    "75215":"PhishingWebsites", "75219":"eeg-eye-state",
                    "75223": "kr-vs-k", "75243":"nursery"
                    }

    return dataset_dict[task_no]





