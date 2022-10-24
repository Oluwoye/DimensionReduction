import os
import pandas as pd
import numpy as np

import scipy.stats as stats
from statsmodels.stats.descriptivestats import sign_test
import matplotlib.pyplot as plt
from common import normalize_input


def postprocess_csv_result_file_experiment_one(result_file, project_column="dataset_name"):
    espadoto_metrics = ["T_train", "C_train", "R_train", "S_train", "N_train"]
    cluster_metrics = ["ha_train", "sh_train", "d_train", "sdbw_train"]
    all_metrics = espadoto_metrics + cluster_metrics
    projects = ["mnist", "fashionmnist", "har", "reuters", "20_newsgroups_tfidf", "ag_news_tfidf", "hatespeech_tfidf",
                "imdb_tfidf", "sms_spam_tfidf"]
    parameters_columns = ["num_epoch", "n_cluster", "patience", "min_delta"]
    result_dir = "evaluation_experiment1"
    os.makedirs(result_dir, exist_ok=True)

    results_df = pd.read_csv(result_file)
    evaluate_results(all_metrics, cluster_metrics, espadoto_metrics, parameters_columns, project_column, projects,
                     result_dir, result_file, results_df)


def evaluate_results(all_metrics, cluster_metrics, espadoto_metrics, parameters_columns, project_column, projects,
                     result_dir, result_file, results_df, min_null=None):
    if min_null is None:
        min_null = [0.1, 0.05, 0.01]
    top_rows = []
    for project in projects:
        project_df = results_df[results_df[project_column] == project]
        for metric in all_metrics:
            metric_values = project_df[metric].to_numpy()
            if metric in cluster_metrics:
                metric_values = np.absolute(metric_values)
            metric_values = normalize_input(metric_values)
            if metric in cluster_metrics:
                metric_values = 1. - metric_values
            max_pos = np.argpartition(metric_values, -5)[-5:]
            max_pos = max_pos[np.argsort(metric_values[max_pos])]
            max_row = project_df.iloc[max_pos].to_numpy()
            max_row = max_row.tolist()
            for i, row in enumerate(max_row):
                row.append("Max " + str(5 - i) + " for " + project + " and metric " + metric)
                top_rows.append(row)

        summed_metrics = None
        for metric in espadoto_metrics:
            metric_values = project_df[metric].to_numpy()
            metric_values = normalize_input(metric_values)
            if summed_metrics is not None:
                summed_metrics += metric_values
            else:
                summed_metrics = metric_values
        max_pos = np.argpartition(summed_metrics, -5)[-5:]
        max_pos = max_pos[np.argsort(summed_metrics[max_pos])]
        max_row = project_df.iloc[max_pos].to_numpy()
        max_row = max_row.tolist()
        for i, row in enumerate(max_row):
            row.append("Max " + str(5 - i) + " for " + project + " and sum of espadoto metrics")
            top_rows.append(row)

        summed_metrics_2 = None
        for metric in cluster_metrics:
            metric_values = project_df[metric].to_numpy()
            metric_values = np.absolute(metric_values)
            metric_values = normalize_input(metric_values)
            metric_values = 1. - metric_values
            if summed_metrics_2 is not None:
                summed_metrics_2 += metric_values
            else:
                summed_metrics_2 = metric_values
        max_pos = np.argpartition(summed_metrics_2, -5)[-5:]
        max_pos = max_pos[np.argsort(summed_metrics_2[max_pos])]
        max_row = project_df.iloc[max_pos].to_numpy()
        max_row = max_row.tolist()
        for i, row in enumerate(max_row):
            row.append("Max " + str(5 - i) + " for " + project + " and sum of cluster metrics")
            top_rows.append(row)

        summed_total = summed_metrics + summed_metrics_2
        max_pos = np.argpartition(summed_total, -5)[-5:]
        max_pos = max_pos[np.argsort(summed_total[max_pos])]
        max_row = project_df.iloc[max_pos].to_numpy()
        max_row = max_row.tolist()
        for i, row in enumerate(max_row):
            row.append("Max " + str(5 - i) + " for " + project + " and sum of all metrics")
            top_rows.append(row)
    columns = results_df.columns.tolist()
    columns.append("Criterion")
    max_df = pd.DataFrame(data=np.array(top_rows), columns=columns)
    max_df.to_csv(os.path.join(result_dir, "evaluation_" + result_file.split(os.sep)[-1]))
    statistic_test_values = []
    for metric in all_metrics:
        metric_values = results_df[metric].to_numpy()
        if metric in cluster_metrics:
            metric_values = np.nan_to_num(metric_values, nan=max(abs(np.min(metric_values)), np.max(metric_values)))
        else:
            metric_values = np.nan_to_num(metric_values)
        if metric in cluster_metrics:
            metric_values = np.absolute(metric_values)
            metric_values = normalize_input(metric_values)
            metric_values = 1. - metric_values
        else:
            metric_values = normalize_input(metric_values)
        stats.probplot(metric_values, dist="norm", plot=plt)
        plt.title("Probability Plot - " + metric)
        plt.savefig(os.path.join(result_dir, metric + "_q_q_plot.png"))
        plt.close()
        for parameter in parameters_columns:
            parameter_values = results_df[parameter].to_numpy()
            try:
                float(parameter_values[0])
            except ValueError:
                unique = np.unique(parameter_values)
                mapping = {value: i for i, value in enumerate(unique)}
                parameter_values = [mapping[value] for value in parameter_values]
            nan_indices = np.argwhere(np.isnan(parameter_values))
            metric_values = [metric_value for i, metric_value in enumerate(metric_values) if i not in nan_indices]
            parameter_values = [parametric_value for i, parametric_value in enumerate(parameter_values)
                                if i not in nan_indices]
            parameter_values = np.absolute(parameter_values)
            parameter_values = normalize_input(parameter_values)
            parameter_values = parameter_values - np.mean(parameter_values)
            metric_values = metric_values - np.mean(metric_values)
            try:
                differences = parameter_values - metric_values
            except:
                continue
            statistic_test_values.append([metric, parameter,
                                          stats.wilcoxon(x=metric_values, y=parameter_values).pvalue,
                                          stats.mannwhitneyu(x=metric_values, y=parameter_values).pvalue,
                                          sign_test(differences)[1]
                                          ])
    statistic_test_values = pd.DataFrame(data=statistic_test_values,
                                         columns=["Metric", "Parameter", "Wilcoxon_statistic_p_value",
                                                  "Mann_Whitney_u_test_p_value", "Sign_test_for_differences_p_value"])
    statistic_test_values.to_csv(os.path.join(result_dir, "statistical_" + result_file.split(os.sep)[-1]))

    wilcoxon_values = statistic_test_values["Wilcoxon_statistic_p_value"].to_numpy()
    mann_values = statistic_test_values["Mann_Whitney_u_test_p_value"].to_numpy()
    sign_values = statistic_test_values["Sign_test_for_differences_p_value"].to_numpy()

    for cut_off_value in min_null:
        over = np.intersect1d([i for i, value in enumerate(wilcoxon_values) if value > cut_off_value],
                              [i for i, value in enumerate(mann_values) if value > cut_off_value])
        over = np.intersect1d(over, [i for i, value in enumerate(sign_values) if value > cut_off_value])
        null_df = statistic_test_values.iloc[over]
        null_df.to_csv(os.path.join(result_dir, "null_hypothesis_uphold_" + str(cut_off_value) + "_" +
                                    result_file.split(os.sep)[-1]))


def posprocess_csv_result_architecture_experiment(base_path, csv_files, project_column="dataset_name"):
    espadoto_metrics = ["T_train", "C_train", "R_train", "S_train", "N_train"]
    cluster_metrics = ["ha_train", "sh_train", "d_train", "sdbw_train"]
    all_metrics = espadoto_metrics + cluster_metrics
    projects = ["mnist", "fashionmnist", "har", "reuters"]
    parameters_columns = ["num_epoch", "n_cluster", "min_delta", "optimizer", "bottleneck_activation",
                          "layer_activation", "layer_initializer", "bias", "l1_regularizer", "l2_regularizer",
                          "model_mode"]
    result_dir = "evaluation_experiment2_architecture"
    os.makedirs(result_dir, exist_ok=True)

    input_dfs = []
    for file in csv_files:
        input_dfs.append(pd.read_csv(os.path.join(base_path, file)))
    results_df = pd.concat(input_dfs)

    evaluate_results(all_metrics, cluster_metrics, espadoto_metrics, parameters_columns, project_column, projects,
                     result_dir, "metrics_model_architecture.csv", results_df)


def main():
    base_path = "/home/tim/Dokumente/IVAPP2023/results_direct"
    csv_files = sorted([file for file in os.listdir(base_path) if file.endswith(".csv")])
    csv_files.reverse()
    postprocess_csv_result_file_experiment_one(os.path.join(base_path, csv_files[0]))

    base_path_11 = "/home/tim/Dokumente/IVAPP2023/results_direct_experiment11"
    csv_files = sorted([file for file in os.listdir(base_path_11) if file.endswith(".csv")])
    csv_files.reverse()
    posprocess_csv_result_architecture_experiment(base_path_11, csv_files)


if __name__ == "__main__":
    main()
