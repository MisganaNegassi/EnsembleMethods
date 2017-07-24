import json
import seaborn as sns
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

d = defaultdict(list)

with open("./data/output.jl", "r") as f:
    for line in f:
        item = json.loads(line)
        dataset = list(item.keys())[0]
        values = item[dataset]

        d[dataset].append(values)

tmp = []
scaler = MinMaxScaler()

for dataset in sorted(d):
    vals = d[dataset]

    def filter_models(condition, models):
        return {model for model in models if condition(model)}

    # condition = lambda model: not(model.startswith("PRUN") or model.startswith("BOTH") or model.startswith("BAG")) #  plain models (ES, ...)
    # condition = lambda model: not(model.endswith("REP"))
    #condition = lambda model: not(model.endswith("REP"))
    #condition = lambda _: True
    condition = lambda model: model not in ("BAG_ST_MLR", "BOTH_ST_MLR", "BOTH_ST_MLR", "PRUN_ST_MLR", "ST_MLR", "BAG_ST_MLR") # remove NaN models
    condition2 = lambda model: not(model.startswith("PRUN") or model.startswith("BOTH") or model.startswith("BAG")) #  plain models (ES, ...)

    conditions = (
            condition,

                 )

    models = filter_models(lambda model: all(condition(model) for condition in conditions),
                           set(list(val.keys())[0] for val in vals))

    models = sorted(models)

    """
    models = {model for model in set(list(val.keys())[0] for val in vals)
              if not(model.startswith("PRUN") or model.startswith("BOTH") or
                  model.startswith)}
    """

    #models = set(list(val.keys())[0] for val in vals)

    metrics = set()
    for val in vals:
        model = list(val.keys())[0]
        metric = list(val[model].keys())[0]
        metrics.add(metric)

    metric_map = defaultdict(list)
    metrics = sorted(metrics)

    for metric in metrics:
        for m in models:
            total, cnt = 0., 0
            for val in vals:
                model = list(val.keys())[0]
                met = list(val[model].keys())[0]
                if m == model and metric == met:
                    total += val[model][met]
                    cnt += 1
            try:
                res = total / cnt
                metric_map[metric].append((m, round(res, 3)))
            except ZeroDivisionError:
                # Value not available, replace with "nan"
                metric_map[metric].append((m, float('nan')))

        metric_map[metric] = dict(metric_map[metric])

    rows = [[] for _ in models]

    for i, model in enumerate(models):
        for metric in metrics:
            rows[i].append(metric_map[metric][model])

    df = pd.DataFrame.from_records(rows, index=[m.replace("_", " ") for m in models], columns=metrics)

    df_norm = (df - df.min()) / (df.max() - df.min())

    # generate latex file with table of results for each dataset
    with open("./{dataset}.tex".format(dataset=dataset), "w") as f:
        f.write(df_norm.to_latex())

    """
    tmp.append(df)

# to generate box plots over datasets

df_2 = pd.DataFrame(columns=["models"] + list(metrics))
lst = []
for df in tmp:
    df_norm = (df - df.min()) / (df.max() - df.min())
    lst.append(df_norm)


df = pd.concat(lst)


cnt=0
for index, row in df.iterrows():
    a = [index] + list(row.values)
    df_2.loc[cnt] = a
    cnt+=1

met = "f1"
ranks = df_2.groupby("models")[met].median().fillna(0).sort_values()[::-1].index

print(type(ranks))
fig = plt.figure(figsize=[15,10])
# We define a fake subplot that is in fact only the plot.
plot = fig.add_subplot(111)

# We change the fontsize of minor ticks label
plot.tick_params(axis='both', which='major', labelsize=15)
sns.boxplot(data=df_2, x=met, y="models", order=ranks)
plt.xlabel("normalized " + met, fontsize=15)
plt.ylabel("")
#sns.stripplot(data=df_2, y="models", x=met, size=3, jitter= True, color = ".3", order=ranks)
plt.tight_layout()
plt.savefig("./%s_test_plot.pdf" %met, dpi=300)
"""
