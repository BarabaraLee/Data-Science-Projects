## Description of the AB Test

* Channel KLMN runs a TV commercial featuring the Mayor of Los Angeles nationwide to promote its political talk show "US Politics This Week" and it is found this commercial only works for people residing in Los Angeles. 
* The Executive Producer of “US Politics This Week” suggested to add commercials featuring Mayors of local cities to the existing TV commercial (featuring the Mayor of Los Angeles).
* After launching the new commercials, along with the old commercial, an AB Test found that: a lower fraction of people who saw the new commercials watched “US Politics This Week” as compared to people who saw the old commercial.

## Guidelines

* Reproduce the negative result found above. Is it actually negative?
* Explain what might be happening. Are the commercials with local Mayors really driving a lower fraction of people to watch the show?
* If you found something wrong with the experiment, design an algorithm that returns FALSE if the problem happens again in the future. If you didn’t find anything wrong, what is your recommendation to the Executive Producer regarding whether or not they should continue airing the new commercials?


```python
%matplotlib notebook
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

from statsmodels.stats import proportion
from statsmodels.stats.proportion import proportions_ztest

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings('ignore')

plt.rc('figure', figsize=(3.0, 3.0))
```

### 1. Read Data


```python
viewer_data = pd.read_csv("../dataset/viewer_data.csv")
test_data = pd.read_csv("../dataset/test_data.csv", parse_dates=["date"])
```


```python
viewer_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>viewer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1918165</td>
      <td>Female</td>
      <td>39</td>
      <td>Dallas</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27662619</td>
      <td>Female</td>
      <td>28</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5493662</td>
      <td>Female</td>
      <td>53</td>
      <td>Detroit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14441247</td>
      <td>Male</td>
      <td>41</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25595927</td>
      <td>Male</td>
      <td>53</td>
      <td>Seattle</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>viewer_id</th>
      <th>date</th>
      <th>tv_make</th>
      <th>tv_size</th>
      <th>uhd_capable</th>
      <th>tv_provider</th>
      <th>total_time_watched</th>
      <th>watched</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24726768</td>
      <td>2018-01-16</td>
      <td>Sony</td>
      <td>70</td>
      <td>0</td>
      <td>Comcast</td>
      <td>10.75</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25001464</td>
      <td>2018-01-18</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.75</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28291998</td>
      <td>2018-01-18</td>
      <td>Sony</td>
      <td>50</td>
      <td>1</td>
      <td>Dish Network</td>
      <td>20.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17057157</td>
      <td>2018-01-19</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>Comcast</td>
      <td>1.50</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29504447</td>
      <td>2018-01-17</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>Comcast</td>
      <td>17.50</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Exploratory Data Analysis (EDA)


```python
pp.ProfileReport(viewer_data)
```




<meta charset="UTF-8">

<style>

        .variablerow {
            border: 1px solid #e1e1e8;
            border-top: hidden;
            padding-top: 2em;
            padding-bottom: 2em;
            padding-left: 1em;
            padding-right: 1em;
        }

        .headerrow {
            border: 1px solid #e1e1e8;
            background-color: #f5f5f5;
            padding: 2em;
        }
        .namecol {
            margin-top: -1em;
            overflow-x: auto;
        }

        .dl-horizontal dt {
            text-align: left;
            padding-right: 1em;
            white-space: normal;
        }

        .dl-horizontal dd {
            margin-left: 0;
        }

        .ignore {
            opacity: 0.4;
        }

        .container.pandas-profiling {
            max-width:975px;
        }

        .col-md-12 {
            padding-left: 2em;
        }

        .indent {
            margin-left: 1em;
        }

        .center-img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block;
        }

        /* Table example_values */
            table.example_values {
                border: 0;
            }

            .example_values th {
                border: 0;
                padding: 0 ;
                color: #555;
                font-weight: 600;
            }

            .example_values tr, .example_values td{
                border: 0;
                padding: 0;
                color: #555;
            }

        /* STATS */
            table.stats {
                border: 0;
            }

            .stats th {
                border: 0;
                padding: 0 2em 0 0;
                color: #555;
                font-weight: 600;
            }

            .stats tr {
                border: 0;
            }

            .stats td{
                color: #555;
                padding: 1px;
                border: 0;
            }


        /* Sample table */
            table.sample {
                border: 0;
                margin-bottom: 2em;
                margin-left:1em;
            }
            .sample tr {
                border:0;
            }
            .sample td, .sample th{
                padding: 0.5em;
                white-space: nowrap;
                border: none;

            }

            .sample thead {
                border-top: 0;
                border-bottom: 2px solid #ddd;
            }

            .sample td {
                width:100%;
            }


        /* There is no good solution available to make the divs equal height and then center ... */
            .histogram {
                margin-top: 3em;
            }
        /* Freq table */

            table.freq {
                margin-bottom: 2em;
                border: 0;
            }
            table.freq th, table.freq tr, table.freq td {
                border: 0;
                padding: 0;
            }

            .freq thead {
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;

            }

            td.fillremaining{
                width:auto;
                max-width: none;
            }

            td.number, th.number {
                text-align:right ;
            }

        /* Freq mini */
            .freq.mini td{
                width: 50%;
                padding: 1px;
                font-size: 12px;

            }
            table.freq.mini {
                 width:100%;
            }
            .freq.mini th {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 5em;
                font-weight: 400;
                text-align:right;
                padding-right: 0.5em;
            }

            .missing {
                color: #a94442;
            }
            .alert, .alert > th, .alert > td {
                color: #a94442;
            }


        /* Bars in tables */
            .freq .bar{
                float: left;
                width: 0;
                height: 100%;
                line-height: 20px;
                color: #fff;
                text-align: center;
                background-color: #337ab7;
                border-radius: 3px;
                margin-right: 4px;
            }
            .other .bar {
                background-color: #999;
            }
            .missing .bar{
                background-color: #a94442;
            }
            .tooltip-inner {
                width: 100%;
                white-space: nowrap;
                text-align:left;
            }

            .extrapadding{
                padding: 2em;
            }

            .pp-anchor{

            }

</style>

<div class="container pandas-profiling">
    <div class="row headerrow highlight">
        <h1>Overview</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-6 namecol">
        <p class="h4">Dataset info</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Number of variables</th>
                <td>4 </td>
            </tr>
            <tr>
                <th>Number of observations</th>
                <td>417464 </td>
            </tr>
            <tr>
                <th>Total Missing (%)</th>
                <td>0.0% </td>
            </tr>
            <tr>
                <th>Total size in memory</th>
                <td>12.7 MiB </td>
            </tr>
            <tr>
                <th>Average record size in memory</th>
                <td>32.0 B </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-6 namecol">
        <p class="h4">Variables types</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Numeric</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Categorical</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Boolean</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Date</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Text (Unique)</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Rejected</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Unsupported</th>
                <td>0 </td>
            </tr>
            </tbody>
        </table>
    </div>
</div>
    <div class="row headerrow highlight">
        <h1>Variables</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_viewer_id">viewer_id<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>417464</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>100.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>18379000</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>10000</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>39999921</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-3713796977191528543">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAACxklEQVR4nO3dO0sjYRiG4cfVQrH20GipnaIE/AFiWIiNELQJ/gVRLITthC1sQto0goUgNoIHWDsRkRE2aW1FsQliaWP03S5s1vCgYg4s9wUpMvkmvM3NN4Fh0hURIQANfWv3AEAn62n3AP9K/fj14XN%2B//zehEkAdhDAIhDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDA6LiHNnzGZx70gM7UaQ/gYAcBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBjK6IiHYPAXQqdhDAIBDAIBDAIBA0xePjo%2Bbm5nR1dfWu9ZlMRlNTU3Wv8fFxFYvFJk/q/Rd/f4DOUiqVtLGxodvb23efc3JyUve%2BUCjo7OxMuVzuq8f7EHYQfKmDgwOtr69rdXX1zWeXl5fKZrNKpVLKZDI6PDxs%2BB1JkmhnZ0eFQkH9/f3NHtkL4AtVKpV4fn6OiIixsbFIkiQiIq6vr2NiYiJOT0%2BjWq1GqVSKmZmZOD8/rzu/Wq1GOp2OYrHY8tkbYQfBlxoYGFBPz9sr9729Pc3OziqdTqu7u1vT09NaXFzU7u5u3bqjoyM9PT1peXm5VSNb/AZBS9zf3ytJEqVSqdqxl5cXjY6O1q3b39/X0tKSent7Wz1iQwSClhgeHtbCwoI2NzdrxyqViuKvGzkeHh5ULpe1tbXVjhEb4hILLZHNZnV8fKyLiwu9vr7q5uZGuVxO29vbtTXlclmDg4MaGRlp46T12EHQEpOTk8rn88rn81pZWVFfX5/m5%2Be1trZWW3N3d6ehoaE2TvkWNysCBpdYgEEggEEggEEggEEggEEggEEggEEggEEggEEggEEggEEggPEHBkmQLA1A3bEAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-3713796977191528543,#minihistogram-3713796977191528543"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-3713796977191528543">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-3713796977191528543"
                                                  aria-controls="quantiles-3713796977191528543" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-3713796977191528543" aria-controls="histogram-3713796977191528543"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-3713796977191528543" aria-controls="common-3713796977191528543"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-3713796977191528543" aria-controls="extreme-3713796977191528543"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-3713796977191528543">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>10000</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>45723</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>7571100</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>18416000</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>29179000</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>37851000</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>39999921</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>39989921</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>21608000</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>12475000</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.67879</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.2393</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>18379000</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>10829000</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.01496</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>7672605290054</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>155640000000000</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>3.2 MiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-3713796977191528543">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA4qklEQVR4nO3dfVRV153/8Y9wVS4icgmCtkuXU4GkaUwlEBFNTYMhJvGhFDGslKY1bbUDNlYnomml1dGAWlPHUEdWRseSJs6KFccxJCY17aSRGkWSWE3SasFmlJSlPAjKo/Jwfn/kx129ovFit3Lvyfu1Fivr7r3PPvt7t%2BZ%2BPOcAAyzLsgQAAABjAvp7AQAAAHZDwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhjn6ewGfFbW1TcbnDAgYoPDwITp3rkXd3Zbx%2BfuLXeuS7FubXeuSqM0f2bUuyb613ci6hg8fanQ%2Bb3EFy48FBAzQgAEDFBAwoL%2BXYpRd65LsW5td65KozR/ZtS7JvrXZsS4CFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhPhmwXn75ZcXFxXl83XHHHbrjjjskSUePHtWcOXMUFxen5ORk7dy50%2BP43bt3KyUlRePHj1daWpqOHDni7uvq6tK6des0adIkxcXFKSsrSzU1Ne7%2B%2Bvp6ZWdnKyEhQYmJicrLy1NnZ%2BfNKRwAANiCTwasWbNm6ciRI%2B6v119/XWFhYcrLy9P58%2Bc1f/58paamqry8XHl5eVqzZo2OHTsmSSorK9Pq1au1du1alZeXa9asWcrKylJbW5skqbCwUAcOHNCuXbtUWlqqoKAg5ebmus%2B9aNEiBQcHq7S0VMXFxTp48KCKior6420AAAB%2ByicD1t%2BzLEs5OTn66le/qq997Wvat2%2BfwsLClJmZKYfDoaSkJM2cOVPbt2%2BXJO3cuVPTp09XfHy8Bg4cqLlz58rlcmnv3r3u/nnz5mnkyJEKCQnR8uXLtX//flVVVenUqVM6fPiwcnJy5HQ6NWrUKGVnZ7vnBgAA8IajvxdwLXv27FFlZaU2b94sSaqoqFBsbKzHmOjoaBUXF0uSKisrNXv27F79x48fV1NTk86cOeNxfEREhIYNG6YTJ05IksLCwhQVFeXuHzt2rKqrq3XhwgWFhoZ6teaamhrV1tZ6tDkcwYqMjPSyau8EBgZ4/Ncu7FqXZN/a7FqXRG3%2ByK51SfatzY51%2BXTA6u7uVmFhof75n/9ZISEhkqSWlhY5nU6PcUFBQWptbb1mf0tLiyQpODi4V39P3%2BXH9rxubW31OmDt2LFDmzZt8mhbsGCBFi5c6NXxfZGw/HXjc94o7%2BQ92KfxoaHOaw/yU3atza51SdTmj%2Bxal2Tf2uxUl08HrLKyMtXU1Cg9Pd3d5nQ61dTU5DGuvb1dQ4YMcfe3t7f36ne5XO6w1PM81uXHW5bVq6/ndc/83sjIyFBycrJHm8MRrIaGFq/n8Ia/JX1v6w8MDFBoqFMXLrSpq6v7Bq/q5rJrbXatS6I2f2TXuiT71nYj63K5vP/8NsmnA9ZvfvMbpaSkeFxxio2N1YEDBzzGVVZWKiYmRpIUExOjioqKXv1TpkzRsGHDFBUVpcrKSvdtwtraWjU2Nio2Nlbd3d1qbGxUXV2dIiIiJEknT57UiBEjNHToUK/XHRkZ2et2YG1tkzo77fOX4Xr0tf6urm7bvmd2rc2udUnU5o/sWpdk39rsVJdPXwJ59913dffdd3u0paSkqK6uTkVFRero6NChQ4dUUlLifu4qPT1dJSUlOnTokDo6OlRUVKT6%2BnqlpKRIktLS0lRYWKiqqio1NzcrPz9fEyZM0OjRozVmzBjFx8crPz9fzc3Nqqqq0ubNmz2uoAEAAFyLT1/B%2Bvjjj3tdCXK5XNq2bZvy8vJUUFCg8PBw5ebmauLEiZKkpKQkrVixQitXrtTZs2cVHR2tLVu2KCwsTNInz0J1dnYqMzNTLS0tSkxM1MaNG93zFxQUaNWqVZo6daoCAgKUmpqq7Ozsm1UyAACwgQGWZVn9vYjPgtrapmsP6iOHI0Apz5Qan/dGeW3RZK/GORwBcrmGqKGhxTaXinvYtTa71iVRmz%2Bya12SfWu7kXUNH%2B79Iz4m%2BfQtQgAAAH9EwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAY5rMBq7GxUUuXLlViYqLuvvtuZWdnq6amRpJ09OhRzZkzR3FxcUpOTtbOnTs9jt29e7dSUlI0fvx4paWl6ciRI%2B6%2Brq4urVu3TpMmTVJcXJyysrLc80pSfX29srOzlZCQoMTEROXl5amzs/PmFA0AAGzBZwPWE088odbWVr3xxht68803FRgYqJ/85Cc6f/685s%2Bfr9TUVJWXlysvL09r1qzRsWPHJEllZWVavXq11q5dq/Lycs2aNUtZWVlqa2uTJBUWFurAgQPatWuXSktLFRQUpNzcXPd5Fy1apODgYJWWlqq4uFgHDx5UUVFRf7wFAADAT/lkwPrggw909OhRrV27VqGhoQoJCdHq1au1ZMkS7du3T2FhYcrMzJTD4VBSUpJmzpyp7du3S5J27typ6dOnKz4%2BXgMHDtTcuXPlcrm0d%2B9ed/%2B8efM0cuRIhYSEaPny5dq/f7%2Bqqqp06tQpHT58WDk5OXI6nRo1apSys7PdcwMAAHjDJwPWsWPHFB0drV//%2BtdKSUnRPffco3Xr1mn48OGqqKhQbGysx/jo6GgdP35cklRZWXnV/qamJp05c8ajPyIiQsOGDdOJEydUUVGhsLAwRUVFufvHjh2r6upqXbhw4QZWDAAA7MTR3wu4kvPnz%2BvEiRO64447tHv3brW3t2vp0qVatmyZIiIi5HQ6PcYHBQWptbVVktTS0nLV/paWFklScHBwr/6evsuP7Xnd2tqq0NBQr9ZfU1Oj2tpajzaHI1iRkZFeHe%2BtwECfzMdX5XB4t96euvytPm/YtTa71iVRmz%2Bya12SfWuzY10%2BGbAGDRokSVq%2BfLkGDx6skJAQLVq0SI888ojS0tLU3t7uMb69vV1DhgyR9EkgulK/y%2BVyh6We57EuP96yrF59Pa975vfGjh07tGnTJo%2B2BQsWaOHChV7PYUcul/fvoSSFhjqvPchP2bU2u9YlUZs/smtdkn1rs1NdPhmwoqOj1d3drY6ODg0ePFiS1N3dLUn64he/qP/6r//yGF9ZWamYmBhJUkxMjCoqKnr1T5kyRcOGDVNUVJTHbcTa2lo1NjYqNjZW3d3damxsVF1dnSIiIiRJJ0%2Be1IgRIzR06FCv15%2BRkaHk5GSPNocjWA0NLX14F67N35K%2Bt/UHBgYoNNSpCxfa1NXVfYNXdXPZtTa71iVRmz%2Bya12SfWu7kXX19R/3pvhkwJo0aZJGjRqlH//4x1qzZo0uXryof/u3f9P999%2BvGTNmqKCgQEVFRcrMzNS7776rkpISbd68WZKUnp6uBQsW6KGHHlJ8fLy2b9%2Bu%2Bvp6paSkSJLS0tJUWFiocePGyeVyKT8/XxMmTNDo0aMlSfHx8crPz9eqVavU0NCgzZs3Kz09vU/rj4yM7HU7sLa2SZ2d9vnLcD36Wn9XV7dt3zO71mbXuiRq80d2rUuyb212qssnL4EMHDhQL7zwggIDAzVt2jRNmzZNI0aMUH5%2Bvlwul7Zt26bXX39diYmJys3NVW5uriZOnChJSkpK0ooVK7Ry5UpNmDBBr776qrZs2aKwsDBJn9yqu/fee5WZmal7771XFy9e1MaNG93nLigoUGdnp6ZOnapHHnlEX/nKV5Sdnd0P7wIAAPBXAyzLsvp7EZ8FtbVNxud0OAKU8kyp8XlvlNcWTfZqnMMRIJdriBoaWmzzL5kedq3NrnVJ1OaP7FqXZN/abmRdw4d7/4iPST55BQsAAMCfEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPPZgLV3717dfvvtiouLc3/l5ORIko4ePao5c%2BYoLi5OycnJ2rlzp8exu3fvVkpKisaPH6%2B0tDQdOXLE3dfV1aV169Zp0qRJiouLU1ZWlmpqatz99fX1ys7OVkJCghITE5WXl6fOzs6bUzQAALAFnw1Y77//vr72ta/pyJEj7q/169fr/Pnzmj9/vlJTU1VeXq68vDytWbNGx44dkySVlZVp9erVWrt2rcrLyzVr1ixlZWWpra1NklRYWKgDBw5o165dKi0tVVBQkHJzc93nXbRokYKDg1VaWqri4mIdPHhQRUVF/fEWAAAAP%2BXTAeuOO%2B7o1b5v3z6FhYUpMzNTDodDSUlJmjlzprZv3y5J2rlzp6ZPn674%2BHgNHDhQc%2BfOlcvl0t69e9398%2BbN08iRIxUSEqLly5dr//79qqqq0qlTp3T48GHl5OTI6XRq1KhRys7Ods8NAADgDUd/L%2BBKuru79eGHH8rpdGrr1q3q6urSvffeqyVLlqiiokKxsbEe46Ojo1VcXCxJqqys1OzZs3v1Hz9%2BXE1NTTpz5ozH8RERERo2bJhOnDghSQoLC1NUVJS7f%2BzYsaqurtaFCxcUGhrq1fprampUW1vr0eZwBCsyMtL7N8ELgYE%2Bm4%2BvyOHwbr09dflbfd6wa212rUuiNn9k17ok%2B9Zmx7p8MmCdO3dOt99%2Bu6ZNm6aCggI1NDRo2bJlysnJ0fDhw%2BV0Oj3GBwUFqbW1VZLU0tJy1f6WlhZJUnBwcK/%2Bnr7Lj%2B153dra6nXA2rFjhzZt2uTRtmDBAi1cuNCr4%2B3K5RrSp/Ghoc5rD/JTdq3NrnVJ1OaP7FqXZN/a7FSXTwasiIgIj9tyTqdTOTk5euSRR5SWlqb29naP8e3t7RoyZIh77JX6XS6XOyz1PI91%2BfGWZfXq63ndM783MjIylJyc7NHmcASroaHF6zm84W9J39v6AwMDFBrq1IULberq6r7Bq7q57FqbXeuSqM0f2bUuyb613ci6%2BvqPe1N8MmAdP35cr7zyip588kkNGDBAknTp0iUFBATozjvv1PPPP%2B8xvrKyUjExMZKkmJgYVVRU9OqfMmWKhg0bpqioKFVWVrpvE9bW1qqxsVGxsbHq7u5WY2Oj6urqFBERIUk6efKkRowYoaFDh3q9/sjIyF63A2trm9TZaZ%2B/DNejr/V3dXXb9j2za212rUuiNn9k17ok%2B9Zmp7p88hJIWFiYtm/frq1bt6qzs1PV1dVav369vv71r2vatGmqq6tTUVGROjo6dOjQIZWUlLifu0pPT1dJSYkOHTqkjo4OFRUVqb6%2BXikpKZKktLQ0FRYWqqqqSs3NzcrPz9eECRM0evRojRkzRvHx8crPz1dzc7Oqqqq0efNmpaen9%2BfbAQAA/IxPXsEaMWKEnnvuOW3YsEGFhYUaPHiwpk%2BfrpycHA0ePFjbtm1TXl6eCgoKFB4ertzcXE2cOFGSlJSUpBUrVmjlypU6e/asoqOjtWXLFoWFhUn65Fmozs5OZWZmqqWlRYmJidq4caP73AUFBVq1apWmTp2qgIAApaamKjs7ux/eBQAA4K8GWJZl9fciPgtqa5uMz%2BlwBCjlmVLj894ory2a7NU4hyNALtcQNTS02OZScQ%2B71mbXuiRq80d2rUuyb203sq7hw71/xMckn7xFCAAA4M8IWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhkPWF1dXaanBAAA8CvGA9aUKVP0s5/9TJWVlaanBgAA8AvGA9YPfvADvffee5oxY4bmzJmjl156SU1NTaZPAwAA4LOMB6xHH31UL730kl5//XVNmjRJW7Zs0T333KMnn3xSb7/9tunTAQAA%2BJwb9pD7mDFjtHjxYr3%2B%2ButasGCBfve73%2Bm73/2ukpOT9ctf/pJntQAAgG05btTER48e1f/8z/9o7969unTpklJSUpSWlqazZ8/q2Wef1fvvv68NGzbcqNMDAAD0G%2BMBa/PmzdqzZ49OnTqlcePGafHixZoxY4ZCQkLcYwIDA/XTn/7U9KkBAAB8gvGA9eKLL2rWrFlKT09XdHT0FceMHTtWS5YsMX1qAAAAn2A8YO3fv1/Nzc1qbGx0t%2B3du1dJSUlyuVySpNtvv12333676VMDAAD4BOMPuf/pT3/StGnTtGPHDnfb%2BvXrNXPmTP3lL38xfToAAACfYzxg/exnP9MDDzygxYsXu9t%2B%2B9vfasqUKVq7dq3p0wEAAPgc4wHrww8/1Pz58zVo0CB3W2BgoObPn68//vGPpk8HAADgc4wHrJCQEJ0%2BfbpX%2B5kzZxQUFGT6dAAAAD7HeMCaNm2aVq5cqbffflvNzc1qaWnRoUOHtGrVKqWkpJg%2BHQAAgM8x/l2ETz75pKqqqvSd73xHAwYMcLenpKRo6dKlpk8HAADgc4wHLKfTqeeee04fffSRTpw4oYEDB2rs2LEaM2bMdc3X1dWluXPn6vOf/7z7IfmjR4/q6aefVmVlpVwul7KysjRnzhz3Mbt379bmzZtVW1urL3zhC/rJT36iuLg493zPPPOM9uzZo7a2Nk2cOFH/%2Bq//qsjISElSfX29fvKTn%2Bjw4cMKDAzUrFmztGzZMjkcN%2ByH3gMAAJu5Yb%2BL8J/%2B6Z/04IMPaurUqdcdriRp06ZNeuedd9yvz58/r/nz5ys1NVXl5eXKy8vTmjVrdOzYMUlSWVmZVq9erbVr16q8vFyzZs1SVlaW2traJEmFhYU6cOCAdu3apdLSUgUFBSk3N9c9/6JFixQcHKzS0lIVFxfr4MGDKioquu71AwCAzx7jAeujjz7S448/rjvvvFNf/OIXe331xcGDB7Vv3z498MAD7rZ9%2B/YpLCxMmZmZcjgcSkpK0syZM7V9%2B3ZJ0s6dOzV9%2BnTFx8dr4MCBmjt3rlwul/bu3evunzdvnkaOHKmQkBAtX75c%2B/fvV1VVlU6dOqXDhw8rJydHTqdTo0aNUnZ2tntuAAAAbxi/77Vy5UpVV1dryZIlGjp06HXPU19fr%2BXLl2vz5s0eV5AqKioUGxvrMTY6OlrFxcWSpMrKSs2ePbtX//Hjx9XU1KQzZ854HB8REaFhw4bpxIkTkqSwsDBFRUW5%2B8eOHavq6mpduHBBoaGh110PAAD47DAesI4cOaLnn3/e/czT9eju7lZOTo4ef/xx3XbbbR59LS0tcjqdHm1BQUFqbW29Zn9LS4skKTg4uFd/T9/lx/a8bm1t9Tpg1dTUqLa21qPN4Qh2P%2BdlSmDgDbvDe0M4HN6tt6cuf6vPG3atza51SdTmj%2Bxal2Tf2uxYl/GA5XK5NGTIkH9ojueee06DBg3SY4891qvP6XSqqanJo629vd19TqfTqfb29l79LpfLHZZ6nse6/HjLsnr19bzuS007duzQpk2bPNoWLFighQsXej2HHblcfftzERrqvPYgP2XX2uxal0Rt/siudUn2rc1OdRkPWI899pg2bNig9evXX/ctwj179qimpkYJCQmS5A5Mv/3tb7V06VIdOHDAY3xlZaViYmIkSTExMaqoqOjVP2XKFA0bNkxRUVGqrKx03yasra1VY2OjYmNj1d3drcbGRtXV1SkiIkKSdPLkSY0YMaJPtWRkZCg5OdmjzeEIVkNDSx/ehWvzt6Tvbf2BgQEKDXXqwoU2dXV13%2BBV3Vx2rc2udUnU5o/sWpdk39puZF19/ce9KcYD1ltvvaU//vGPSkxM1C233OLxK3Mk6Xe/%2B90153j99dc9Xj/11FOSpLVr16qhoUHr169XUVGRMjMz9e6776qkpESbN2%2BWJKWnp2vBggV66KGHFB8fr%2B3bt6u%2Bvt79Q07T0tJUWFiocePGyeVyKT8/XxMmTNDo0aMlSfHx8crPz9eqVavU0NCgzZs3Kz09vU/vQWRkZK/bgbW1TerstM9fhuvR1/q7urpt%2B57ZtTa71iVRmz%2Bya12SfWuzU13GA1ZiYqISExNNT%2Bvmcrm0bds25eXlqaCgQOHh4crNzdXEiRMlSUlJSVqxYoVWrlyps2fPKjo6Wlu2bFFYWJikT27VdXZ2KjMzUy0tLUpMTNTGjRvd8xcUFGjVqlWaOnWqAgIClJqaquzs7BtWDwAAsJ8BlmVZ/b2Iz4La2qZrD%2BojhyNAKc%2BUGp/3Rnlt0WSvxjkcAXK5hqihocU2/5LpYdfa7FqXRG3%2ByK51Sfat7UbWNXz49f9Eg3/EDfnx5MePH9fzzz%2Bvjz76SM8%2B%2B6x%2B%2B9vfKjo6%2BoZe2QKAGyVh%2BevXHoTr4u0/vAB/YzxgffDBB3r00Uc1fvx4ffDBB7p06ZL%2B/Oc/Kz8/X5s2bdJ9991n%2BpTwEw9tPHDtQfhM4EMVPfj/Anq8k/dgfy/BKOMB65lnntF3vvMdLV682P2zsJ5%2B%2BmkNHTqUgAVAEh%2BqAOzP%2BPf5f/DBB0pNTe3V/uijj%2Bqvf/2r6dMBAAD4HOMBa%2BDAgWpubu7VXl1d3eunpAMAANiR8YB1//336%2Bc//7kaGhrcbSdPnlReXp6%2B%2BtWvmj4dAACAzzEesJYtW6b29nZNmjRJbW1tSktL04wZM%2BRwOLR06VLTpwMAAPA5xh9yDwkJ0UsvvaSDBw/qT3/6k7q7uxUbG6uvfOUrCgjwr1/tAgAAcD1uyM/Bkj75iepJSUk3anoAAACfZTxgJScna8CAAVft9%2BZ3EQIAAPgz4wHr61//ukfA6ujo0KlTp7R//34tWrTI9OkAAAB8jvGA9cQTT1yx/cUXX9S7776rb33rW6ZPCQAA4FNu2lPn9913n956662bdToAAIB%2Bc9MC1uHDhzV48OCbdToAAIB%2BY/wW4eW3AC3LUnNzs06cOMHtQQAA8JlgPGB97nOf6/VdhAMHDtS3v/1tzZw50/TpAAAAfI7xgLV27VrTUwIAAPgV4wGrvLzc67F333236dMDAAD0O%2BMBa%2B7cubIsy/3Vo%2Be2YU/bgAED9Oc//9n06QEAAPqd8YD1i1/8QmvWrNGyZcs0ceJEDRw4UEePHtXKlSv1jW98Q/fdd5/pUwIAAPgU4z%2BmYd26dVqxYoXuv/9%2BhYSEaPDgwZowYYJWrVqlbdu26fOf/7z7CwAAwI6MB6yamhqNHDmyV3tISIgaGhpMnw4AAMDnGA9Y48eP14YNG9Tc3Oxua2xs1Pr165WUlGT6dAAAAD7H%2BDNYubm5%2Bva3v60pU6ZozJgxkqSPPvpIw4cP169%2B9SvTpwMAAPA5xgPW2LFjtXfvXpWUlOjkyZOSpG984xuaPn26nE6n6dMBAAD4HOMBS5JCQ0M1Z84cffzxxxo1apSkT36aOwAAwGeB8WewLMvSM888o7vvvlszZszQmTNntGzZMv3oRz9SR0eH6dMBAAD4HOMB64UXXtCePXu0YsUKDRo0SJJ0//3363//93/17LPPmj4dAACAzzEesHbs2KGf/vSnSktLc//09ocfflh5eXl69dVXTZ8OAADA5xgPWB9//LG%2B%2BMUv9mq/9dZbVVdXZ/p0AAAAPsd4wPr85z%2BvY8eO9Wp/66233A%2B8AwAA2Jnx7yL87ne/q3/913/V2bNnZVmWDh48qJdeekkvvPCCfvSjH5k%2BHQAAgM8xHrBmz56tzs5OFRYWqr29XT/96U91yy23aPHixXr00UdNnw4AAMDnGA9YL7/8sh588EFlZGTo3LlzsixLt9xyi%2BnTAAAA%2BCzjz2A9/fTT7ofZw8PDrztcHTx4UHPmzNFdd92lyZMna/Xq1Wpvb5ckHT16VHPmzFFcXJySk5O1c%2BdOj2N3796tlJQUjR8/XmlpaTpy5Ii7r6urS%2BvWrdOkSZMUFxenrKws1dTUuPvr6%2BuVnZ2thIQEJSYmKi8vT52dnddVAwAA%2BGwyHrDGjBmjEydO/ENznDt3Tt///vf16KOP6p133tHu3bt1%2BPBh/cd//IfOnz%2Bv%2BfPnKzU1VeXl5crLy9OaNWvcD9aXlZVp9erVWrt2rcrLyzVr1ixlZWWpra1NklRYWKgDBw5o165dKi0tVVBQkHJzc93nXrRokYKDg1VaWqri4mIdPHhQRUVF/1A9AADgs8X4LcKYmBgtWbJEW7du1ZgxYzR48GCP/jVr1lxzjvDwcL399tsKCQmRZVlqbGzUxYsXFR4ern379iksLEyZmZmSpKSkJM2cOVPbt2/XnXfeqZ07d2r69OmKj4%2BXJM2dO1c7duzQ3r17NXv2bO3cuVNLlizRyJEjJUnLly/XPffco6qqKnV3d%2Bvw4cPav3%2B/nE6nRo0apezsbK1fv17f%2B973DL9TAADArowHrNOnT7vDTW1t7XXPExISIkm69957dfbsWSUkJCgtLU0bN25UbGysx9jo6GgVFxdLkiorKzV79uxe/cePH1dTU5POnDnjcXxERISGDRvmvuoWFhamqKgod//YsWNVXV2tCxcuKDQ01Ku119TU9Krd4QhWZGSkl9V7JzDQ%2BAVIAAD6jZ0%2B14wErDVr1uiHP/yhgoOD9cILL5iY0m3fvn06f/68lixZooULFyoqKkpOp9NjTFBQkFpbWyVJLS0tV%2B1vaWmRJAUHB/fq7%2Bm7/Nie162trV4HrB07dmjTpk0ebQsWLNDChQu9Oh4AgM%2Bi0FDntQf5CSMB61e/%2BpXmz5/vEVy%2B%2B93vas2aNf/wVZugoCAFBQUpJydHc%2BbM0WOPPaampiaPMe3t7RoyZIikTwJRz8Pwf9/vcrncYanneazLj7csq1dfz%2Bue%2Bb2RkZGh5ORkjzaHI1gNDS1ez%2BENOyV9AAAuXGhTV1e30TldLu8/v00yErAsy%2BrV9t577%2BnixYvXNd97772nH//4x3r55ZfdvzD60qVLGjhwoKKjo3XgwAGP8ZWVlYqJiZH0yTNgFRUVvfqnTJmiYcOGKSoqSpWVle7bhLW1tWpsbFRsbKy6u7vV2Niouro6RURESJJOnjypESNGaOjQoV6vPzIyslewrK1tUmen2T80AADYSVdXt20%2BK33yEsitt96q9vZ2/fznP9elS5f0t7/9TevWrVN6erqmTZumuro6FRUVqaOjQ4cOHVJJSYn7uav09HSVlJTo0KFD6ujoUFFRkerr65WSkiJJSktLU2FhoaqqqtTc3Kz8/HxNmDBBo0eP1pgxYxQfH6/8/Hw1NzerqqpKmzdvVnp6en%2B%2BHQAAwM8Yf8jdhCFDhmjr1q3Kz8/X5MmTNXToUM2cOVMLFizQoEGDtG3bNuXl5amgoEDh4eHKzc3VxIkTJX3yXYUrVqzQypUrdfbsWUVHR2vLli0KCwuT9MmzUJ2dncrMzFRLS4sSExO1ceNG97kLCgq0atUqTZ06VQEBAUpNTVV2dnY/vAsAAMBfDbCudH%2Bvj2677Ta9/fbbCg8Pd7fFxcXp5Zdf5hc8/3%2B1tU3XHtRHDkeAUp4pNT4vAAA32zt5D6qhocX4LcLhw71/xMckY1ewnn76aY%2BfedXR0aH169f3ejjcm5%2BDBQAA4M%2BMBKy777671899iouLU0NDgxoaGkycAgAAwG8YCVimf/YVAACAP/PJ7yIEAADwZwQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMMxnA9bx48f1%2BOOPa8KECZo8ebKWLl2qc%2BfOSZKOHj2qOXPmKC4uTsnJydq5c6fHsbt371ZKSorGjx%2BvtLQ0HTlyxN3X1dWldevWadKkSYqLi1NWVpZqamrc/fX19crOzlZCQoISExOVl5enzs7Om1M0AACwBZ8MWO3t7fre976nuLg4/eEPf9Arr7yixsZG/fjHP9b58%2Bc1f/58paamqry8XHl5eVqzZo2OHTsmSSorK9Pq1au1du1alZeXa9asWcrKylJbW5skqbCwUAcOHNCuXbtUWlqqoKAg5ebmus%2B9aNEiBQcHq7S0VMXFxTp48KCKior6420AAAB%2ByicDVnV1tW677TYtWLBAgwYNksvlUkZGhsrLy7Vv3z6FhYUpMzNTDodDSUlJmjlzprZv3y5J2rlzp6ZPn674%2BHgNHDhQc%2BfOlcvl0t69e9398%2BbN08iRIxUSEqLly5dr//79qqqq0qlTp3T48GHl5OTI6XRq1KhRys7Ods8NAADgDZ8MWF/4whe0detWBQYGutt%2B85vf6Etf%2BpIqKioUGxvrMT46OlrHjx%2BXJFVWVl61v6mpSWfOnPHoj4iI0LBhw3TixAlVVFQoLCxMUVFR7v6xY8equrpaFy5cuBGlAgAAG3L09wKuxbIsbdy4UW%2B%2B%2BaZefPFF/epXv5LT6fQYExQUpNbWVklSS0vLVftbWlokScHBwb36e/ouP7bndWtrq0JDQ71ac01NjWpraz3aHI5gRUZGenW8twIDfTIfAwBwXez0uebTAau5uVk/%2BtGP9OGHH%2BrFF1/UrbfeKqfTqaamJo9x7e3tGjJkiKRPAlF7e3uvfpfL5Q5LPc9jXX68ZVm9%2Bnpe98zvjR07dmjTpk0ebQsWLNDChQu9ngMAgM%2Ba0FDntQf5CZ8NWKdPn9a8efP0uc99TsXFxQoPD5ckxcbG6sCBAx5jKysrFRMTI0mKiYlRRUVFr/4pU6Zo2LBhioqK8riNWFtbq8bGRsXGxqq7u1uNjY2qq6tTRESEJOnkyZMaMWKEhg4d6vXaMzIylJyc7NHmcASroaGlb2/CNdgp6QMAcOFCm7q6uo3O6XJ5f4HEJJ/8hD5//ry%2B/e1v66677tJ//ud/usOVJKWkpKiurk5FRUXq6OjQoUOHVFJSotmzZ0uS0tPTVVJSokOHDqmjo0NFRUWqr69XSkqKJCktLU2FhYWqqqpSc3Oz8vPzNWHCBI0ePVpjxoxRfHy88vPz1dzcrKqqKm3evFnp6el9Wn9kZKS%2B9KUveXyFh0eos7Pb6JfpP4QAAPSnri6zn5Odnf33OTnAsiyr385%2BFb/85S%2B1du1aOZ1ODRgwwKPvyJEjev/995WXl6e//OUvCg8PV3Z2ttLS0txj9uzZo8LCQp09e1bR0dHKzc3Vl7/8ZUlSR0eHnn32Wb388stqaWlRYmKiVq9erVtuuUWSVFdXp1WrVqmsrEwBAQFKTU3VkiVLPB64vx61tU3XHtRHDkeAUp4pNT4vAAA32zt5D6qhocV4KBo%2B3Ps7UCb5ZMCyIwIWAABXZ7eA5ZO3CAEAAPwZAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMJ8PWOfOnVNKSorKysrcbUePHtWcOXMUFxen5ORk7dy50%2BOY3bt3KyUlRePHj1daWpqOHDni7uvq6tK6des0adIkxcXFKSsrSzU1Ne7%2B%2Bvp6ZWdnKyEhQYmJicrLy1NnZ%2BeNLxQAANiGTwesd999VxkZGTp9%2BrS77fz585o/f75SU1NVXl6uvLw8rVmzRseOHZMklZWVafXq1Vq7dq3Ky8s1a9YsZWVlqa2tTZJUWFioAwcOaNeuXSotLVVQUJByc3Pd8y9atEjBwcEqLS1VcXGxDh48qKKioptaNwAA8G8%2BG7B2796tJUuWaPHixR7t%2B/btU1hYmDIzM%2BVwOJSUlKSZM2dq%2B/btkqSdO3dq%2BvTpio%2BP18CBAzV37ly5XC7t3bvX3T9v3jyNHDlSISEhWr58ufbv36%2BqqiqdOnVKhw8fVk5OjpxOp0aNGqXs7Gz33AAAAN5w9PcCruaee%2B7RzJkz5XA4PEJWRUWFYmNjPcZGR0eruLhYklRZWanZs2f36j9%2B/Liampp05swZj%2BMjIiI0bNgwnThxQpIUFhamqKgod//YsWNVXV2tCxcuKDQ01Ku119TUqLa21qPN4QhWZGSkV8d7KzDQZ/MxAAB9ZqfPNZ8NWMOHD79ie0tLi5xOp0dbUFCQWltbr9nf0tIiSQoODu7V39N3%2BbE9r1tbW70OWDt27NCmTZs82hYsWKCFCxd6dTwAAJ9FoaHOaw/yEz4bsK7G6XSqqanJo629vV1Dhgxx97e3t/fqd7lc7rDU8zzW5cdbltWrr%2Bd1z/zeyMjIUHJyskebwxGshoYWr%2Bfwhp2SPgAAFy60qaur2%2BicLpf3n98m%2BV3Aio2N1YEDBzzaKisrFRMTI0mKiYlRRUVFr/4pU6Zo2LBhioqKUmVlpfs2YW1trRobGxUbG6vu7m41Njaqrq5OERERkqSTJ09qxIgRGjp0qNdrjIyM7HU7sLa2SZ2dZv/QAABgJ11d3bb5rPS7SyApKSmqq6tTUVGROjo6dOjQIZWUlLifu0pPT1dJSYkOHTqkjo4OFRUVqb6%2BXikpKZKktLQ0FRYWqqqqSs3NzcrPz9eECRM0evRojRkzRvHx8crPz1dzc7Oqqqq0efNmpaen92fJAADAz/jdFSyXy6Vt27YpLy9PBQUFCg8PV25uriZOnChJSkpK0ooVK7Ry5UqdPXtW0dHR2rJli8LCwiR98ixUZ2enMjMz1dLSosTERG3cuNE9f0FBgVatWqWpU6cqICBAqampys7O7odKAQCAvxpgWZbV34v4LKitbbr2oD5yOAKU8kyp8XkBALjZ3sl7UA0NLcZvEQ4f7v0jPib53S1CAAAAX0fAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwLqC%2Bvp6ZWdnKyEhQYmJicrLy1NnZ2d/LwsAAPgJAtYVLFq0SMHBwSotLVVxcbEOHjyooqKi/l4WAADwEwSsy5w6dUqHDx9WTk6OnE6nRo0apezsbG3fvr2/lwYAAPwEAesyFRUVCgsLU1RUlLtt7Nixqq6u1oULF/pxZQAAwF84%2BnsBvqalpUVOp9Ojred1a2urQkNDrzlHTU2NamtrPdocjmBFRkaaW6ikwEDyMQDAPuz0uUbAukxwcLDa2to82npeDxkyxKs5duzYoU2bNnm0/eAHP9ATTzxhZpH/X01Njb49okIZGRnGw1t/qqmp0Y4dO2xXl2Tf2uxal0Rt/siudUn2ra2mpka/%2BMUvbFWXfaKiITExMWpsbFRdXZ277eTJkxoxYoSGDh3q1RwZGRn67//%2Bb4%2BvjIwM42utra3Vpk2bel0t83d2rUuyb212rUuiNn9k17ok%2B9Zmx7q4gnWZMWPGKD4%2BXvn5%2BVq1apUaGhq0efNmpaenez1HZGSkbRI4AADoO65gXUFBQYE6Ozs1depUPfLII/rKV76i7Ozs/l4WAADwE1zBuoKIiAgVFBT09zIAAICf4gqWHxs%2BfLh%2B8IMfaPjw4f29FKPsWpdk39rsWpdEbf7IrnVJ9q3NjnUNsCzL6u9FAAAA2AlXsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbB8XH19vbKzs5WQkKDExETl5eWps7PzimPfeustzZw5U%2BPHj9dDDz2kN9988yav1nt9qet73/uexo0bp7i4OPfX/v37b/KK%2B%2B7cuXNKSUlRWVnZVcf405718KYuf9uz48eP6/HHH9eECRM0efJkLV26VOfOnbviWH/as77U5W97dvDgQc2ZM0d33XWXJk%2BerNWrV6u9vf2KY/1pz6S%2B1eZv%2ByZJXV1deuyxx/TUU09ddYy/7dkVWfBp3/zmN60nn3zSam1ttU6fPm1Nnz7d2rJlS69xH330kTVu3DjrjTfesDo6OqxXX33VuvPOO60zZ870w6qvzdu6LMuyEhMTrbKyspu8wn/MO%2B%2B8Y91///1WbGysdejQoSuO8bc9syzv6rIs/9qztrY2a/Lkydazzz5rXbx40Tp37pw1b9486/vf/36vsf60Z32py7L8a8/q6%2ButcePGWbt27bK6urqss2fPWjNmzLCeffbZXmP9ac8sq2%2B1WZZ/7VuPjRs3Wrfddpu1bNmyK/b7255dDVewfNipU6d0%2BPBh5eTkyOl0atSoUcrOztb27dt7jd29e7cSEhJ0//33y%2BFw6OGHH9bdd9%2BtHTt29MPKP11f6qqqqtL58%2Bd1%2B%2B2398NKr8/u3bu1ZMkSLV68%2BJrj/GXPJO/r8rc9q66u1m233aYFCxZo0KBBcrlcysjIUHl5ea%2Bx/rRnfanL3/YsPDxcb7/9ttLS0jRgwAA1Njbq4sWLCg8P7zXWn/ZM6ltt/rZv0idX5/bt26cHHnjgqmP8bc%2BuhoDlwyoqKhQWFqaoqCh329ixY1VdXa0LFy54jK2srFRsbKxHW3R0tI4fP35T1toXfanr/fff15AhQ7R48WJNnDhRM2bMUHFx8c1ecp/cc889euONN/Twww9/6jh/2jPJ%2B7r8bc%2B%2B8IUvaOvWrQoMDHS3/eY3v9GXvvSlXmP9ac/6Upe/7ZkkhYSESJLuvfdezZw5U8OHD1daWlqvcf60Zz28rc3f9q2%2Bvl7Lly/Xz3/%2BczmdzquO88c9uxJHfy8AV9fS0tLrD2HP69bWVoWGhn7q2KCgILW2tt74hfZRX%2Bq6dOmSxo8fr8WLFysmJkZlZWV64oknNGTIED300EM3dd3e8va3wfvTnkne1%2BWPe9bDsixt3LhRb775pl588cVe/f62Zz2uVZc/79m%2Bfft0/vx5LVmyRAsXLtTWrVs9%2Bv11z6Rr1%2BZP%2B9bd3a2cnBw9/vjjuu222z51rD/v2d/jCpYPCw4OVltbm0dbz%2BshQ4Z4tDudzl4PQba3t/ca5wv6Uldqaqq2bt2q22%2B/XQMHDtQ999yj1NRUvfbaazdtvTeKP%2B1ZX/jrnjU3N2vhwoUqKSnRiy%2B%2BqFtvvbXXGH/cM2/q8tc9kz754I2KilJOTo5KS0t1/vx5j35/3LMe16rNn/btueee06BBg/TYY49dc6w/79nfI2D5sJiYGDU2Nqqurs7ddvLkSY0YMUJDhw71GBsbG6uKigqPtsrKSsXExNyUtfZFX%2BoqLi7u9T%2BLS5cuafDgwTdlrTeSP%2B1ZX/jjnp0%2BfVqzZ89Wc3OziouLrxhCJP/bM2/r8rc9e%2B%2B99/Tggw/q0qVL7rZLly5p4MCBva58%2BNue9aU2f9q3PXv26PDhw0pISFBCQoJeeeUVvfLKK0pISOg11t/27Kr6%2Byl7fLpHH33UWrx4sdXU1OT%2BbruCgoJe4yorK61x48ZZr776qvu7LsaNG2f99a9/7YdVX5u3df3yl7%2B0kpKSrA8//NDq6uqy3nzzTevOO%2B%2B0ysvL%2B2HVffdp323nb3v29z6tLn/bs8bGRuurX/2q9dRTT1ldXV2fOtaf9qwvdfnbnjU3N1v33nuvlZ%2Bfb128eNH6%2BOOPrfT0dGvFihW9xvrTnllW32rzt337e8uWLbvqdxH6255dDQHLx9XW1lpPPPGENWHCBGvixInW2rVrrc7OTsuyLGv8%2BPHWnj173GP3799vzZo1yxo/frw1ffp06/e//31/LfuavK2ru7vb%2Bvd//3frvvvus%2B68805r%2BvTp1muvvdafS%2B%2BTy4OIP%2B/Z3/u0uvxtz7Zt22bFxsZaX/7yl63x48d7fFmW/%2B5ZX%2Brytz2zLMuqqKiwHn/8cSshIcG67777rA0bNlgXL160LMt/96yHt7X54771uDxg%2BfueXckAy7Ks/r6KBgAAYCc8gwUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAACfce7cOaWkpKisrMyr8dOnT1dcXJzH16233qrnnnvuBq/00/HLngEAgE9499139dRTT%2Bn06dNeH/Pqq696vN64caN%2B//vf65vf/Kbp5fUJV7AAAEC/2717t5YsWaLFixf36nv77beVnp6uhIQETZ8%2BXS%2B//PIV5zh06JCef/55bdy4sd9/OTQBCwAA9Lt77rlHb7zxhh5%2B%2BGGP9uPHjysrK0vz589XWVmZVq9erfz8fJWWlnqM6%2Brq0ooVK5SVlaUxY8bcxJVfGQELAAD0u%2BHDh8vh6P3k0ksvvaSpU6fqgQceUGBgoO666y498sgj2r59u8e4kpIStba26lvf%2BtbNWvKn4hksAADgs/72t7/p0KFDSkhIcLd1dXVp9OjRHuN%2B/etfKyMjQ0FBQTd7iVdEwAIAAD5rxIgR%2BvrXv65Vq1a522pqamRZlvt1XV2d3nvvPa1bt64/lnhF3CIEAAA%2BKz09Xa%2B88or%2B8Ic/qLu7W//3f/%2Bnb37zm9q2bZt7zHvvvafIyEiNGjWqH1fqiStYAADAZ335y1/Whg0btGHDBv3whz%2BU0%2BnUjBkz9C//8i/uMVVVVYqKiurHVfb2/wChPxLcrmzL5gAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-3713796977191528543">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">31171488</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1016669</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">18647868</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">19888962</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6255427</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">26729227</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">24068933</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">21969738</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">18815822</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">53257</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (417454)</td>
        <td class="number">417454</td>
        <td class="number">100.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-3713796977191528543">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">10000</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10002</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10003</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10005</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10006</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">39999726</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">39999736</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">39999781</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">39999808</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">39999921</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_gender">gender<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-7065568492004359376">
    <table class="mini freq">
        <tr class="">
    <th>Female</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 50.2%">
            209416
        </div>

    </td>
</tr><tr class="">
    <th>Male</th>
    <td>
        <div class="bar" style="width:99%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 49.8%">
            208048
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-7065568492004359376, #minifreqtable-7065568492004359376"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-7065568492004359376">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">Female</td>
        <td class="number">209416</td>
        <td class="number">50.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Male</td>
        <td class="number">208048</td>
        <td class="number">49.8%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_age">age<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>55</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>40.473</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>18</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>74</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram8661973349582570581">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABMUlEQVR4nO3dy20CQRBAQWMRkoMgJ87OiSDIaZyA9QSWlm0vVWcOc3nq6eV3WmutD%2BBXn3sfACY7732A/%2BTrenvq9ffvy0Yn4VVMEAgCgXCIK9azVx94lAkCQSAQBALhEDvIVH/ZjTwansUEgWCCDGPqzGKCQBAIBIFAEAgEgUAQCASBQBAIBIFA8E76Afgq8HZMEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgTDu4%2B7%2ByoBJTBAIAoEw7orFTO/6m8ECeUP2vMe5YkEQCASBQLCDsJkjLPYmCASBQBAIBIFAOK211t6HgKlMEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAg/U/Qe1ASszG0AAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives8661973349582570581,#minihistogram8661973349582570581"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives8661973349582570581">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles8661973349582570581"
                                                  aria-controls="quantiles8661973349582570581" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram8661973349582570581" aria-controls="histogram8661973349582570581"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common8661973349582570581" aria-controls="common8661973349582570581"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme8661973349582570581" aria-controls="extreme8661973349582570581"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles8661973349582570581">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>18</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>24</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>31</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>39</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>48</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>63</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>74</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>56</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>17</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>12.056</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.29788</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.25923</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>40.473</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>9.8917</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.50618</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>16896145</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>145.35</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>3.2 MiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram8661973349582570581">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA2MklEQVR4nO3df1SU553//5cwKoMIDEHQ9uhxK5AfjakEI0ET02Cosf4oRQgnZdOYttoypC45EU0jrS4ExJqk6nrgdJOmtAm7oeBaNbWJbTeNLCFKjFWTVgNuasx6lAEBcYAoMJ8/8mW%2BHYlxgpfOgM/HORzOXNd9X9c1b2duXt73zTDC5XK5BAAAAGMCfL0AAACA4YaABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMs/h6AdcLh6PD10vwEBAwQhERY3TmjFN9fS5fL2fIoG6DQ90Gh7oNDnUbnOFat3HjxvpkXs5gXacCAkZoxIgRCggY4eulDCnUbXCo2%2BBQt8GhboND3cwiYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAY5pcBa8eOHYqPj/f4uvXWW3XrrbdKkg4ePKiMjAzFx8crOTlZVVVVHvtv27ZNKSkpmjZtmtLS0nTgwAF3X29vr9avX6%2BZM2cqPj5e2dnZampqcve3tLTIbrdr%2BvTpSkxMVFFRkXp6eq7NEwcAAMOCXwasRYsW6cCBA%2B6vV199VeHh4SoqKlJ7e7uWLVum1NRU1dfXq6ioSOvWrdOhQ4ckSXv37lVhYaFKSkpUX1%2BvRYsWKTs7W11dXZKksrIy1dbWauvWraqpqVFQUJDy8/Pdc%2Bfm5io4OFg1NTWqrq5WXV2dysvLfVEGAAAwRPllwPpHLpdLeXl5%2BupXv6pvfOMb2r17t8LDw5WVlSWLxaKkpCQtXLhQFRUVkqSqqirNnz9fCQkJGjlypJYsWSKbzaZdu3a5%2B5cuXaoJEyYoJCREq1ev1p49e3TixAkdP35c%2B/btU15enqxWqyZOnCi73e4eGwAAwBt%2BH7C2b9%2BuxsZGPfHEE5KkhoYGxcXFeWwTExOjI0eOSJIaGxsv2d/R0aFTp0559EdGRiosLExHjx5VQ0ODwsPDFR0d7e6fMmWKTp48qbNnz16tpwgAAIYZi68X8Fn6%2BvpUVlamH/zgBwoJCZEkOZ1OWa1Wj%2B2CgoLU2dl52X6n0ylJCg4OHtDf33fxvv2POzs7FRoa6tW6m5qa5HA4PNoslmBFRUV5tf%2B1EBgY4PEd3vHnuqU8XePrJXjtDyvu9vUShgR/fr35M%2Bo2ONTNLL8OWHv37lVTU5PS09PdbVarVR0dHR7bdXd3a8yYMe7%2B7u7uAf02m80dlvrvx7p4f5fLNaCv/3H/%2BN6orKzUli1bPNpycnK0fPlyr8e4VkJDrZffCANQtytjs3n/fgKvt8GiboND3czw64D12muvKSUlxeOMU1xcnGpraz22a2xsVGxsrCQpNjZWDQ0NA/pnz56tsLAwRUdHe1xGdDgcamtrU1xcnPr6%2BtTW1qbm5mZFRkZKko4dO6bx48dr7NixXq87MzNTycnJHm0WS7BaW53eP/mrLDAwQKGhVp0926Xe3j5fL2fIoG5m%2BNN7wZ/xehsc6jY4w7VuvvoPnV8HrP379%2Bvb3/62R1tKSoo2bNig8vJyZWVlaf/%2B/dq5c6dKS0slSenp6crJydG8efOUkJCgiooKtbS0KCUlRZKUlpamsrIyTZ06VTabTcXFxZoxY4YmTZokSUpISFBxcbEKCgrU2tqq0tJSjzNo3oiKihpwOdDh6FBPj/%2B9YHt7%2B/xyXf6Oul0Zavf58HobHOo2ONTNDL8OWB999NGAoGKz2fTCCy%2BoqKhImzdvVkREhPLz83XnnXdKkpKSkrRmzRqtXbtWp0%2BfVkxMjJ577jmFh4dL%2BuRSXU9Pj7KysuR0OpWYmKiNGze6x9%2B8ebMKCgo0Z84cBQQEKDU1VXa7/Vo9ZQAAMAyMcLlcLl8v4nrgcHRcfqNryGIJkM02Rq2tTv6n8jn4c93mbay9/EZ%2B4ve5s3y9hCHBn19v/oy6Dc5wrdu4cd7f4mMSvyoAAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYJhf/6kcDC9D6ZPGJT5tHAAweJzBAgAAMIyABQAAYBgBCwAAwDACFgAAgGHc5A5cwlC7KR8A4D84gwUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGF%2BG7Da2tq0cuVKJSYm6o477pDdbldTU5Mk6eDBg8rIyFB8fLySk5NVVVXlse%2B2bduUkpKiadOmKS0tTQcOHHD39fb2av369Zo5c6bi4%2BOVnZ3tHleSWlpaZLfbNX36dCUmJqqoqEg9PT3X5kkDAIBhwW8D1g9/%2BEN1dnbqD3/4g15//XUFBgbqxz/%2Bsdrb27Vs2TKlpqaqvr5eRUVFWrdunQ4dOiRJ2rt3rwoLC1VSUqL6%2BnotWrRI2dnZ6urqkiSVlZWptrZWW7duVU1NjYKCgpSfn%2B%2BeNzc3V8HBwaqpqVF1dbXq6upUXl7uixIAAIAhyi8D1rvvvquDBw%2BqpKREoaGhCgkJUWFhoVasWKHdu3crPDxcWVlZslgsSkpK0sKFC1VRUSFJqqqq0vz585WQkKCRI0dqyZIlstls2rVrl7t/6dKlmjBhgkJCQrR69Wrt2bNHJ06c0PHjx7Vv3z7l5eXJarVq4sSJstvt7rEBAAC8YfH1Aj7NoUOHFBMTo9/85jf6z//8T3V1denuu%2B/WqlWr1NDQoLi4OI/tY2JiVF1dLUlqbGzU4sWLB/QfOXJEHR0dOnXqlMf%2BkZGRCgsL09GjRyVJ4eHhio6OdvdPmTJFJ0%2Be1NmzZxUaGurV%2BpuamuRwODzaLJZgRUVFeV%2BEqywwMMDjO3AtWSy87rzB%2B3RwqNvgUDez/DJgtbe36%2BjRo7r11lu1bds2dXd3a%2BXKlVq1apUiIyNltVo9tg8KClJnZ6ckyel0XrLf6XRKkoKDgwf09/ddvG//487OTq8DVmVlpbZs2eLRlpOTo%2BXLl3u1/7UUGmq9/EaAYSlP1/h6CZ/L20X3%2B3R%2B3qeDQ90Gh7qZ4ZcBa9SoUZKk1atXa/To0QoJCVFubq4eeOABpaWlqbu722P77u5ujRkzRtIngejT%2Bm02mzss9d%2BPdfH%2BLpdrQF//4/7xvZGZmank5GSPNoslWK2tTq/HuNoCAwMUGmrV2bNd6u3t8/VyAL/mq/cu79PBoW6DM1zrZrN5//PbJL8MWDExMerr69OFCxc0evRoSVJf3yf/2DfffLP%2B4z/%2Bw2P7xsZGxcbGSpJiY2PV0NAwoH/27NkKCwtTdHS0Ghsb3ZcJHQ6H2traFBcXp76%2BPrW1tam5uVmRkZGSpGPHjmn8%2BPEaO3as1%2BuPiooacDnQ4ehQT4//vWB7e/v8cl2AP/H1e4T36eBQt8Ghbmb45YXWmTNnauLEiXryySfldDp15swZ/exnP9N9992nBQsWqLm5WeXl5bpw4YLeeust7dy5033fVXp6unbu3Km33npLFy5cUHl5uVpaWpSSkiJJSktLU1lZmU6cOKFz586puLhYM2bM0KRJkzR58mQlJCSouLhY586d04kTJ1RaWqr09HRflgMAAAwxfhmwRo4cqRdffFGBgYGaO3eu5s6dq/Hjx6u4uFg2m00vvPCCXn31VSUmJio/P1/5%2Bfm68847JUlJSUlas2aN1q5dqxkzZuh3v/udnnvuOYWHh0v65F6oe%2B65R1lZWbrnnnv08ccfa%2BPGje65N2/erJ6eHs2ZM0cPPPCA7r77btntdh9UAQAADFUjXC6Xy9eLuB44HB2%2BXoIHiyVANtsYtbY6r9mp4Hkba6/JPIBpv8%2Bd5ZN5ffE%2BHQ6o2%2BAM17qNG%2Bf9LT4m%2BeUZLAAAgKGMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADD/PKDRgHAnwy134D11W89Avj/cQYLAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw/w2YO3atUu33HKL4uPj3V95eXmSpIMHDyojI0Px8fFKTk5WVVWVx77btm1TSkqKpk2bprS0NB04cMDd19vbq/Xr12vmzJmKj49Xdna2mpqa3P0tLS2y2%2B2aPn26EhMTVVRUpJ6enmvzpAEAwLDgtwHr8OHD%2BsY3vqEDBw64vzZs2KD29nYtW7ZMqampqq%2BvV1FRkdatW6dDhw5Jkvbu3avCwkKVlJSovr5eixYtUnZ2trq6uiRJZWVlqq2t1datW1VTU6OgoCDl5%2Be7583NzVVwcLBqampUXV2turo6lZeX%2B6IEAABgiPLrgHXrrbcOaN%2B9e7fCw8OVlZUli8WipKQkLVy4UBUVFZKkqqoqzZ8/XwkJCRo5cqSWLFkim82mXbt2ufuXLl2qCRMmKCQkRKtXr9aePXt04sQJHT9%2BXPv27VNeXp6sVqsmTpwou93uHhsAAMAbfhmw%2Bvr69N577%2BnPf/6z7r33Xs2ePVs//vGP1d7eroaGBsXFxXlsHxMToyNHjkiSGhsbL9nf0dGhU6dOefRHRkYqLCxMR48eVUNDg8LDwxUdHe3unzJlik6ePKmzZ89exWcMAACGE4uvF/Bpzpw5o1tuuUVz587V5s2b1draqlWrVikvL0/jxo2T1Wr12D4oKEidnZ2SJKfTecl%2Bp9MpSQoODh7Q39938b79jzs7OxUaGurV%2BpuamuRwODzaLJZgRUVFebX/tRAYGODxHcDwYbFc3%2B9rjm%2BDQ93M8suAFRkZ6XFZzmq1Ki8vTw888IDS0tLU3d3tsX13d7fGjBnj3vbT%2Bm02mzss9d%2BPdfH%2BLpdrQF//4/7xvVFZWaktW7Z4tOXk5Gj58uVej3GthIZaL78RgCHFZvP%2BeDWccXwbHOpmhl8GrCNHjuiVV17R448/rhEjRkiSzp8/r4CAAN1222361a9%2B5bF9Y2OjYmNjJUmxsbFqaGgY0D979myFhYUpOjra4zKiw%2BFQW1ub4uLi1NfXp7a2NjU3NysyMlKSdOzYMY0fP15jx471ev2ZmZlKTk72aLNYgtXa6vx8hbiKAgMDFBpq1dmzXert7fP1cgAY5E/HGl/g%2BDY4w7VuvvoPh18GrPDwcFVUVCgsLEyPPPKImpqatGHDBn3zm9/U3Llz9cwzz6i8vFxZWVnav3%2B/du7cqdLSUklSenq6cnJyNG/ePCUkJKiiokItLS1KSUmRJKWlpamsrExTp06VzWZTcXGxZsyYoUmTJkmSEhISVFxcrIKCArW2tqq0tFTp6emfa/1RUVEDLgc6HB3q6fG/F2xvb59frgvA4PGe/gTHt8GhbmaMcLlcLl8v4tPs27dPzz77rN5//32NHj1a8%2BfPV15enkaPHq3Dhw%2BrqKhI77//viIiImS325WWlubed/v27SorK9Pp06cVExOj/Px8feUrX5EkXbhwQZs2bdKOHTvkdDqVmJiowsJC3XDDDZKk5uZmFRQUaO/evQoICFBqaqpWrFihwMDAK3o%2BDkfHFe1vmsUSIJttjFpbndfsjTRvY%2B01mQe43v0%2Bd5avl%2BBTvji%2BDQfDtW7jxnl/Bcokvw1Yww0Bi4AFXCsErOEZFK624Vo3XwUsflUAAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhxgNWb2%2Bv6SEBAACGFOMBa/bs2frpT3%2BqxsZG00MDAAAMCcYD1qOPPqp33nlHCxYsUEZGhl5%2B%2BWV1dHSYngYAAMBvGQ9YDz74oF5%2B%2BWW9%2Buqrmjlzpp577jndddddevzxx/Xmm2%2Bang4AAMDvXLWb3CdPnqzHHntMr776qnJycvSnP/1J3/3ud5WcnKxf/vKX3KsFAACGLcvVGvjgwYP67W9/q127dun8%2BfNKSUlRWlqaTp8%2BrU2bNunw4cN69tlnr9b0AAAAPmM8YJWWlmr79u06fvy4pk6dqscee0wLFixQSEiIe5vAwED95Cc/MT01AACAXzAesF566SUtWrRI6enpiomJ%2BdRtpkyZohUrVpieGgAAwC8YD1h79uzRuXPn1NbW5m7btWuXkpKSZLPZJEm33HKLbrnlFtNTAwAA%2BAXjN7n/9a9/1dy5c1VZWelu27BhgxYuXKj333/f9HQAAAB%2Bx3jA%2BulPf6qvfe1reuyxx9xtf/zjHzV79myVlJSYng4AAMDvGA9Y7733npYtW6ZRo0a52wIDA7Vs2TL95S9/MT0dAACA3zEesEJCQvThhx8OaD916pSCgoJMTwcAAOB3jAesuXPnau3atXrzzTd17tw5OZ1OvfXWWyooKFBKSorp6QAAAPyO8d8ifPzxx3XixAl95zvf0YgRI9ztKSkpWrlypenpAAAXmbex1tdL8Nrvc2f5egnAVWE8YFmtVv385z/XBx98oKNHj2rkyJGaMmWKJk%2BebHoqAAAAv3TV/hbhP/3TP%2Bn%2B%2B%2B/XnDlzrihc9fb26qGHHtITTzzhbjt48KAyMjIUHx%2Bv5ORkVVVVeeyzbds2paSkaNq0aUpLS9OBAwc8xlu/fr1mzpyp%2BPh4ZWdnq6mpyd3f0tIiu92u6dOnKzExUUVFRerp6Rn0%2BgEAwPXHeMD64IMP9Mgjj%2Bi2227TzTffPODr89qyZYvefvtt9%2BP29nYtW7ZMqampqq%2BvV1FRkdatW6dDhw5Jkvbu3avCwkKVlJSovr5eixYtUnZ2trq6uiRJZWVlqq2t1datW1VTU6OgoCDl5%2Be7x8/NzVVwcLBqampUXV2turo6lZeXX1lRAADAdcX4JcK1a9fq5MmTWrFihcaOHXtFY9XV1Wn37t362te%2B5m7bvXu3wsPDlZWVJUlKSkrSwoULVVFRodtuu01VVVWaP3%2B%2BEhISJElLlixRZWWldu3apcWLF6uqqkorVqzQhAkTJEmrV6/WXXfdpRMnTqivr0/79u3Tnj17ZLVaNXHiRNntdm3YsEHf%2B973rui5AACA64fxgHXgwAH96le/Unx8/BWN09LSotWrV6u0tNTjDFJDQ4Pi4uI8to2JiVF1dbUkqbGxUYsXLx7Qf%2BTIEXV0dOjUqVMe%2B0dGRiosLExHjx6VJIWHhys6OtrdP2XKFJ08eVJnz55VaGioV2tvamqSw%2BHwaLNYghUVFeXV/tdCYGCAx3cA8AWLxfwxiOPb4FA3s4wHLJvNpjFjxlzRGH19fcrLy9Mjjzyim266yaPP6XTKarV6tAUFBamzs/Oy/U6nU5IUHBw8oL%2B/7%2BJ9%2Bx93dnZ6HbAqKyu1ZcsWj7acnBwtX77cq/2vpdBQ6%2BU3AoCrxGa7sp8Xn4Xj2%2BBQNzOMB6yHHnpIzz77rDZs2DDoS4Q///nPNWrUKD300EMD%2BqxWqzo6Ojzauru73aHOarWqu7t7QL/NZnOHpf77sS7e3%2BVyDejrf/x5QmNmZqaSk5M92iyWYLW2Or0e42oLDAxQaKhVZ892qbe3z9fLAXCduhrHRY5vgzNc63Y1Q/xnMR6w3njjDf3lL39RYmKibrjhBo8/mSNJf/rTny47xvbt29XU1KTp06dLkjsw/fGPf9TKlStVW%2Bv5GS%2BNjY2KjY2VJMXGxqqhoWFA/%2BzZsxUWFqbo6Gg1Nja6LxM6HA61tbUpLi5OfX19amtrU3NzsyIjIyVJx44d0/jx4z9XWIyKihpwOdDh6FBPj/%2B9YHt7%2B/xyXQCuD1fz%2BMPxbXComxnGA1ZiYqISExOvaIxXX33V43H/RzSUlJSotbVVGzZsUHl5ubKysrR//37t3LlTpaWlkqT09HTl5ORo3rx5SkhIUEVFhVpaWtyfIp%2BWlqaysjJNnTpVNptNxcXFmjFjhiZNmiRJSkhIUHFxsQoKCtTa2qrS0lKlp6df0fMBAADXF%2BMB69FHHzU9pAebzaYXXnhBRUVF2rx5syIiIpSfn68777xT0ie/VbhmzRqtXbtWp0%2BfVkxMjJ577jmFh4dL%2BuReqJ6eHmVlZcnpdCoxMVEbN250j79582YVFBRozpw5CggIUGpqqux2%2B1V9TgAAYHgZ4XK5XKYHPXLkiH71q1/pgw8%2B0KZNm/THP/5RMTExV3xmayhzODouv9E1ZLEEyGYbo9ZW5zU7FTyU/nwHgGvjavypHF8c34aD4Vq3ceOu7COjBsv472K%2B%2B%2B67ysjI0EcffaR3331X58%2Bf19/%2B9jd95zvf0euvv256OgAAAL9jPGA9/fTT%2Bs53vqMXX3xRI0eOlCQ99dRT%2Bva3vz3gowsAAACGo6tyBis1NXVA%2B4MPPqj//d//NT0dAACA3zEesEaOHKlz584NaD958uSAD/EEAAAYjowHrPvuu0/PPPOMWltb3W3Hjh1TUVGRvvrVr5qeDgAAwO8YD1irVq1Sd3e3Zs6cqa6uLqWlpWnBggWyWCxauXKl6ekAAAD8jvHPwQoJCdHLL7%2Bsuro6/fWvf1VfX5/i4uJ09913KyCAPyAJAACGP%2BMBq19SUpKSkpKu1vAAAAB%2By3jASk5O1ogRIy7Z783fIgQAABjKjAesb37zmx4B68KFCzp%2B/Lj27Nmj3Nxc09MBAAD4HeMB64c//OGntr/00kvav3%2B/vv3tb5ueEgAAwK9cs7vO7733Xr3xxhvXajoAAACfuWYBa9%2B%2BfRo9evS1mg4AAMBnjF8ivPgSoMvl0rlz53T06FEuDwIAgOuC8YD1hS98YcBvEY4cOVIPP/ywFi5caHo6AAAAv2M8YJWUlJgeEgAAYEgxHrDq6%2Bu93vaOO%2B4wPT0AAIDPGQ9YS5Yskcvlcn/1679s2N82YsQI/e1vfzM9PQAAgM8ZD1j/9m//pnXr1mnVqlW68847NXLkSB08eFBr167Vt771Ld17772mpwQAAPArxj%2BmYf369VqzZo3uu%2B8%2BhYSEaPTo0ZoxY4YKCgr0wgsv6Itf/KL7CwAAYDgyHrCampo0YcKEAe0hISFqbW01PR0AAIDfMR6wpk2bpmeffVbnzp1zt7W1tWnDhg1KSkoyPR0AAIDfMX4PVn5%2Bvh5%2B%2BGHNnj1bkydPliR98MEHGjdunH7961%2Bbng4AAMDvGA9YU6ZM0a5du7Rz504dO3ZMkvStb31L8%2BfPl9VqNT0dAACA3zEesCQpNDRUGRkZ%2BuijjzRx4kRJn3yaOwAAwPXA%2BD1YLpdLTz/9tO644w4tWLBAp06d0qpVq/SjH/1IFy5cMD0dAACA3zEesF588UVt375da9as0ahRoyRJ9913n/77v/9bmzZtMj0dAACA3zEesCorK/WTn/xEaWlp7k9v//rXv66ioiL97ne/Mz0dAACA3zEesD766CPdfPPNA9pvvPFGNTc3m54OAADA7xgPWF/84hd16NChAe1vvPGG%2B4Z3AACA4cz4bxF%2B97vf1b/%2B67/q9OnTcrlcqqur08svv6wXX3xRP/rRj0xPBwAA4HeMB6zFixerp6dHZWVl6u7u1k9%2B8hPdcMMNeuyxx/Tggw%2Bang4AAMDvGA9YO3bs0P3336/MzEydOXNGLpdLN9xwg%2BlpAAAA/Jbxe7Ceeuop983sERERgw5XdXV1ysjI0O23365Zs2apsLBQ3d3dkqSDBw8qIyND8fHxSk5OVlVVlce%2B27ZtU0pKiqZNm6a0tDQdOHDA3dfb26v169dr5syZio%2BPV3Z2tpqamtz9LS0tstvtmj59uhITE1VUVKSenp5BPQcAAHB9Mh6wJk%2BerKNHj17RGGfOnNH3v/99Pfjgg3r77be1bds27du3T//%2B7/%2Bu9vZ2LVu2TKmpqaqvr1dRUZHWrVvnvrF%2B7969KiwsVElJierr67Vo0SJlZ2erq6tLklRWVqba2lpt3bpVNTU1CgoKUn5%2Bvnvu3NxcBQcHq6amRtXV1aqrq1N5efkVPR8AAHB9MX6JMDY2VitWrNDzzz%2BvyZMna/To0R7969atu%2BwYERERevPNNxUSEiKXy6W2tjZ9/PHHioiI0O7duxUeHq6srCxJUlJSkhYuXKiKigrddtttqqqq0vz585WQkCBJWrJkiSorK7Vr1y4tXrxYVVVVWrFihSZMmCBJWr16te666y6dOHFCfX192rdvn/bs2SOr1aqJEyfKbrdrw4YN%2Bt73vme4UgAAYLgyHrA%2B/PBDd7hxOByDHickJESSdM899%2Bj06dOaPn260tLStHHjRsXFxXlsGxMTo%2BrqaklSY2OjFi9ePKD/yJEj6ujo0KlTpzz2j4yMVFhYmPusW3h4uKKjo939U6ZM0cmTJ3X27FmFhoYO%2BvkAAIDrh5GAtW7dOv3Lv/yLgoOD9eKLL5oY0m337t1qb2/XihUrtHz5ckVHR8tqtXpsExQUpM7OTkmS0%2Bm8ZL/T6ZQkBQcHD%2Bjv77t43/7HnZ2dXgespqamAeHSYglWVFSUV/tfC4GBAR7fAcAXLBbzxyCOb4ND3cwyErB%2B/etfa9myZR7B5bvf/a7WrVt3xaEiKChIQUFBysvLU0ZGhh566CF1dHR4bNPd3a0xY8ZI%2BiQQ9d8M/4/9NpvNHZb678e6eH%2BXyzWgr/9x//jeqKys1JYtWzzacnJytHz5cq/HuFZCQ62X3wgArhKbzftj6%2BfF8W1wqJsZRgKWy%2BUa0PbOO%2B/o448/HtR477zzjp588knt2LHD/Qejz58/r5EjRyomJka1tbUe2zc2Nio2NlbSJ/eANTQ0DOifPXu2wsLCFB0drcbGRvdlQofDoba2NsXFxamvr09tbW1qbm5WZGSkJOnYsWMaP368xo4d6/X6MzMzlZyc7NFmsQSrtdX5%2BQpxFQUGBig01KqzZ7vU29vn6%2BUAuE5djeMix7fBGa51u5oh/rMYvwfLhBtvvFHd3d165pln9Pjjj8vhcGj9%2BvVKT0/X3Llz9cwzz6i8vFxZWVnav3%2B/du7cqdLSUklSenq6cnJyNG/ePCUkJKiiokItLS1KSUmRJKWlpamsrExTp06VzWZTcXGxZsyYoUmTJkmSEhISVFxcrIKCArW2tqq0tFTp6emfa/1RUVEDztw5HB3q6fG/F2xvb59frgvA9eFqHn84vg0OdTPDLwPWmDFj9Pzzz6u4uFizZs3S2LFjtXDhQuXk5GjUqFF64YUXVFRUpM2bNysiIkL5%2Bfm68847JX3yW4Vr1qzR2rVrdfr0acXExOi5555TeHi4pE8u1fX09CgrK0tOp1OJiYnauHGje%2B7NmzeroKBAc%2BbMUUBAgFJTU2W3231QBQAAMFSNcH3a9b3P6aabbtKbb76piIgId1t8fLx27NjBH3j%2B/zgcHZff6BqyWAJks41Ra6vzmv1PZd7G2stvBOC68vvcWcbH9MXxbTgYrnUbN877W3xMMnYG66mnnvL4zKsLFy5ow4YNA24O9%2BZzsAAAAIYyIwHrjjvuGPCxBPHx8WptbVVra6uJKQAAAIYMIwHL9GdfAQAADGV8mhgAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgmN8GrCNHjuiRRx7RjBkzNGvWLK1cuVJnzpyRJB08eFAZGRmKj49XcnKyqqqqPPbdtm2bUlJSNG3aNKWlpenAgQPuvt7eXq1fv14zZ85UfHy8srOz1dTU5O5vaWmR3W7X9OnTlZiYqKKiIvX09FybJw0AAIYFvwxY3d3d%2Bt73vqf4%2BHj9z//8j1555RW1tbXpySefVHt7u5YtW6bU1FTV19erqKhI69at06FDhyRJe/fuVWFhoUpKSlRfX69FixYpOztbXV1dkqSysjLV1tZq69atqqmpUVBQkPLz891z5%2BbmKjg4WDU1NaqurlZdXZ3Ky8t9UQYAADBE%2BWXAOnnypG666Sbl5ORo1KhRstlsyszMVH19vXbv3q3w8HBlZWXJYrEoKSlJCxcuVEVFhSSpqqpK8%2BfPV0JCgkaOHKklS5bIZrNp165d7v6lS5dqwoQJCgkJ0erVq7Vnzx6dOHFCx48f1759%2B5SXlyer1aqJEyfKbre7xwYAAPCGxdcL%2BDRf%2BtKX9Pzzz3u0vfbaa/ryl7%2BshoYGxcXFefTFxMSourpaktTY2KjFixcP6D9y5Ig6Ojp06tQpj/0jIyMVFhamo0ePSpLCw8MVHR3t7p8yZYpOnjyps2fPKjQ01Kv1NzU1yeFweLRZLMGKioryav9rITAwwOM7APiCxWL%2BGMTxbXCom1l%2BGbD%2Bkcvl0saNG/X666/rpZde0q9//WtZrVaPbYKCgtTZ2SlJcjqdl%2Bx3Op2SpODg4AH9/X0X79v/uLOz0%2BuAVVlZqS1btni05eTkaPny5V7tfy2FhlovvxEAXCU225irNjbHt8Ghbmb4dcA6d%2B6cfvSjH%2Bm9997TSy%2B9pBtvvFFWq1UdHR0e23V3d2vMmE/epFarVd3d3QP6bTabOyz134918f4ul2tAX//j/vG9kZmZqeTkZI82iyVYra1Or8e42gIDAxQaatXZs13q7e3z9XIAXKeuxnGR49vgDNe6Xc0Q/1n8NmB9%2BOGHWrp0qb7whS%2BourpaERERkqS4uDjV1tZ6bNvY2KjY2FhJUmxsrBoaGgb0z549W2FhYYqOjlZjY6P7MqHD4VBbW5vi4uLU19entrY2NTc3KzIyUpJ07NgxjR8/XmPHjvV67VFRUQMuBzocHerpMf%2BCnbex9vIbAYCfuhrHxX69vX1XdfzhirqZ4ZcXWtvb2/Xwww/r9ttv1y9%2B8Qt3uJKklJQUNTc3q7y8XBcuXNBbb72lnTt3uu%2B7Sk9P186dO/XWW2/pwoULKi8vV0tLi1JSUiRJaWlpKisr04kTJ3Tu3DkVFxdrxowZmjRpkiZPnqyEhAQVFxfr3LlzOnHihEpLS5Wenu6TOgAAgKFphMvlcvl6ERf75S9/qZKSElmtVo0YMcKj78CBAzp8%2BLCKior0/vvvKyIiQna7XWlpae5ttm/frrKyMp0%2BfVoxMTHKz8/XV77yFUnShQsXtGnTJu3YsUNOp1OJiYkqLCzUDTfcIElqbm5WQUGB9u7dq4CAAKWmpmrFihUKDAy8oufkcHRcfqNB4AwWgKHs97mzjI9psQTIZhuj1lYnZ2I%2Bh%2BFat3HjvL8CZZJfBqzhiIAFAAMRsPzHcK2brwKWX14iBAAAGMoIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYRZfLwAAgKFi3sZaXy/hc/l97ixfL%2BG6xRksAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgf0wAA8Jmh9rEHgLc4gwUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgmN8HrDNnziglJUV79%2B51tx08eFAZGRmKj49XcnKyqqqqPPbZtm2bUlJSNG3aNKWlpenAgQPuvt7eXq1fv14zZ85UfHy8srOz1dTU5O5vaWmR3W7X9OnTlZiYqKKiIvX09Fz9JwoAAIYNvw5Y%2B/fvV2Zmpj788EN3W3t7u5YtW6bU1FTV19erqKhI69at06FDhyRJe/fuVWFhoUpKSlRfX69FixYpOztbXV1dkqSysjLV1tZq69atqqmpUVBQkPLz893j5%2BbmKjg4WDU1NaqurlZdXZ3Ky8uv6fMGAABDm98GrG3btmnFihV67LHHPNp3796t8PBwZWVlyWKxKCkpSQsXLlRFRYUkqaqqSvPnz1dCQoJGjhypJUuWyGazadeuXe7%2BpUuXasKECQoJCdHq1au1Z88enThxQsePH9e%2BffuUl5cnq9WqiRMnym63u8cGAADwht8GrLvuukt/%2BMMf9PWvf92jvaGhQXFxcR5tMTExOnLkiCSpsbHxkv0dHR06deqUR39kZKTCwsJ09OhRNTQ0KDw8XNHR0e7%2BKVOm6OTJkzp79qzppwgAAIYpi68XcCnjxo371Han0ymr1erRFhQUpM7Ozsv2O51OSVJwcPCA/v6%2Bi/ftf9zZ2anQ0FCv1t7U1CSHw%2BHRZrEEKyoqyqv9AQAwwWLx/jxKYGCAx3dcGb8NWJditVrV0dHh0dbd3a0xY8a4%2B7u7uwf022w2d1jqvx/r4v1dLteAvv7H/eN7o7KyUlu2bPFoy8nJ0fLly70eAwCAK2Wzef%2Bzq19oqPXyG%2BGyhlzAiouLU21trUdbY2OjYmNjJUmxsbFqaGgY0D979myFhYUpOjra4zKiw%2BFQW1ub4uLi1NfXp7a2NjU3NysyMlKSdOzYMY0fP15jx471eo2ZmZlKTk72aLNYgtXa6vzczxcAgMH6PD93AgMDFBpq1dmzXert7buKq7q2BhMyTRhyASslJUUbNmxQeXm5srKytH//fu3cuVOlpaWSpPT0dOXk5GjevHlKSEhQRUWFWlpalJKSIklKS0tTWVmZpk6dKpvNpuLiYs2YMUOTJk2SJCUkJKi4uFgFBQVqbW1VaWmp0tPTP9cao6KiBlwOdDg61NMzfF6wAAD/N5ifO729ffy8MmDIBSybzaYXXnhBRUVF2rx5syIiIpSfn68777xTkpSUlKQ1a9Zo7dq1On36tGJiYvTcc88pPDxc0ieX6np6epSVlSWn06nExERt3LjRPf7mzZtVUFCgOXPmKCAgQKmpqbLb7T54pgAAYKga4XK5XL5exPXA4ei4/EaDMG9j7eU3AgBcl36fO8vrbS2WANlsY9Ta6hxWZ7DGjfP%2BFh%2BT%2BFUBAAAAwwhYAAAAhhGwAAAADCNgAQAAGDbkfosQAAB4Zyj9ItTnuSF/KOAMFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYH2KlpYW2e12TZ8%2BXYmJiSoqKlJPT4%2BvlwUAAIYIAtanyM3NVXBwsGpqalRdXa26ujqVl5f7elkAAGCIIGBd5Pjx49q3b5/y8vJktVo1ceJE2e12VVRU%2BHppAABgiCBgXaShoUHh4eGKjo52t02ZMkUnT57U2bNnfbgyAAAwVFh8vQB/43Q6ZbVaPdr6H3d2dio0NPSyYzQ1NcnhcHi0WSzBioqKMrdQAACGEYtleJ3zIWBdJDg4WF1dXR5t/Y/HjBnj1RiVlZXasmWLR9ujjz6qH/7wh2YW%2BQ/eLrp/UPs1NTWpsrJSmZmZBL/PgboNDnUbHOo2ONRtcKibWcMrLhoQGxurtrY2NTc3u9uOHTum8ePHa%2BzYsV6NkZmZqf/6r//y%2BMrMzLxaSx4Uh8OhLVu2DDjThs9G3QaHug0OdRsc6jY41M0szmBdZPLkyUpISFBxcbEKCgrU2tqq0tJSpaenez1GVFQU6R8AgOsYZ7A%2BxebNm9XT06M5c%2BbogQce0N133y273e7rZQEAgCGCM1ifIjIyUps3b/b1MgAAwBDFGazr1Lhx4/Too49q3Lhxvl7KkELdBoe6DQ51GxzqNjjUzawRLpfL5etFAAAADCecwQIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsC6Dhw5ckSPPPKIZsyYoVmzZmnlypU6c%2BaMJOngwYPKyMhQfHy8kpOTVVVV5ePV%2Bo%2B6ujplZGTo9ttv16xZs1RYWKju7m5J1M0bvb29euihh/TEE0%2B426jbpe3atUu33HKL4uPj3V95eXmSqNtnaWtr08qVK5WYmKg77rhDdrtdTU1NkqjbpezYscPjdRYfH69bb71Vt956qyTqZowLw1pXV5dr1qxZrk2bNrk%2B/vhj15kzZ1xLly51ff/733e1tbW5ZsyY4XrppZdcFy5ccL355puu%2BPh418GDB329bJ9raWlxTZ061bV161ZXb2%2Bv6/Tp064FCxa4Nm3aRN28tHHjRtdNN93kWrVqlcvlclG3yygpKXE98cQTA9qp22f753/%2BZ1dOTo6rvb3d1dHR4Xr00Uddy5Yto26fw6lTp1yzZs1y/fa3v6VuBnEGa5g7efKkbrrpJuXk5GjUqFGy2WzKzMxUfX29du/erfDwcGVlZclisSgpKUkLFy5URUWFr5ftcxEREXrzzTeVlpamESNGqK2tTR9//LEiIiKomxfq6uq0e/dufe1rX3O3UbfPdvjwYfcZhH9E3S7t3Xff1cGDB1VSUqLQ0FCFhISosLBQK1asoG5ecrlcysvL01e/%2BlV94xvfoG4GEbCGuS996Ut6/vnnFRgY6G577bXX9OUvf1kNDQ2Ki4vz2D4mJkZHjhy51sv0SyEhIZKke%2B65RwsXLtS4ceOUlpZG3S6jpaVFq1ev1jPPPCOr1epup26X1tfXp/fee09//vOfde%2B992r27Nn68Y9/rPb2dur2GQ4dOqSYmBj95je/UUpKiu666y6tX79e48aNo25e2r59uxobG92X8qmbOQSs64jL5dLPfvYzvf7661q9erWcTqfHD0BJCgoKUmdnp49W6J92796tPXv2KCAgQMuXL6dun6Gvr095eXl65JFHdNNNN3n0UbdLO3PmjG655RbNnTtXu3bt0ssvv6y///3vysvLo26fob29XUePHtXf//53bdu2Tb/97W91%2BvRprVq1irp5oa%2BvT2VlZfrBD37g/g8ldTOHgHWdOHfunJYvX66dO3fqpZde0o033iir1eq%2Babtfd3e3xowZ46NV%2BqegoCBFR0crLy9PNTU11O0z/PznP9eoUaP00EMPDeijbpcWGRmpiooKpaeny2q16gtf%2BILy8vK0Z88euVwu6nYJo0aNkiStXr1aISEhioyMVG5urt544w3q5oW9e/eqqalJ6enp7jbep%2BYQsK4DH374oRYvXqxz586purpaN954oyQpLi5ODQ0NHts2NjYqNjbWF8v0K%2B%2B8847uv/9%2BnT9/3t12/vx5jRw5UjExMdTtErZv3659%2B/Zp%2BvTpmj59ul555RW98sormj59Oq%2B3z3DkyBE9/fTTcrlc7rbz588rICBAt912G3W7hJiYGPX19enChQvutr6%2BPknSzTffTN0u47XXXlNKSoqCg4PdbbxPzSFgDXPt7e16%2BOGHdfvtt%2BsXv/iFIiIi3H0pKSlqbm5WeXm5Lly4oLfeeks7d%2B7U4sWLfbhi/3DjjTequ7tbzzzzjM6fP6//%2B7//0/r165Wenq65c%2BdSt0t49dVX9c477%2Bjtt9/W22%2B/rQULFmjBggV6%2B%2B23eb19hvDwcFVUVOj5559XT0%2BPTp48qQ0bNuib3/wmr7fPMHPmTE2cOFFPPvmknE6nzpw5o5/97Ge67777tGDBAup2Gfv379cdd9zh0cb71JwRrn/8LxOGnV/%2B8pcqKSmR1WrViBEjPPoOHDigw4cPq6ioSO%2B//74iIiJkt9uVlpbmo9X6l8bGRhUXF%2Bvw4cMaO3asFi5c6P5tTOrmnf4bZ0tKSiSJun2Gffv26dlnn9X777%2Bv0aNHa/78%2BcrLy9Po0aOp22c4ffq0SkpKVF9fr48//ljJyclavXq1QkNDqdtlxMfHa%2BPGjbrnnns82qmbGQQsAAAAw7hECAAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAY9v8AziG8NHma1RIAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common8661973349582570581">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">28</td>
        <td class="number">15028</td>
        <td class="number">3.6%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">31</td>
        <td class="number">14611</td>
        <td class="number">3.5%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">29</td>
        <td class="number">14193</td>
        <td class="number">3.4%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">32</td>
        <td class="number">13776</td>
        <td class="number">3.3%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">39</td>
        <td class="number">13776</td>
        <td class="number">3.3%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30</td>
        <td class="number">13776</td>
        <td class="number">3.3%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">37</td>
        <td class="number">13358</td>
        <td class="number">3.2%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">33</td>
        <td class="number">12941</td>
        <td class="number">3.1%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">45</td>
        <td class="number">12523</td>
        <td class="number">3.0%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">35</td>
        <td class="number">12134</td>
        <td class="number">2.9%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (45)</td>
        <td class="number">281348</td>
        <td class="number">67.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme8661973349582570581">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">18</td>
        <td class="number">2922</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:70%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">19</td>
        <td class="number">2504</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:60%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">20</td>
        <td class="number">1252</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:30%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">21</td>
        <td class="number">3339</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:80%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">22</td>
        <td class="number">4174</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">68</td>
        <td class="number">2922</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">71</td>
        <td class="number">2504</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:85%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">72</td>
        <td class="number">2087</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:71%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">73</td>
        <td class="number">1252</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:43%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">74</td>
        <td class="number">2087</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:71%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_city">city<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>15</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable5929565719802986504">
    <table class="mini freq">
        <tr class="">
    <th>New York</th>
    <td>
        <div class="bar" style="width:27%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 16.7%">
            69893
        </div>

    </td>
</tr><tr class="">
    <th>Los Angeles</th>
    <td>
        <div class="bar" style="width:20%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 12.6%">
            &nbsp;
        </div>
        52513
    </td>
</tr><tr class="">
    <th>Chicago</th>
    <td>
        <div class="bar" style="width:13%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 7.9%">
            &nbsp;
        </div>
        33043
    </td>
</tr><tr class="other">
    <th>Other values (12)</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 62.8%">
            262015
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable5929565719802986504, #minifreqtable5929565719802986504"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable5929565719802986504">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">New York</td>
        <td class="number">69893</td>
        <td class="number">16.7%</td>
        <td>
            <div class="bar" style="width:78%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Los Angeles</td>
        <td class="number">52513</td>
        <td class="number">12.6%</td>
        <td>
            <div class="bar" style="width:58%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Chicago</td>
        <td class="number">33043</td>
        <td class="number">7.9%</td>
        <td>
            <div class="bar" style="width:37%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Philadelphia</td>
        <td class="number">28756</td>
        <td class="number">6.9%</td>
        <td>
            <div class="bar" style="width:32%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Dallas</td>
        <td class="number">26681</td>
        <td class="number">6.4%</td>
        <td>
            <div class="bar" style="width:30%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Boston</td>
        <td class="number">24418</td>
        <td class="number">5.8%</td>
        <td>
            <div class="bar" style="width:27%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Houston</td>
        <td class="number">24391</td>
        <td class="number">5.8%</td>
        <td>
            <div class="bar" style="width:27%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Atlanta</td>
        <td class="number">24326</td>
        <td class="number">5.8%</td>
        <td>
            <div class="bar" style="width:27%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">San Francisco</td>
        <td class="number">24300</td>
        <td class="number">5.8%</td>
        <td>
            <div class="bar" style="width:27%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Phoenix</td>
        <td class="number">19332</td>
        <td class="number">4.6%</td>
        <td>
            <div class="bar" style="width:22%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (5)</td>
        <td class="number">89811</td>
        <td class="number">21.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div>
    <div class="row headerrow highlight">
        <h1>Correlations</h1>
    </div>
    <div class="row variablerow">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmoAAAIfCAYAAADAPfANAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAABCqklEQVR4nO3dfXzPhf7/8ednNnZBRjbL5TRbDmJjGkLM5JCrU1gHIReVZS5CF7%2BSJOF8d7qyOpXEKYpyFVGccLQ6rsrIRbQZWsbmMnZlm31%2Bf/j6fPucjU3vvfu8N4/77bbbsffVXp%2B3Y59nz/f78/nY7Ha7XQAAALAcN1cPAAAAgOIR1AAAACyKoAYAAGBRBDUAAACLIqgBAABYFEENAADAoghqAAAAFkVQAwAAsCiCGgAAgEW5u3oAAH%2BMX375RV27dr3meg8PD1WtWlWBgYHq3LmzhgwZoqpVq/6BEwIA/puNj5ACbg6/DWohISFFQlh%2Bfr7Onj2r48ePS5Lq1KmjhQsXqmHDhn/4rACAKwhqwE3it0Htgw8%2BUERERLHbbd%2B%2BXTExMcrMzFRYWJiWLFnyR44JAPgN7lED4CQiIkJPPPGEJCkxMVH79u1z8UQAcPMiqAEoolu3bo4/79mzx4WTAMDNjRcTACiiWrVqjj9nZWU5rdu5c6c%2B/PBD7dq1S%2BfPn9ctt9yi0NBQPfTQQ2rXrl2xx7tw4YKWLFmiLVu2KDk5WZmZmfLy8lKDBg3UpUsXDR06VNWrV3fa54477pAkffvtt5o9e7Y2btwoNzc3NWvWTO%2B//77c3d21Z88eLVy4UAcOHNCJEydUpUoVNWrUSFFRURo0aFCxL4bIzc3VkiVLtG7dOiUnJys/P1%2B1a9dW%2B/btNWLECAUGBjptv337dg0dOlQtW7bU4sWL9eGHH2rVqlU6duyYPDw81KxZMz300EOKior6PacaAK6LoAagiGPHjjn%2BHBAQ4PhzXFyc5s2bJ0mqXr26QkJClJGRoY0bN2rjxo0aNWqUpkyZ4nSso0ePavjw4Tpx4oTc3d3VoEED1a1bV8ePH9f%2B/fu1f/9%2BrV27VsuXL5ePj0%2BRWWJjY5WYmKiQkBCdPXtWfn5%2Bcnd314YNGzRx4kQVFBSoRo0aaty4sbKysvTDDz9oz549Wr16tZYsWeIU1k6ePKmHH35YKSkpkqTAwED5%2BPjo8OHDWrp0qVatWqXZs2erZ8%2BeRebIz8/X6NGjtXXrVtWoUUNBQUE6cuSItm3bpm3btumFF17QX//6V2MnHgD%2Bmx3ATSE1NdUeEhJiDwkJsW/btu262z755JP2kJAQe7NmzeynTp2y2%2B12%2B8cff2wPCQmxh4eH2z/77DPHtoWFhfa1a9faQ0ND7SEhIfZPPvnE6VhDhgyxh4SE2AcOHGhPT0932m/lypX2Jk2a2ENCQuyLFi1y2u/qrM2bN7fv2LHDbrfb7ZcvX7afO3fOfvnyZfvdd99tDwkJsc%2BbN89eUFDg2G/fvn32tm3b2kNCQuzvvPOOY3lBQYG9b9%2B%2B9pCQEHv37t3tP/74o2PdxYsX7c8%2B%2B6zjMe/evduxbtu2bY5ZQkND7atXr3asu3Dhgn3YsGH2kJAQ%2B1133WXPz8%2B/7nkFgBvFPWoAJF25JHjgwAFNmzZNq1atkiQNHz5ctWrVUl5enubOnStJevnll9WnTx/HfjabTT179nQ0aXPnzlVBQYEk6cyZM0pKSpIkzZgxQ/7%2B/k779evXT3fddZck6dChQ8XO1aNHD7Vp00aS5ObmJl9fX509e1anTp2SJA0cOFCVKlVybN%2BsWTNNnDhRUVFR8vX1dSz/8ssv9eOPP6pKlSqaN2%2BemjRp4lhXtWpVvfTSS%2BrYsaPy8/P16quvFjvLuHHj1Lt3b8f31apVczzu8%2BfP68iRI8XuBwC/F5c%2BgZvQ0KFDS9xmwIABGj9%2BvKQrr/48ffq0fHx8rvmmuX369NGMGTOUnp6uAwcOqEWLFrr11lu1bds25ebmytPTs8g%2Bly9fdlyazM3NLfa4rVu3LrKsRo0aql69un799VdNnjxZY8aMUcuWLeXmduW/PQcOHKiBAwc67bNp0yZJUmRkpOrXr1/sz3r44YeVkJCgHTt26OLFi0736klSly5diuwTFBTk%2BPOFCxeKPS4A/F4ENeAm9N9veGuz2VSlShX5%2BvrqjjvuUFRUlBo3buxYf7UVy8/P1%2BDBg6953EqVKqmwsFApKSlq0aKFY7mnp6dOnDihPXv26Oeff1ZqaqoOHz6sH3/8UdnZ2ZKkwsLCYo/p5%2BdX7M%2BZPHmypk6dqi1btmjLli2qXr26IiIidPfdd6tz585O99ZJcrRdzZo1u%2Bb8V9ddvnxZx44dU/PmzZ3W165du8g%2Bvw2gly9fvuaxAeD3IKgBN6Hnnnvumm94W5yLFy9KkvLy8rRr164St/9ts5SSkqK//e1v2rJli1MYq1q1qsLDw5WRkaGDBw9e81jFNXHSldasYcOGWrBggf7zn//o119/1YYNG7RhwwbZbDZ17txZL7zwgiOwZWZmSlKRluy3fhte//vVrtKVj9m6HjvvHw6gjBHUAJTIy8tL0pXGacWKFaXe78yZMxoyZIjOnDmjOnXqaODAgWratKluv/121atXTzabTZMmTbpuULueiIgIRUREKDc3V99995127typhIQE7d%2B/X5s3b9aJEye0atUq2Ww2xytKr4bO4vw2YBb3ClQA%2BKMR1ACUqFGjRpKuvNVGQUGB3N2L/uqw2%2B3avn27AgICVKdOHVWuXFnLly/XmTNn5Ovrq%2BXLl6tmzZpF9ktPT7/hefLy8pSamqrMzEy1bNlSnp6e6tChgzp06KCJEydq7dq1euKJJ3Tw4EEdOnRITZo00e23364DBw5o//791zzu3r17JV25FNygQYMbngsAyhqv%2BgRQojZt2qhatWrKysq6ZqO2Zs0aDRs2TD169NDJkyclXfl8UenKB7wXF9KSk5O1e/duSTd2f9fXX3%2Btnj176pFHHlFeXl6R9e3bt3f8%2Bepxr74QYNOmTUpNTS32uB988IEkKTQ0VLfcckup5wEAsxDUAJTI29tbjzzyiCRp5syZWr58udP9Zl999ZWmTZsm6crbaVxto26//XZJ0sGDB7V%2B/XrH9na7XV9//bVGjRql/Px8SVJOTk6p5%2BnUqZNq1Kih8%2BfP66mnntL58%2Bcd67KysjRnzhxJ0m233abg4GBJ0p///GfdcccdunTpkkaPHu10uTUzM1NTp07VN998I3d3d02ePLnUswCAmbj0CaBURo8erdTUVH3yySf6f//v/%2Bl//ud/VK9ePaWnpysjI0OS1KpVK7300kuOffr376%2BPPvpIx44d07hx41S3bl3VqFFDJ06c0JkzZ%2BTh4aG77rpLO3bsuKFLoJUrV9brr7%2BukSNHat26ddq4caMaNGggNzc3paamKjs7W15eXpo9e7YqV64sSXJ3d9dbb72l0aNHKyUlRX379nX6ZIKrbyEyffp0hYeHl%2B3JA4DfiaAGoFRsNptmzJih7t27a8mSJdq9e7fjDWRDQ0PVq1cvRUdHO4KRdOVVlMuWLdO8efO0efNm/fLLLzp9%2BrQCAgLUuXNnDRs2TN7e3oqKitLBgweVlpamOnXqlGqeiIgIffrpp1qwYIG%2B//57HT16VO7u7goICFCHDh00YsSIIseqV6%2Beli9fro8//lhffvmlDh8%2BrJMnT%2Bq2225Tx44dNXjw4CKf9QkArmSz83pyAAAAS%2BIeNQAAAIsiqAEAAFgUQQ0AAMCiCGoAAOCmdPbsWXXr1k3bt2%2B/5jZbtmxR7969FRoaqh49emjz5s1O6%2BfNm6dOnTopNDRUDz30kFJSUsp0RoIaAAC46Xz//feKjo7Wzz//fM1tjh49qtjYWI0fP17fffedYmNjNWHCBMfbCa1cuVIffvih5s%2Bfr%2B3bt6tZs2YaN25cmX7uL0ENAADcVFauXKnJkydr4sSJJW4XHh6uqKgoubu7q2fPnmrTpo2WLl0qSfrkk080aNAgBQcHq0qVKpo0aZLS0tKu29DdKIIaAAAoVzIyMrR//36nr6tvvF0aHTp00L/%2B9S/17NnzutslJycrJCTEaVnjxo0dn2zy3%2Bs9PDwUGBjo9MknRvGGt2XJZnP1BChJo0ZSUpIUHCwdOeLqaXAd9kLe4hEoKy59ejLhhy994w3Fx8c7LRs7dqxiY2NLtb%2Bfn1%2BptsvKypKXl5fTMk9PT2VnZ5dqfVkgqOHm4usrVap05X8BGGazSbxtOv5o0dHRioyMdFpW2vB1I7y8vJSbm%2Bu0LDc3Vz4%2BPqVaXxYIagAAwDxuZX%2BXlb%2B/v/z9/cv8uP8tJCRE%2B/fvd1qWnJys5s2bS5KCg4OVlJSkLl26SJLy8/N19OjRIpdLjeAeNQAAgGL06dNHO3bs0Lp161RQUKB169Zpx44d6tu3ryTpgQce0KJFi3Tw4EFdunRJf//731WrVi2Fh4eX2Qw0agAAwDwmNGpmCgsL0/Tp09WnTx8FBQXpzTffVFxcnJ599lnVrVtXc%2BfOVaNGjSRJ/fv318WLF/X444/r7NmzuvPOO/XOO%2B/Iw8OjzObhQ9nLEi8msL6wMGnXLqlVKykx0dXT4Dp4MUH5wD1q5YNLn57KMLQ45OeX/TEtikYNAACYp5w1alZDUAMAAOYhqBnC2QMAALAoGjUAAGAeGjVDCGoAAMA8BDVDOHsAAAAWRaMGAADMQ6NmCGcPAADAomjUAACAeWjUDCGoAQAA8xDUDOHsAQAAWBSNGgAAMA%2BNmiGcPQAAAIuiUQMAAOahUTOEoAYAAMxDUDOEswcAAGBRNGoAAMA8NGqGcPYAAAAsikYNAACYh0bNEIIaAAAwD0HNEM4eAACARdGoAQAA89CoGcLZAwAAsCgaNQAAYB4aNUMIagAAwDwENUM4ewAAABZFowYAAMxDo2YIZw8AAMCiaNQAAIB5aNQMIagBAADzENQM4ewBAABYFI0aAAAwD42aIZw9AAAAi6JRAwAA5qFRM4SgBgAAzENQM4SzBwAAYFE0agAAwDw0aoZw9gAAACyKRg0AAJiHRs0QghoAADAPQc0Qzh4AAIBF0agBAADz0KgZwtkDAACwKBo1AABgHgs2amfOnNHUqVO1Y8cOVapUSX369NFTTz0ld3fnWDRq1Ch9//33Tsuys7MVHR2tF198UYWFhWrdurXsdrtsNptjm2%2B//Vbe3t5lMitBDQAAmMeCQW3ChAmqXbu2EhISdPr0aY0ZM0YLFy7UqFGjnLZ77733nL5ftmyZ4uPjNXbsWElScnKy8vPztWvXLlWuXNmUWa139gAAAExy7Ngx7dixQ1OmTJGXl5fq16%2BvmJgYLV68%2BLr7paSkaMaMGYqLi5O/v78kae/evbrjjjtMC2kSjRoAADCTCY1aRkaGTp065bTMz8/PEaCuJykpSb6%2Bvqpdu7ZjWVBQkNLS0nThwgXdcsstxe43ffp09evXT%2BHh4Y5le/fu1aVLl/TAAw/o%2BPHjCgoK0qRJk9SqVavf%2BciKIqgBAADzmBDUli5dqvj4eKdlY8eOVWxsbIn7ZmVlycvLy2nZ1e%2Bzs7OLDWrfffed9uzZo7i4OKflnp6eatGihcaPH6/q1atr8eLFGjlypFavXq369evf6MMqFkENAACUK9HR0YqMjHRa5ufnV6p9vb29lZOT47Ts6vc%2BPj7F7rN06VL16NGjyM94%2Bumnnb4fOXKkVqxYoS1btmjIkCGlmqckBDUAAGAeExo1f3//Ul3mLE5wcLDOnz%2Bv06dPq1atWpKkw4cPKyAgQNWqVSuyfUFBgTZu3Kg333yzyLpXX31V3bt3V9OmTR3L8vLyVKVKld81W3F4MQEAALhpBAYGqnXr1nr55ZeVmZmp1NRUvfXWW%2Brfv3%2Bx2x86dEiXLl0q9r6zn376STNnztSpU6eUl5en%2BPh4ZWZmqlu3bmU2L0ENAACYx82t7L8MeuONN1RQUKCuXbtq4MCB6tixo2JiYiRJYWFhWr16tWPb1NRUVa9evdiWbNasWWrQoIH69u2riIgI7dixQwsWLJCvr6/hGa%2By2e12e5kd7Wb3mze7g0WFhUm7dkmtWkmJia6eBtdhL%2BRXU3lgs0k8i1ifS5%2Beuncv%2B2OuX1/2x7QoGjUAAACL4sUEAADAPBb8ZILyhLMHAABgUTRqAADAPDRqhhDUAACAeQhqhnD2AAAALIpGDQAAmIdGzRDOHgAAgEXRqAEAAPPQqBlCUAMAAOYhqBnC2QMAALAoGjUAAGAeGjVDOHsAAAAWRaMGAADMQ6NmCEENAACYh6BmCGcPAADAomjUAACAeWjUDOHsAQAAWBSNGgAAMA%2BNmiEENQAAYB6CmiGcPQAAAIuiUQMAAOahUTOEswcAAGBRNGoAAMA8NGqGENQAAIB5CGqGcPYAAAAsikYNAACYh0bNEM4eAACARdGoAQAA89CoGUJQAwAA5iGoGcLZAwAAsCgaNQAAYB4aNUM4ewAAABZFowYAAMxDo2YIQQ0AAJiHoGYIZw8AAMCiaNQAAIB5aNQMIagBAADzENQM4ewBAABYFI0aAAAwD42aIZw9AAAAi6JRAwAA5qFRM4SgBgAAzENQM4SzBwAAbipnzpxRTEyMwsPDFRERoZkzZ6qgoKDYbUeNGqU777xTYWFhjq%2Bvv/7asX7evHnq1KmTQkND9dBDDyklJaVMZzUc1NLS0hQWFqa0tLSymMcUJc24YsUKRUZG/sFTAQBwE3BzK/svgyZMmCBvb28lJCRo2bJl2rp1qxYuXFjstvv27dP8%2BfOVmJjo%2BOrUqZMkaeXKlfrwww81f/58bd%2B%2BXc2aNdO4ceNkt9sNz3iV4Udbp04dJSYmqk6dOmUxjynKw4wAAMB8x44d044dOzRlyhR5eXmpfv36iomJ0eLFi4tsm5qaql9//VVNmzYt9liffPKJBg0apODgYFWpUkWTJk1SWlqatm/fXmbzlvoetSeffFKXL1/W3//%2Bd8eyCRMmKDs7W1u2bNHGjRtVr149nT59WrNnz9bWrVtls9kUGRmpJ598Ut7e3urQoYNefPFFRUVFSZIiIyPVokULvfbaa5KkOXPm6MyZM/rb3/6m/fv3a/bs2Tp48KBq1KihQYMGadiwYbLZbJo7d64SExP166%2B/KjU1VW%2B%2B%2BabatGlzzdl/%2BeUXde3a1THj4cOH9cILL2jfvn2qV6%2BeIiIibvjEZWRk6NSpU07L/Bo1kr%2Bv7w0fC3%2BgJk2c/xcAYC4T7lEr9jnYz0/%2B/v4l7puUlCRfX1/Vrl3bsSwoKEhpaWm6cOGCbrnlFsfyvXv3ysfHRxMnTtTevXtVq1YtDR8%2BXP3795ckJScna/To0Y7tPTw8FBgYqIMHD6pt27ZGH6akGwhqAwcO1MiRI5WZmamqVavqwoUL2rRpkz7%2B%2BGNt2bJFklRYWKiYmBgFBgZq/fr1ys/P1zPPPKPnn39er7zyiiIjI/X1118rKipKKSkpOnPmjLZu3Sq73S6bzaZNmzZpypQpSk9P17BhwzRx4kS9//77OnbsmGJiYuTp6akHH3xQkrR161a9//77atGihapUqVLqB5yfn69HH31UnTp10nvvvaeff/5Zo0ePltsN/h9p6dKlio%2BPd1o2dvx4xY4ff0PHgYt89JGrJ0AJbK4eAKVm4y8L12NCUCv2OXjsWMXGxpa4b1ZWlry8vJyWXf0%2BOzvbKajl5eUpNDRUEydOVHBwsLZv367Y2Fj5%2BPioR48exR7L09NT2dnZv/ehFVHqoBYeHq7bbrtNX3zxhQYMGKDPP/9ct99%2Bu6pXr%2B7YZt%2B%2Bfdq/f78WLFggHx8fSdJTTz2lP//5z5o6daqioqL0wgsvSJK%2B%2BeYb9ezZU1999ZUOHDggT09PZWRkqEOHDvrwww8VFBSkwYMHS5IaN26skSNHatGiRY6gVr9%2BfbVr1%2B6GH3BiYqJOnDihJ598UlWqVFFwcLAefvhh/fOf/7yh40RHRxe5r82vd2/pBo%2BDP1iTJldC2qBB0sGDrp4G12H/fperR0Ap2GxSGd6OA5NUtDBd7HOwn1%2Bp9vX29lZOTo7TsqvfX80uV/Xr10/9%2BvVzfN%2BhQwf169dPX3zxhXr06CEvLy/l5uY67ZObm1vkOEbc0NtzDBgwQJ999pkGDBiglStXasCAAU7rf/nlF12%2BfFn33HOP0/LKlSsrNTVV7du314ULF5SUlKSEhAT169dPFy5c0H/%2B8x/Z7XZ17NhRnp6eOn78uPbv36/w8HDHMQoLC1WpUiXH96WpN4uTnp6uGjVqyNPT07GsQYMGN3wcf3//ojMcOfK7ZoILHDwoJSa6egoAqPhMaNSKfQ4upeDgYJ0/f16nT59WrVq1JEmHDx9WQECAqlWr5rTtsmXLHO3ZVXl5eY4recHBwUpKSlKXLl0kXblqd/ToUYWEhPyu2YpzQ0HtL3/5i1577TX95z//0aFDh9SrVy9dvHjRsT4gIECenp7avn27I1Tl5eUpNTVVDRs2lLu7uzp27KiNGzfq%2B%2B%2B/15w5c3ThwgX961//Uk5OjqNBCwgIUEREhObPn%2B849rlz55SVleX43vY7//Pgtttu09mzZ5WVleVIvCdPnvxdxwIAAOVLYGCgWrdurZdfflkvvviizp07p7feestx39lvZWZm6pVXXlHDhg3VpEkTff311/r8888d%2BeSBBx7Q3Llz1alTJzVq1EivvvqqatWq5VQ0GXVDMbdmzZrq0qWLnnvuOd17771Olz0lqUWLFmrYsKFmz56trKws5ebm6uWXX9bw4cN1%2BfJlSVK3bt20cOFCBQYGqmbNmurQoYO%2B%2B%2B47HThwQJ07d5Yk9e7dW7t379bq1atVUFCgjIwMPfbYY5o9e7bhBxwWFqZGjRrppZdeUk5Ojo4dO6b333/f8HEBAEAxLPj2HG%2B88YYKCgrUtWtXDRw4UB07dlRMTIykKzlh9erVkqRhw4ZpyJAhGjt2rMLCwhQXF6c5c%2BY4glj//v01fPhwPf7442rbtq0OHDigd955Rx4eHoZnvOqGP5lg4MCBWr9%2BvV5%2B%2BeWiB3N31zvvvKM5c%2Bbo3nvv1aVLl9SiRQstWLDAURN27txZTz/9tDp06CDpyr1mAQEBCgwMVNWqVSVJdevW1Xvvvae4uDi99NJLqlSpkjp37qxnn33WyGOVJFWqVEnvvvuunn/%2BebVv3161atVS165dtWHDBsPHBgAA/8WCn0xQq1YtvfHGG8WuS/zNbTE2m00xMTGOEPffbDabRowYoREjRpgypyTZ7GX5rmw3u4p2t2ZFFBYm7doltWrFPWoWZy/kV1N5wIsJygeXPj3NmlX2x3zmmbI/pkXxWZ8AAMA8FmzUypMKEdQiIiKUl5d3zfVr167lUwkAAEC5UyGCWll%2BVAMAAChDNGqGVIigBgAALIqgZghnDwAAwKJo1AAAgHlo1Azh7AEAAFgUjRoAADAPjZohBDUAAGAegpohnD0AAACLolEDAADmoVEzhLMHAABgUTRqAADAPDRqhhDUAACAeQhqhnD2AAAALIpGDQAAmIdGzRDOHgAAgEXRqAEAAPPQqBlCUAMAAOYhqBnC2QMAALAoGjUAAGAeGjVDOHsAAAAWRaMGAADMQ6NmCEENAACYh6BmCGcPAADAomjUAACAeWjUDOHsAQAAWBSNGgAAMA%2BNmiEENQAAYB6CmiGcPQAAAIuiUQMAAOahUTOEoAYAAMxDUDOEswcAAGBRNGoAAMA8NGqGcPYAAAAsikYNAACYh0bNEIIaAAAwD0HNEM4eAACARdGoAQAA89CoGcLZAwAAsCgaNQAAYB4aNUMIagAAwDwENUMIagAA4KZy5swZTZ06VTt27FClSpXUp08fPfXUU3J3LxqLPv74Yy1cuFAZGRny9/fX0KFDNXjwYElSYWGhWrduLbvdLpvN5tjn22%2B/lbe3d5nMSlADAADmsWCjNmHCBNWuXVsJCQk6ffq0xowZo4ULF2rUqFFO23311Vd65ZVXNG/ePLVs2VK7d%2B/WI488olq1aql79%2B5KTk5Wfn6%2Bdu3apcqVK5syq/XOHgAAgEmOHTumHTt2aMqUKfLy8lL9%2BvUVExOjxYsXF9k2PT1do0ePVmhoqGw2m8LCwhQREaGdO3dKkvbu3as77rjDtJAm0agBAAAzmdCoZWRk6NSpU07L/Pz85O/vX%2BK%2BSUlJ8vX1Ve3atR3LgoKClJaWpgsXLuiWW25xLL96ifOqM2fOaOfOnXrmmWckXQlqly5d0gMPPKDjx48rKChIkyZNUqtWrYw8PCcENQAAYB4TgtrSpUsVHx/vtGzs2LGKjY0tcd%2BsrCx5eXk5Lbv6fXZ2tlNQ%2B61Tp07p0UcfVfPmzdWrVy9Jkqenp1q0aKHx48erevXqWrx4sUaOHKnVq1erfv36v%2BehFUFQAwAA5Up0dLQiIyOdlvn5%2BZVqX29vb%2BXk5Dgtu/q9j49Psfvs3r1b48ePV3h4uGbNmuV40cHTTz/ttN3IkSO1YsUKbdmyRUOGDCnVPCUhqAEAAPOY0Kj5%2B/uX6jJncYKDg3X%2B/HmdPn1atWrVkiQdPnxYAQEBqlatWpHtly1bppdeeknjxo3TiBEjnNa9%2Buqr6t69u5o2bepYlpeXpypVqvyu2YrDiwkAAMBNIzAwUK1bt9bLL7%2BszMxMpaam6q233lL//v2LbLt%2B/Xq98MILmjt3bpGQJkk//fSTZs6cqVOnTikvL0/x8fHKzMxUt27dymxeghoAADCPm1vZfxn0xhtvqKCgQF27dtXAgQPVsWNHxcTESJLCwsK0evVqSVJ8fLwuX76scePGKSwszPH1/PPPS5JmzZqlBg0aqG/fvoqIiNCOHTu0YMEC%2Bfr6Gp7xKpvdbreX2dFudr95sztYVFiYtGuX1KqVlJjo6mlwHfZCfjWVBzabxLOI9bn06Wnr1rI/Zrt2ZX9Mi6JRAwAAsCheTAAAAMxjwU8mKE84ewAAABZFowYAAMxDo2YIQQ0AAJiHoGYIZw8AAMCiaNQAAIB5aNQM4ewBAABYFI0aAAAwD42aIQQ1AABgHoKaIZw9AAAAi6JRAwAA5qFRM4SzBwAAYFE0agAAwDw0aoYQ1AAAgHkIaoZw9gAAACyKRg0AAJiHRs0Qzh4AAIBF0agBAADz0KgZQlADAADmIagZwtkDAACwKBo1AABgHho1QwhqAADAPAQ1Qzh7AAAAFkWjBgAAzEOjZghnDwAAwKJo1AAAgHlo1AwhqAEAAPMQ1Azh7AEAAFgUjRoAADAPjZohnD0AAACLolEDAADmoVEzhKAGAADMQ1AzhLMHAABgUTRqAADAPDRqhnD2AAAALIpGDQAAmIdGzRCCGgAAMA9BzRDOHgAAgEXRqAEAAPPQqBnC2QMAALAoGjUAAGAeGjVDCGoAAMA8BDVDOHsAAOCmcubMGcXExCg8PFwRERGaOXOmCgoKit12y5Yt6t27t0JDQ9WjRw9t3rzZaf28efPUqVMnhYaG6qGHHlJKSkqZzkpQAwAA5nFzK/svgyZMmCBvb28lJCRo2bJl2rp1qxYuXFhku6NHjyo2Nlbjx4/Xd999p9jYWE2YMEHp6emSpJUrV%2BrDDz/U/PnztX37djVr1kzjxo2T3W43PONVBDUAAHDTOHbsmHbs2KEpU6bIy8tL9evXV0xMjBYvXlxk25UrVyo8PFxRUVFyd3dXz5491aZNGy1dulSS9Mknn2jQoEEKDg5WlSpVNGnSJKWlpWn79u1lNi/3qAEAAPOYcI9aRkaGTp065bTMz89P/v7%2BJe6blJQkX19f1a5d27EsKChIaWlpunDhgm655RbH8uTkZIWEhDjt37hxYx08eNCxfvTo0Y51Hh4eCgwM1MGDB9W2bdvf9dj%2BG0GtDNkLy67qhHlskuzf73L1GCiBzc3m6hFQkrAwadcu2Vq3khITXT0NrqcML8Xd8I9W2f9bXrp0qeLj452WjR07VrGxsSXum5WVJS8vL6dlV7/Pzs52CmrFbevp6ans7OxSrS8LBDUAAFCuREdHKzIy0mmZn59fqfb19vZWTk6O07Kr3/v4%2BDgt9/LyUm5urtOy3Nxcx3YlrS8LBDUAAGCawsKyP6a/v3%2BpLnMWJzg4WOfPn9fp06dVq1YtSdLhw4cVEBCgatWqOW0bEhKi/fv3Oy1LTk5W8%2BbNHcdKSkpSly5dJEn5%2Bfk6evRokculRvBiAgAAcNMIDAxU69at9fLLLyszM1Opqal666231L9//yLb9unTRzt27NC6detUUFCgdevWaceOHerbt68k6YEHHtCiRYt08OBBXbp0SX//%2B99Vq1YthYeHl9m8NntZvob0JseZLB9sNv6uygPuUSsH/vceNbXiHjXLc%2BEvvfz8sj%2Bmh4ex/U%2BfPq0XX3xR27dvl5ubm/r166fJkyerUqVKCgsL0/Tp09WnTx9JUkJCguLi4vTzzz%2Brbt26mjJliu655x5Jkt1u14IFC7R48WKdPXtWd955p6ZPn65GjRoZfYgOBLUyxJksHwhq5QNBrRwgqJUfLvyld%2BlS2R%2BzSpWyP6ZVcekTAADAongxAQAAMI0ZLya4mdCoAQAAWBSNGgAAMA2NmjEENQAAYBqCmjFc%2BgQAALAoGjUAAGAaGjVjaNQAAAAsikYNAACYhkbNGIIaAAAwDUHNGC59AgAAWBSNGgAAMA2NmjEENQAAYBqCmjFc%2BgQAALAoGjUAAGAaGjVjaNQAAAAsikYNAACYhkbNGIIaAAAwDUHNGC59AgAAWBSNGgAAMA2NmjE0agAAABZFowYAAExDo2YMQQ0AAJiGoGYMlz4BAAAsikYNAACYhkbNGBo1AAAAi6JRAwAApqFRM4agBgAATENQM4ZLnwAAABZFowYAAExDo2YMjRoAAIBF0agBAADT0KgZQ1ADAACmIagZw6VPAAAAi6JRAwAApqFRM4ZGDQAAwKJo1AAAgGlo1IwhqAEAANMQ1Izh0icAAIBF0agBAADT0KgZQ6MGAABgUTRqAADANDRqxhDUAACAaQhqxhDUAAAA/ld2drZmzJihTZs2qaCgQF27dtW0adPk4%2BNT7Pbr16/XW2%2B9pdTUVPn6%2Bur%2B%2B%2B9XTEyM3Nyu3F3Wo0cPpaWlOb6XpGXLlikoKKhU8xDUAACAacpbozZjxgydOHFC69ev1%2BXLlzVhwgTFxcVp2rRpRbbdt2%2BfnnzySb322mu65557dOTIEY0ePVre3t4aMWKEMjMzdeTIEW3cuFF169b9XfPwYgIAAABJOTk5WrNmjcaNGydfX1/deuutmjx5slasWKGcnJwi2x8/flwPPvigunTpIjc3NwUFBalbt27auXOnpCtBztfX93eHNIlGDQAAmMiMRi0jI0OnTp1yWubn5yd/f/8S983NzVV6enqx63JycpSfn6%2BQkBDHsqCgIOXm5uro0aP605/%2B5LR99%2B7d1b17d6dj//vf/1bv3r0lSXv37pWXl5eGDBmipKQk1a1bV7GxserSpUupHytBDQAAmMaMoLZ06VLFx8c7LRs7dqxiY2NL3HfPnj0aOnRosevGjx8vSfL29nYs8/LykiRlZWVd97iZmZkaP368PD09NXz4cEmSzWbTnXfeqSeeeEJ16tTRl19%2BqdjYWC1atEihoaElzioR1AAAQDkTHR2tyMhIp2V%2Bfn6l2jciIkKHDh0qdt2BAwf0%2BuuvKycnx/HigauXPKtWrXrNY6akpGjcuHG69dZb9cEHHzi2HTVqlNN2ffr00eeff67169cT1AAAgOuZ0aj5%2B/uX6jLnjWrUqJE8PDyUnJysli1bSpIOHz4sDw8PBQYGFrvPli1b9MQTT2jgwIGaNGmS3N3/L1rNnz9fTZs2Vbt27RzL8vLyVKVKlVLPxIsJAAAAdOUyZ48ePRQXF6ezZ8/q7NmziouLU69eveTp6Vlk%2B927d%2Bvxxx/XM888o6eeesoppEnSiRMnNH36dKWmpqqgoEDLli1TYmKi/vKXv5R6JpvdbrcbfmSQJHEmywebjb%2Br8sDmZnP1CChJWJi0a5fUqpWUmOjqaXA9Lvyl969/lf0xu3Ur%2B2NelZmZqTlz5mjTpk3Kz89X165dNXXqVMd9a/fdd5969%2B6txx57TI899pj%2B/e9/O%2B5ju6p169Z67733lJeXp7i4OH3xxRe6ePGiGjdurClTpigiIqLU8xDUyhBnsnwgqJUPBLVygKBWfrjwl9769WV/zN%2B80LLC49InAACARfFiAgAAYJry9skEVkNQAwAApiGoGcOlTwAAAIuiUQMAAKahUTOGRg0AAMCiaNQAAIBpaNSMIagBAADTENSM4dInAACARdGoAQAA09CoGUOjBgAAYFE0agAAwDQ0asYQ1AAAgGkIasZw6RMAAMCiaNQAAIBpaNSMoVEDAACwKBo1AABgGho1YwhqAADANAQ1Y7j0CQAAYFE0agAAwDQ0asbQqAEAAFgUjRoAADANjZoxBDUAAGAagpoxXPoEAACwKBo1AABgGho1Y2jUAAAALIpGDQAAmIZGzRiCGgAAMA1BzRgufQIAAFgUjRoAADANjZoxNGoAAAAWRaMGAABMQ6NmDEENAACYhqBmDJc%2BAQAALIpGDQAAmIZGzRgaNQAAAIuiUQMAAKahUTOGoAYAAExDUDOGS58AAAAWRaMGAABMQ6NmDI0aAACARdGoAQAA09CoGUOjBgAATFNYWPZfZsrOztYzzzyjiIgItW7dWk8%2B%2BaSysrKuuf20adPUvHlzhYWFOb6WLl3qWL9y5Up169ZNoaGhuv/%2B%2B5WYmHhD8xDUAAAA/teMGTN04sQJrV%2B/Xhs2bNCJEycUFxd3ze337t2rGTNmKDEx0fEVHR0tSdq%2BfbtmzJih2bNna%2BfOnerTp4/GjBmjnJycUs9TroPapk2b9OCDD6pdu3Zq2bKlhgwZoqNHj0qS1q5dq%2B7duys8PFwjR47U1KlT9fTTT0uS7Ha7PvjgA8f6QYMGad%2B%2BfS58JAAAVEzlqVHLycnRmjVrNG7cOPn6%2BurWW2/V5MmTtWLFimLDVV5enn766Sc1b9682ON9%2Bumnuu%2B%2B%2B9S6dWt5eHho%2BPDhqlGjhtatW1fqmcptUDt58qTGjx%2BvRx55RFu3btW///1v2e12vfnmm0pMTNRTTz2lp556Stu2bdODDz6oFStWOPb96KOPtGDBAr3%2B%2BuvaunWr7r//fj388MM6ffq0Cx8RAAAwW25uro4dO3bNr/z8fIWEhDi2DwoKUm5urqMI%2Bq2DBw%2BqoKBAb7zxhtq3b6/u3bvr3XffVeH/psnk5GSnY0lS48aNdfDgwVLPW25fTFCzZk2tXbtWDRo0UGZmpk6ePKkaNWooPT1dy5cv17333qvIyEhJUrdu3RQVFeXYd/HixXr00UfVpEkTSVL//v21bNkyrV69WiNGjCjVz8/IyNCpU6ecltWq5Sd/f/8yeoTATS4szNUToCT/%2BzvU8b9AMcxowIp7DvbzK91z8J49ezR06NBi140fP16S5O3t7Vjm5eUlScXep3bx4kXdddddeuihh/TKK6/oxx9/1OOPPy43NzeNGjVKWVlZjv2v8vT0VHZ2dolzXlVug5qHh4c%2B//xzLVmyRDabTSEhIcrMzJS7u7tOnDihpk2bOm1fv359R2N2/PhxzZkzx%2Bmac0FBwTWry%2BIsXbpU8fHxTssef3ysxo2LNfCo8Eex2Vw9AUq0a5erJ0BpffSRqyeAhZkR1Ip7Dh47dqxiY0t%2BDo6IiNChQ4eKXXfgwAG9/vrrysnJkY%2BPjyQ5LnlWrVq1yPZ333237r77bsf3LVq00LBhw7Ru3TqNGjVKXl5eys3NddonNzdXNWrUKHHOq8ptUPviiy%2B0aNEiffzxx2rYsKGkKzcA/vTTT6pbt67S0tKctk9LS1PlypUlSQEBARo3bpzuu%2B8%2Bx/qff/5Zvr6%2Bpf750dHRjsbuqlq1/GS3/84HhD%2BMzSb%2BnsoBW%2BtWrh4BJWnS5EpIGzRIuoFLOXCBCvYfPsU9B/v5%2BRk%2BbqNGjeTh4aHk5GS1bNlSknT48GF5eHgoMDCwyPZfffWVTp8%2BrQcffNCxLC8vT56enpKk4OBgJSUlOe2TnJysTp06lXqmchvULl68KDc3N3l6esputyshIUGrVq1ScHCwBgwYoMGDByshIUHt27fXN998ow0bNqhXr16SpIEDB%2Bof//iHmjRpoqCgICUkJCgmJkavvfaaunbtWqqf7%2B/vX6Ri5ckfKEM3%2BBJ2uNDBg/x94ZrMaNSKew4uC15eXurRo4fi4uL0%2BuuvS5Li4uLUq1cvR/j6LbvdrlmzZqlhw4Zq27atdu/erQ8%2B%2BEDPPPOMpCu3Vj3%2B%2BOPq0aOHWrdurcWLF%2BvMmTPq1q1bqWey2e3lM17k5eXpueee06ZNm1SpUiXdfvvtateunRYvXqyEhAStXbtW8fHxOnfunMLDw2W32xUQEKAZM2bo8uXLWrBggT799FNlZGSodu3aGjlypAYMGGBopvJ5Jm8%2BNGrlg82N69OWFxZ2palp1YqgZnUu/KX37LNlf8yZM8v%2BmFdlZmZqzpw52rRpk/Lz89W1a1dNnTrVcd/afffdp969e%2Buxxx6TJC1ZskQLFixQenq6atWqpYcffliDBw92HO%2Bzzz7TP/7xD6Wnp6tx48Z67rnnHG1daZTboHY9R44cUWFhoYKCghzLYmNjdfvtt2vixImm/dyKdyYrJoJa%2BUBQKwcIauUHQa3cKrdvz3E9ycnJGjZsmH7%2B%2BWdJV95wLiEhQffcc4%2BLJwMA4OZSnt5HzYrK7T1q19OtWzclJydr6NCh%2BvXXX1W3bl3NmDFDrVpxczIAACg/KmRQk6QxY8ZozJgxrh4DAICb2s3WgJW1ChvUAACA6xHUjKmQ96gBAABUBDRqAADANDRqxtCoAQAAWBSNGgAAMA2NmjEENQAAYBqCmjFc%2BgQAALAoGjUAAGAaGjVjaNQAAAAsikYNAACYhkbNGIIaAAAwDUHNGC59AgAAWBSNGgAAMA2NmjE0agAAABZFowYAAExDo2YMQQ0AAJiGoGYMlz4BAAAsikYNAACYhkbNGBo1AAAAi6JRAwAApqFRM4agBgAATENQM4ZLnwAAABZFowYAAExDo2YMjRoAAIBF0agBAADT0KgZQ1ADAACmIagZw6VPAAAAi6JRAwAApqFRM4ZGDQAAwKJo1AAAgGlo1IwhqAEAANMQ1Izh0icAAIBF0agBAADT0KgZQ6MGAABgUTRqAADANDRqxhDUAACAaQhqxnDpEwAAwKJo1AAAgGlo1IwhqAEAANMQ1Izh0icAAIBF0agBAADTlLdGLTs7WzNmzNCmTZtUUFCgrl27atq0afLx8Smy7fPPP681a9Y4LcvNzVX79u01f/58SVKPHj2UlpYmN7f/68aWLVumoKCgUs1js9vtdgOPB7/BmSwfbDb%2BrsoDm5vN1SOgJGFh0q5dUqtWUmKiq6fB9bjwl163bmV/zH/9q%2ByPedUzzzyjEydO6LXXXtPly5c1YcIENW7cWNOmTStx32%2B%2B%2BUaTJk3SokWLFBwcrMzMTIWHh2vjxo2qW7fu75qHS58AAMA0hYVl/2WWnJwcrVmzRuPGjZOvr69uvfVWTZ48WStWrFBOTs519z179qwmT56sZ599VsHBwZKkffv2ydfX93eHNIlLnwAAwERWu/SZm5ur9PT0Ytfl5OQoPz9fISEhjmVBQUHKzc3V0aNH9ac//emax42Li1Pz5s3Vp08fx7K9e/fKy8tLQ4YMUVJSkurWravY2Fh16dKl1PMS1AAAQLmSkZGhU6dOOS3z8/OTv79/ifvu2bNHQ4cOLXbd%2BPHjJUne3t6OZV5eXpKkrKysax4zNTVVq1ev1qeffuq03Gaz6c4779QTTzyhOnXq6Msvv1RsbKwWLVqk0NDQEmeVCGoAAMBEZjRqS5cuVXx8vNOysWPHKjY2tsR9IyIidOjQoWLXHThwQK%2B//rpycnIcLx64esmzatWq1zzm8uXLFRYWVqRxGzVqlNP3ffr00eeff67169cT1AAAQMUUHR2tyMhIp2V%2Bfn6Gj9uoUSN5eHgoOTlZLVu2lCQdPnxYHh4eCgwMvOZ%2BGzZs0IgRI4osnz9/vpo2bap27do5luXl5alKlSqlnomgBgAATGNGo%2Bbv71%2Bqy5w3ysvLSz169FBcXJxef/11SVfuPevVq5c8PT2L3efcuXM6fPiw2rRpU2TdiRMn9Omnn2revHm67bbbtGrVKiUmJmr69OmlnomgBgAATGO1FxOUZNq0aZozZ4569%2B6t/Px8de3aVVOnTnWsv%2B%2B%2B%2B9S7d2899thjkqRffvlFklS7du0ix3ryySfl5uamQYMG6eLFi2rcuLHeffddNWzYsNTz8D5qZYgzWT7wPmrlA%2B%2BjVg7wPmrlhwt/6d19d9kf89tvy/6YVkWjBgAATFPeGjWr4Q1vAQAALIpGDQAAmIZGzRiCGgAAMA1BzRgufQIAAFgUjRoAADANjZoxNGoAAAAWRaMGAABMQ6NmDEENAACYhqBmDJc%2BAQAALIpGDQAAmIZGzRgaNQAAAIuiUQMAAKahUTOGoAYAAExDUDOGS58AAAAWRaMGAABMQ6NmDI0aAACARdGoAQAA09CoGUNQAwAApiGoGcOlTwAAAIuiUQMAAKahUTOGRg0AAMCiaNQAAIBpaNSMIagBAADTENSM4dInAACARdGoAQAA09CoGUOjBgAAYFE0agAAwDQ0asYQ1AAAgGkIasZw6RMAAMCiaNQAAIBpaNSMIagBAADTENSM4dInAACARdGoAQAA09CoGUOjBgAAYFE0agAAwDQ0asYQ1AAAgGkIasZw6RMAAMCiaNQAAIBpaNSMoVEDAACwKBo1AABgGho1YwhqAADANAQ1Y7j0CQAAYFEENQAAYJrCwrL/%2BiPk5OQoOjpaK1asuO52e/bs0YABAxQWFqbIyEh9%2BumnTutXrlypbt26KTQ0VPfff78SExNvaA6CGgAAwG8kJSVp8ODB2r1793W3%2B/XXX/XII4%2BoX79%2B2rlzp2bOnKlZs2bphx9%2BkCRt375dM2bM0OzZs7Vz50716dNHY8aMUU5OTqlnIagBAADTlLdGbevWrRo2bJj%2B8pe/qE6dOtfddsOGDfL19dXgwYPl7u6udu3aqXfv3lq8eLEk6dNPP9V9992n1q1by8PDQ8OHD1eNGjW0bt26Us/DiwkAAIBprPZigtzcXKWnpxe7zs/PT02aNNHmzZtVpUoVLViw4LrHSkpKUkhIiNOyxo0ba9myZZKk5ORkPfDAA0XWHzx4sNTzEtQAAEC5kpGRoVOnTjkt8/Pzk7%2B/f4n77tmzR0OHDi123ZtvvqmoqKhSz5GVlSUvLy%2BnZZ6ensrOzi7V%2BtIgqJUhm83VE6AkGRkZWrp0qaKjo0v1DxouZLe7egKUICMjQ0vnzlX0l1/y7wnXZMY/5blzlyo%2BPt5p2dixYxUbG1vivhERETp06FCZzOHl5aWLFy86LcvNzZWPj49jfW5ubpH1NWrUKPXPIKjhpnLq1CnFx8crMjKSJxbAIP49wVWio6MVGRnptMzPz%2B8PnyMkJETffvut07Lk5GQFBwdLkoKDg5WUlFRkfadOnUr9M3gxAQAAKFf8/f3VrFkzpy9X/MdCt27ddPr0aS1cuFD5%2Bfnatm2b1qxZ47gvrX///lqzZo22bdum/Px8LVy4UGfOnFG3bt1K/TMIagAAAKV033336e2335Yk1ahRQ%2B%2B//76%2B/PJLRURE6LnnntNzzz2ntm3bSpLatWunadOm6YUXXtBdd92ltWvXat68efL19S31z%2BPSJwAAQDE2bdpUZNnatWudvr/zzju1ZMmSax6jb9%2B%2B6tu37%2B%2BegUYNNxU/Pz%2BNHTvWJfcyABUN/54A89nsdl5aBQAAYEU0agAAABZFUAMAALAoghoAAIBFEdQAAAAsiqAGAABgUQQ1AAAAiyKoAQAAWBRBDQAAwKIIagAAABZFUAMAALAoPpQdFVZkZKRsNtt1t9m4ceMfNA1QcZw9e1arV6/W8ePHNX78eO3cuVNdunRx9VhAhUSjhgorNjZWY8eOVZcuXVRYWKhhw4Zp6tSpGjVqlCpVqqR7773X1SMC5c7%2B/fv15z//WV9%2B%2BaWWLVumc%2BfOafz48Vq%2BfLmrRwMqJD6UHRVenz599OqrryooKMix7NixY3rkkUe0fv16F04GlD9DhgzR/fffr/vvv19t2rTRzp07lZCQoFmzZmndunWuHg%2BocGjUUOGlpqaqQYMGTstq166tjIwMF00ElF8//fST%2BvbtK0mOWws6duyo9PR0V44FVFgENVR4zZs315w5c5SXlydJysnJ0YwZM9S6dWsXTwaUPzVr1lRKSorTspSUFNWqVctFEwEVGy8mQIU3ffp0Pfroo1qyZIlq1Kihc%2BfOqVGjRnr33XddPRpQ7gwaNEiPPvqoHnvsMRUUFGjdunX6xz/%2BoejoaFePBlRI3KOGm0JBQYF27dqljIwMBQQEqFWrVnJzo1AGfo/Fixfro48%2B0vHjx1W7dm1FR0dr%2BPDh/JsCTEBQQ4V18uRJBQQEKC0t7Zrb1KlT5w%2BcCACAG0NQQ4XVqlUr7dq1S02aNCnyfmp2u102m00//viji6YDyqdnnnmm2OUeHh6qWbOmOnfurNDQ0D92KKAC4x41VFhr166VVLo3tb3avgG4Pg8PD61YsUJRUVGqX7%2B%2B0tLStGHDBrVv317nz5/XP//5T82cOVM9e/Z09ahAhUCjBuj/2jcA1zd69GhFR0crKirKsWzLli36%2BOOP9fbbb2v79u166aWXtGbNGhdOCVQc3PkJ6MqlUAAl27NnjyIjI52WdezYUd99950kKSIiQsePH3fFaECFRFADpBI/ExTAFTVr1lRCQoLTsq1bt8rX11fSlTeYrl69ugsmAyom7lEDAJTa1c/Qvffee1WvXj398ssv%2BuqrrzR9%2BnSlpKRo2LBhGjJkiKvHBCoM7lEDxD1qwI3YvXu3li9frpMnT6pOnToaOHCg7rjjDv3yyy9KTk52un8NgDEENUAENaC0fv75Z7355ptKT09XYWGhJCk/P19HjhzRtm3bXDwdUPFwjxoAoNSeffZZHT9%2BXNWqVdPly5cVEhKipKQkLncCJiGoocL77rvvHP/lfy2VK1f%2Bg6YByrd9%2B/bpzTffVExMjKpWrarnnntOr7zyirZu3erq0YAKiaCGCu/xxx/XpUuXrrsNl2yA0vHy8lL16tXVoEED/fTTT5KkTp06KSUlxcWTARUTQQ0VXv369bV3715XjwFUCA0aNNCWLVvk4%2BOjwsJCpaamKj09XQUFBa4eDaiQeHsOVHjVq1fXww8/rHr16snf39/pPdM%2B%2BOADF04GlD%2BPPPKIxo0bp88//1zR0dF68MEHValSJXXt2tXVowEVEq/6RIUXHx9/zXVjx479AycBKob09HTdeuutcnd317p165SZmal%2B/fpxrydgAoIaAACARXGPGm4Kn3zyiXr37q2IiAilpaVp3LhxysrKcvVYAABcF0ENFd7ChQs1f/58PfTQQ7p8%2BbJ8fHyUnp6uWbNmuXo0AACui6CGCu/jjz/WW2%2B9pYEDB8rNzU3Vq1fX3LlztXnzZlePBgDAdRHUUOGdO3dOjRo1kiRdvSXz1ltv5e0EAACWR1BDhdekSRMtXbpUkhxvzbFu3ToFBwe7ciwAAErEqz5R4e3fv1/Dhw9XUFCQ9u3bp3bt2mn37t1677331LJlS1ePBwDANRHUcFPIyMjQZ599prS0NAUEBKh3796qU6eOq8cCAOC6CGqo8J5%2B%2Bmnde%2B%2B96tChA2/ICQAoV7hHDRVejRo19Le//U1t27bVhAkTtG7dOt5DDQBQLtCo4aaRkpKijRs3avPmzfrxxx8VERGht99%2B29VjAQBwTTRquGlcunRJNptNXl5eKiws1JEjR1w9EgAA10WjhgrviSee0NatW1VYWKi77rpLd999tzp06KB69eq5ejQAAK7L3dUDAGbbvXu3cnJy1KNHD3Xs2FHt27eXr6%2Bvq8cCAKBENGq4KRw5ckTffPONEhIStGvXLgUGBqpDhw6aMGGCq0cDAOCaCGq4qSQnJ2vjxo1auHChsrKy9MMPP7h6JAAAromghgpv48aN%2Bvrrr5WQkKALFy6oY8eOioqK0j333KOqVau6ejwAAK6JoIYK7%2B6771ZkZKSioqLUrl073vQWAFBuENRQ4dntdtlsNv36669KTU1V06ZNVVBQQGADAFge76OGCi8nJ0eTJk1SRESEhgwZoqNHj6pbt25KSUlx9WgAAFwXQQ0V3pw5c5Sdna0vvvhCHh4eql%2B/vrp06aKZM2e6ejQAAK6L91FDhbd582atWbNG1atXl81mk4eHh55%2B%2Bml16tTJ1aMBAHBdNGqo8AoLCx33o129JfO3ywAAsCqCGiq8tm3b6sUXX1ROTo5sNpsk6bXXXtNdd93l4skAALg%2BXvWJCu/MmTMaM2aMDhw4oMuXL8vT01OBgYF6%2B%2B23Vbt2bVePBwDANRHUcFOw2%2B3au3evjh8/roCAALVo0UKVKlVy9VgAAFwXQQ0V1smTJxUQEKC0tLRrblOnTp0/cCIAAG4MQQ0VVqtWrbRr1y41adJENpvN8ca30v%2B9Ce6PP/7o4ikBALg23p4DFdbatWslSffcc48mTJigW265xcUTAQBwYwhqqLBuu%2B02SZK3t7f%2B%2Bte/6vbbb9eAAQPUq1cvVatWzcXTAQBQMi594qZw8eJFrVmzRqtWrdJPP/2ke%2B%2B9VwMGDFCbNm1cPRoAANdEUMNNZ%2BvWrXr22Wd14sQJ7lEDAFgalz5xU8jKytKXX36pVatW6YcfflDnzp01Y8YMV48FAMB10aihwps0aZI2bdqkgIAADRgwQP369VPNmjVdPRYAACWiUUOF5%2B7urnnz5ik8PNzVowAAcENo1AAAACyKD2UHAACwKIIaAACARRHUAAAALIqgBgAAYFEENQAAAIsiqAEAAFgUQQ0AAMCiCGoAAAAW9f8BoDdi4d4QSdMAAAAASUVORK5CYII%3D" class="center-img">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmoAAAIfCAYAAADAPfANAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAABEyElEQVR4nO3de3zO9eP/8ec1xjZkDpvl1DRbcp5DQ4ixfJRThRVCoSIzJUWRJKfPbx3RSaKiklMRRQ5JfYTayNnmOKaNOe5km12/P3x3fbo%2BG6b33l3vzeN%2Bu%2B3WZ6/3Ya/r7WPX0/N9uGx2u90uAAAAWI6bqycAAACA/BHUAAAALIqgBgAAYFEENQAAAIsiqAEAAFgUQQ0AAMCiCGoAAAAWRVADAACwKIIaAACARZV09QQAGLdv3z4tXrxYmzdvVmJioi5duqSKFSsqMDBQ7dq1U8%2BePeXh4eHqaQIAbpCNj5ACirZ33nlH7733nnJyclS2bFnVrFlT7u7uOnXqlBISEiRJt956q2bNmqV69eq5eLYAgBtBUAOKsCVLlujFF1%2BUl5eXpk6dqrCwMJUoUcKx/ODBg3rxxRe1fft2VahQQatWrVLFihVdOGMAwI3gGjWgCHv//fclSc8//7z%2B9a9/OYU0SQoICNB7772nSpUq6ezZs/r0009dMU0AwN9EUAOKqAsXLujYsWOSpEaNGl11vYoVK6pjx46SpD/%2B%2BOMfmRsAoHBwMwFQRJUs%2Bd%2B/vhs2bFDdunWvum5ERIT69%2B%2BvSpUqOcbGjBmjZcuWaezYsWrTpo3efPNNbdu2TZmZmbrtttv0wAMP6OGHH1bp0qXz3ee2bdv02WefKTo6WufOndMtt9yixo0b69FHH1XLli3z3ebChQv68ssvtXHjRsXFxSklJUWenp6qWbOm2rdvr/79%2B6t8%2BfJO29xxxx2SpF9%2B%2BUXTpk3TunXr5Obmpnr16unjjz/WuHHjtGzZMr322mtq3ry5ZsyYoV9//VUXL15U9erV1atXLw0cOFA2m01r1qzRJ598or179yonJ0d16tTR0KFDdc899%2BSZa0ZGhpYsWaK1a9dq//79unDhgkqVKqWqVauqdevWeuyxx1SlShWnbUJDQ3XixAmtWrVKycnJ%2Buijj7Rjxw6lpaWpevXq6ty5swYNGqQyZcpc9c8KAP6Ka9SAIuyRRx5RdHS0bDabunfvrp49e6pJkyZ5ToHmJzeoPfjgg1q9erXS0tIUGBio7OxsHTp0SJLUtGlTffDBBypXrpzTtlFRUZo9e7YkqXz58qpevbqSkpJ06tQpSdLgwYM1evRop22OHDmigQMH6uTJkypZsqRq1qwpT09PnThxQufOnZMk1apVS0uWLHEKMrlBrUmTJoqJiVFQUJDOnDmjkJAQvf76606v47vvvlN2drYCAgKUnJzsmM8TTzwhm82mDz74QLfccotq1Kihw4cPKy0tTTabTR9%2B%2BKHatm3r%2BJlnzpzRgAEDdODAAdlsNtWsWVPlypVTYmKiY5%2BVKlXS0qVL5efn59guN6g99thjmjdvnkqVKiV/f3%2BdP39ef/75pyQpODhYCxYsKNCfEQDIDqDI2r17t71x48b2oKAgx1eTJk3sQ4YMsX/wwQf27du32y9fvpzvti%2B88IJjm/bt29v37NnjWBYdHW1v1aqVPSgoyD5%2B/Hin7b744gt7UFCQvVmzZvZvvvnGMZ6Tk2NfuXKlYz5fffWV03b9%2BvWzBwUF2Xv37m1PTEx02m7ZsmX2OnXq2IOCguzz58932i53jvXr17dv3brVbrfb7ZcvX7afPXs2z%2Bt45JFH7ElJSY51xowZYw8KCrLXqVPHfscdd9jnzJnjOB5nzpyx9%2BjRwx4UFGTv169fvscmLCzMfvjwYadlP/30k71Ro0b2oKAg%2B7Rp05yWtW/f3jGXMWPG2C9cuOB4jfPnz3cs%2B%2BGHH/L9MwGA/8U1akARVrduXS1atEhNmzZ1jKWkpGjjxo16/fXX1bt3b7Vu3Vpvvvmm0tPT892Hm5ub3n33Xd15552OseDgYE2fPl2StGjRIiUmJkqSMjMzNWPGDEnSlClT1K1bN8c2NptN9913n6NJmzFjhrKzsyVJycnJio2NlSRNmjRJvr6%2BTtv16NFDd911lyRp//79%2Bc6zc%2BfOat68uWPO3t7eTstLliypN954Qz4%2BPo51nnjiCUlSTk6Ounfvrscff1xubld%2B7VWoUEH9%2B/eXJO3Zs8exn%2BzsbP3222%2By2WwaO3as/P39nX5OmzZtdN9990mSDhw4kO9c69SpoylTpjiaSJvNpr59%2Bzrawd9//z3f7QDgfxHUgCKudu3a%2Bvzzz/X1119r%2BPDhCg4Olru7u2N5cnKy3n//fXXr1s1x%2Bu2vWrRooTp16uQZb926tapXr66cnBxt2LBBkhQTE6PTp0%2BrTJky6tChQ77z6datm9zc3JSYmOgIQJUqVdKvv/6qHTt2KCgoKM82ly9fVtmyZSVduTYsP38No/m54447nE5DSlK1atUc/zu/69ByA2NKSopjrGTJklq7dq127Nihdu3a5dnGbrfLy8vrmnNt166dbDZbnvHbb79dknTx4sVrvhYAyMXNBEAxceedd%2BrOO%2B9URESE0tPTFR0drZ9//lnffPONkpOTdezYMUVGRmrhwoVO2zVs2PCq%2B7zjjjt0/PhxHTlyRJIcrVhWVpb69u171e1KlCihnJwcHTp0yGn/Hh4eOnnypHbs2KFjx44pPj5eBw8e1N69e5WWlibpSvuVn9ym7GpuvfXWPGOlSpVy/O8KFSrkWf7XGzL%2BV%2BnSpZWcnKzt27fryJEjOn78uA4dOqS9e/fq/Pnz15zrXxvDv8r9dIjLly9f/YUAwF8Q1IBiyNPTU3fffbfuvvtuRUZG6sUXX9TKlSu1fft27d692%2BkTCv73Lsu/ym2OLly4IOm/TVBmZqaio6OvO4/c7STp0KFD%2Bve//62NGzc6BZyyZcuqWbNmSkpK0r59%2B666r%2Bt9BJanp%2Bc1l%2Bee8iyIU6dOafr06fr%2B%2B%2B%2BVlZXl9DMaNGigy5cvX/P05V8DYn7s3MMFoIAIakAR9fLLL%2BvXX3/VAw88oKFDh151PQ8PD7366qtas2aNsrKydPjwYaeglttk5Sf3lGDuYz1yw1C9evW0dOnSAs81OTlZ/fr1U3JysqpWrarevXurbt26uv3221W9enXZbDaNGjXqmkHtn3Lp0iUNGDBABw8elLe3tx555BHVr19fAQEBqlmzpkqUKKE333yT68wA/CMIakARdenSJR09elRr1669ZlCTrrRWZcqU0blz5/J8hFTu6cz85Aan2rVrS7ry%2BAzpyqM2srOz8z11aLfbtWXLFvn5%2Balq1aoqVaqUlixZouTkZHl7e2vJkiX5foxV7g0LrrZ27VodPHhQJUuW1MKFC/PcTCAp32v9AMAM3EwAFFG5d1zu2rXruu3Wzz//rHPnzsnb2zvPpxj89NNPjmeD/dWGDRt08uRJlSpVSqGhoZKk5s2bq1y5ckpNTb3qz1yxYoUGDBigzp07OwLN8ePHJUlVq1bNN6TFxcVp%2B/btklx//VbuXMuUKZNvSDt9%2BrR%2B/PFHSa6fK4Dij6AGFFF33323OnXqJEkaN26cJk%2Be7AgZuS5duqQlS5Zo5MiRkqTIyMg8T8VPS0vTsGHDdPLkScfYli1bNHbsWElXHhab%2B5gJLy8vxyMvJk%2BerCVLljhdb7Z27VpNmDBB0pXHadSsWVPSf%2B923Ldvn1avXu1Y326366efftLgwYMd14Jd7TEi/5TcuZ4/f16ffPKJ0/Vk27dv12OPPeZ4QK%2Br5wqg%2BOPUJ1CERUVFycvLS19//bU%2B/fRTffrpp6pataoqVaqkS5cu6ciRI8rMzJS7u7tGjRqlPn365NmHv7%2B/9u7dq44dOyooKEhpaWmOuzy7dOmiJ5980mn9IUOGKD4%2BXl999ZVefPFF/b//9/9UvXp1JSYmKikpSdKVTxF47bXXHNv07NlTn3/%2BuY4ePaoRI0aoWrVqqlChgk6ePKnk5GS5u7vrrrvu0tatW11%2BCjQ0NFTBwcGKiYnRlClTNHv2bFWpUkWnTp1SYmKibDabWrVqpf/85z9KSkqS3W7P91EcAFAYCGpAEVaqVClNmzZNffv21apVq7RlyxYlJiZq37598vT0VK1atdS6dWv17NnT0RT9rwYNGigqKkrvvPOOfv/9d5UsWVJ33XWXHnnkEceDXf/KZrNp0qRJ6tSpk7788ktt375de/fuVenSpdW4cWN16dJF4eHhTnc%2Bli1bVosXL9bs2bO1YcMGHT9%2BXKdPn5afn5/atWunAQMGyMvLSx07dtS%2BffuUkJCgqlWrmnbcrqVEiRKaN2%2BePvvsM61cuVLx8fE6cOCAfHx8dN9996lv376qV6%2BeQkJCdO7cOUVHR1/3GW8A8HfxWZ/ATSr3MzK7du2qqKgoV08HAJAPrlEDAACwKIIaAACARRHUAAAALIqgBgAAbkpnzpxRWFiYtmzZctV1Nm7cqK5du6px48bq3LmzNmzY4LR89uzZatu2rRo3bqxHH31Uhw4dKtQ5EtSAm9S0adO0f/9%2BbiQAcFP6/fffFR4ermPHjl11nSNHjigiIkKRkZH67bffFBERoZEjRzoeI7Rs2TJ99tlnmjNnjrZs2aJ69eppxIgRhfp5vgQ1AABwU1m2bJmee%2B45PfPMM9ddr1mzZurYsaNKliyp%2B%2B67T82bN9fChQslSV999ZX69OmjwMBAlS5dWqNGjVJCQsI1G7obRVADAABFSlJSknbv3u30lfvA7YJo3bq1fvjhh3yfFflXcXFxCgoKchqrXbu243OQ/3e5u7u7/P39HcsLAw%2B8LUw8ndz6atWSYmOlwEDp8GFXzwbXYM/hEY9AYXHp25MJP3zhO%2B9o5syZTmPDhw9XREREgbb38fEp0Hqpqany9PR0GvPw8FBaWlqBlhcGghpuLt7eUokSV/4LwDCbTeKx6finhYeHKzQ01GmsoOHrRnh6eiojI8NpLCMjw/GZyddbXhgIagAAwDxuhX%2BVla%2Bvr3x9fQt9v/8rKChIu3fvdhqLi4tT/fr1JUmBgYGKjY1V%2B/btJUlZWVk6cuRIntOlRnCNGgAAQD66deumrVu3atWqVcrOztaqVau0detWde/eXZL00EMPaf78%2Bdq3b58uXbqk119/XZUrV1azZs0KbQ40agAAwDwmNGpmCg4O1sSJE9WtWzcFBARo1qxZioqK0ksvvaRq1appxowZqlWrliSpZ8%2Beunjxop5%2B%2BmmdOXNGDRo00AcffCB3d/dCmw8fyl6YuJnA%2BoKDpehoqUkTKSbG1bPBNXAzQdHANWpFg0vfngoxtDhkZRX%2BPi2KRg0AAJiniDVqVkNQAwAA5iGoGcLRAwAAsCgaNQAAYB4aNUMIagAAwDwENUM4egAAABZFowYAAMxDo2YIRw8AAMCiaNQAAIB5aNQMIagBAADzENQM4egBAABYFI0aAAAwD42aIRw9AAAAi6JRAwAA5qFRM4SgBgAAzENQM4SjBwAAYFE0agAAwDw0aoZw9AAAACyKRg0AAJiHRs0QghoAADAPQc0Qjh4AAIBF0agBAADz0KgZwtEDAACwKBo1AABgHho1QwhqAADAPAQ1Qzh6AAAAFkWjBgAAzEOjZghHDwAAwKJo1AAAgHlo1AwhqAEAAPMQ1Azh6AEAAFgUjRoAADAPjZohHD0AAACLolEDAADmoVEzhKAGAADMQ1AzhKMHAABgUTRqAADAPDRqhnD0AAAALIpGDQAAmIdGzRCCGgAAMA9BzRCOHgAAgEXRqAEAAPPQqBnC0QMAALAoGjUAAGAeCzZqycnJGj9%2BvLZu3aoSJUqoW7dueuGFF1SypHMsGjx4sH7//XensbS0NIWHh%2BvVV19VTk6OmjZtKrvdLpvN5ljnl19%2BkZeXV6HMlaAGAADMY8GgNnLkSFWpUkWbNm3S6dOnNXToUM2bN0%2BDBw92Wu%2Bjjz5y%2Bn7x4sWaOXOmhg8fLkmKi4tTVlaWoqOjVapUKVPmar2jBwAAYJKjR49q69atGj16tDw9PVWjRg0NGzZMCxYsuOZ2hw4d0qRJkxQVFSVfX19J0s6dO3XHHXeYFtIkGjUAAGAmExq1pKQknTp1ymnMx8fHEaCuJTY2Vt7e3qpSpYpjLCAgQAkJCbpw4YJuueWWfLebOHGievTooWbNmjnGdu7cqUuXLumhhx7SiRMnFBAQoFGjRqlJkyZ/85XlRVADAADmMSGoLVy4UDNnznQaGz58uCIiIq67bWpqqjw9PZ3Gcr9PS0vLN6j99ttv2rFjh6KiopzGPTw81LBhQ0VGRqp8%2BfJasGCBBg0apOXLl6tGjRo3%2BrLyRVADAABFSnh4uEJDQ53GfHx8CrStl5eX0tPTncZyvy9Tpky%2B2yxcuFCdO3fO8zPGjBnj9P2gQYO0dOlSbdy4Uf369SvQfK6HoAYAAMxjQqPm6%2BtboNOc%2BQkMDNS5c%2Bd0%2BvRpVa5cWZJ08OBB%2Bfn5qVy5cnnWz87O1rp16zRr1qw8y95880116tRJdevWdYxlZmaqdOnSf2tu%2BeFmAgAAcNPw9/dX06ZNNWXKFKWkpCg%2BPl7vvvuuevbsme/6%2B/fv16VLl/K97uzAgQOaPHmyTp06pczMTM2cOVMpKSkKCwsrtPkS1AAAgHnc3Ar/y6B33nlH2dnZ6tChg3r37q02bdpo2LBhkqTg4GAtX77csW58fLzKly%2Bfb0s2depU1axZU927d1dISIi2bt2quXPnytvb2/Acc9nsdru90PZ2s/vLw%2B5gUcHBUnS01KSJFBPj6tngGuw5/GoqCmw2iXcR63Pp21OnToW/z9WrC3%2BfFkWjBgAAYFHcTAAAAMxjwU8mKEo4egAAABZFowYAAMxDo2YIQQ0AAJiHoGYIRw8AAMCiaNQAAIB5aNQM4egBAABYFI0aAAAwD42aIQQ1AABgHoKaIRw9AAAAi6JRAwAA5qFRM4SjBwAAYFE0agAAwDw0aoYQ1AAAgHkIaoZw9AAAACyKRg0AAJiHRs0Qjh4AAIBF0agBAADz0KgZQlADAADmIagZwtEDAACwKBo1AABgHho1Qzh6AAAAFkWjBgAAzEOjZghBDQAAmIegZghHDwAAwKJo1AAAgHlo1Azh6AEAAFgUjRoAADAPjZohBDUAAGAegpohHD0AAACLolEDAADmoVEzhKMHAABgUTRqAADAPDRqhhDUAACAeQhqhnD0AAAALIpGDQAAmIdGzRCCGgAAMA9BzRCOHgAAgEXRqAEAAPPQqBnC0QMAALAoGjUAAGAeGjVDCGoAAMA8BDVDOHoAAOCmkpycrGHDhqlZs2YKCQnR5MmTlZ2dne%2B6gwcPVoMGDRQcHOz4%2BumnnxzLZ8%2BerbZt26px48Z69NFHdejQoUKdq%2BGglpCQoODgYCUkJBTGfExxvTkuXbpUoaGh//CsAAC4Cbi5Ff6XQSNHjpSXl5c2bdqkxYsXa/PmzZo3b16%2B6%2B7atUtz5sxRTEyM46tt27aSpGXLlumzzz7TnDlztGXLFtWrV08jRoyQ3W43PMdchl9t1apVFRMTo6pVqxbGfExRFOYIAADMd/ToUW3dulWjR4%2BWp6enatSooWHDhmnBggV51o2Pj9f58%2BdVt27dfPf11VdfqU%2BfPgoMDFTp0qU1atQoJSQkaMuWLYU23wJfo/b888/r8uXLev311x1jI0eOVFpamjZu3Kh169apevXqOn36tKZNm6bNmzfLZrMpNDRUzz//vLy8vNS6dWu9%2Buqr6tixoyQpNDRUDRs21FtvvSVJmj59upKTk/Xvf/9bu3fv1rRp07Rv3z5VqFBBffr00YABA2Sz2TRjxgzFxMTo/Pnzio%2BP16xZs9S8efOrzv348ePq0KGDY44HDx7UK6%2B8ol27dql69eoKCQm54QOXlJSkU6dOOY351KolX2/vG94X/kF16jj/FwBgLhOuUcv3PdjHR76%2BvtfdNjY2Vt7e3qpSpYpjLCAgQAkJCbpw4YJuueUWx/jOnTtVpkwZPfPMM9q5c6cqV66sgQMHqmfPnpKkuLg4DRkyxLG%2Bu7u7/P39tW/fPrVo0cLoy5R0A0Gtd%2B/eGjRokFJSUlS2bFlduHBB69ev1xdffKGNGzdKknJycjRs2DD5%2B/tr9erVysrK0tixY/Xyyy/rjTfeUGhoqH766Sd17NhRhw4dUnJysjZv3iy73S6bzab169dr9OjRSkxM1IABA/TMM8/o448/1tGjRzVs2DB5eHjo4YcfliRt3rxZH3/8sRo2bKjSpUsX%2BAVnZWXpySefVNu2bfXRRx/p2LFjGjJkiNxu8P9ICxcu1MyZM53GhkdGKiIy8ob2Axf5/HNXzwDXYXP1BFBgNv6wcC0mBLV834OHD1dERMR1t01NTZWnp6fTWO73aWlpTkEtMzNTjRs31jPPPKPAwEBt2bJFERERKlOmjDp37pzvvjw8PJSWlvZ3X1oeBQ5qzZo106233qrvvvtOvXr10rfffqvbb79d5cuXd6yza9cu7d69W3PnzlWZMmUkSS%2B88IL%2B9a9/afz48erYsaNeeeUVSdLPP/%2Bs%2B%2B67T2vXrtWePXvk4eGhpKQktW7dWp999pkCAgLUt29fSVLt2rU1aNAgzZ8/3xHUatSooZYtW97wC46JidHJkyf1/PPPq3Tp0goMDNRjjz2mTz755Ib2Ex4enue6Np%2BuXaUb3A/%2BYXXqXAlpffpI%2B/a5eja4Bvvv0a6eAgrAZpMK8XIcmKS4hel834N9fAq0rZeXl9LT053Gcr/PzS65evTooR49eji%2Bb926tXr06KHvvvtOnTt3lqenpzIyMpy2ycjIyLMfI27o8Ry9evXSN998o169emnZsmXq1auX0/Ljx4/r8uXLuueee5zGS5Uqpfj4eLVq1UoXLlxQbGysNm3apB49eujChQv6z3/%2BI7vdrjZt2sjDw0MnTpzQ7t271axZM8c%2BcnJyVKJECcf3Bak385OYmKgKFSrIw8PDMVazZs0b3o%2Bvr2/eORw%2B/LfmBBfYt0%2BKiXH1LACg%2BDOhUcv3PbiAAgMDde7cOZ0%2BfVqVK1eWJB08eFB%2Bfn4qV66c07qLFy92tGe5MjMzHWfyAgMDFRsbq/bt20u6ctbuyJEjCgoK%2Bltzy88NBbUHHnhAb731lv7zn/9o//796tKliy5evOhY7ufnJw8PD23ZssURqjIzMxUfH6/bbrtNJUuWVJs2bbRu3Tr9/vvvmj59ui5cuKAffvhB6enpjgbNz89PISEhmjNnjmPfZ8%2BeVWpqquN729/858Gtt96qM2fOKDU11ZF4//zzz7%2B1LwAAULT4%2B/uradOmmjJlil599VWdPXtW7777ruO6s79KSUnRG2%2B8odtuu0116tTRTz/9pG%2B//daRTx566CHNmDFDbdu2Va1atfTmm2%2BqcuXKTkWTUTcUcytWrKj27dtr3Lhxuvfee51Oe0pSw4YNddttt2natGlKTU1VRkaGpkyZooEDB%2Bry5cuSpLCwMM2bN0/%2B/v6qWLGiWrdurd9%2B%2B0179uxRu3btJEldu3bV9u3btXz5cmVnZyspKUlPPfWUpk2bZvgFBwcHq1atWnrttdeUnp6uo0eP6uOPPza8XwAAkA8LPp7jnXfeUXZ2tjp06KDevXurTZs2GjZsmKQrOWH58uWSpAEDBqhfv34aPny4goODFRUVpenTpzuCWM%2BePTVw4EA9/fTTatGihfbs2aMPPvhA7u7uhueY64Y/maB3795avXq1pkyZkndnJUvqgw8%2B0PTp03Xvvffq0qVLatiwoebOneuoCdu1a6cxY8aodevWkq5ca%2Bbn5yd/f3%2BVLVtWklStWjV99NFHioqK0muvvaYSJUqoXbt2eumll4y8VklSiRIl9OGHH%2Brll19Wq1atVLlyZXXo0EFr1qwxvG8AAPA/LPjJBJUrV9Y777yT77KYv1wWY7PZNGzYMEeI%2B182m02PP/64Hn/8cVPmKUk2e2E%2Ble1mV9yu1iyOgoOl6GipSROuUbM4ew6/mooCbiYoGlz69jR1auHvc%2BzYwt%2BnRfFZnwAAwDwWbNSKkmIR1EJCQpSZmXnV5StXruRTCQAAQJFTLIJaYX5UAwAAKEQ0aoYUi6AGAAAsiqBmCEcPAADAomjUAACAeWjUDOHoAQAAWBSNGgAAMA%2BNmiEENQAAYB6CmiEcPQAAAIuiUQMAAOahUTOEowcAAGBRNGoAAMA8NGqGENQAAIB5CGqGcPQAAAAsikYNAACYh0bNEI4eAACARdGoAQAA89CoGUJQAwAA5iGoGcLRAwAAsCgaNQAAYB4aNUM4egAAABZFowYAAMxDo2YIQQ0AAJiHoGYIRw8AAMCiaNQAAIB5aNQM4egBAABYFI0aAAAwD42aIQQ1AABgHoKaIRw9AAAAi6JRAwAA5qFRM4SgBgAAzENQM4SjBwAAYFE0agAAwDw0aoZw9AAAACyKRg0AAJiHRs0QghoAADAPQc0Qjh4AAIBF0agBAADz0KgZwtEDAACwKBo1AABgHho1QwhqAADAPAQ1QwhqAADgppKcnKzx48dr69atKlGihLp166YXXnhBJUvmjUVffPGF5s2bp6SkJPn6%2Bqp///7q27evJCknJ0dNmzaV3W6XzWZzbPPLL7/Iy8urUOZKUAMAAOaxYKM2cuRIValSRZs2bdLp06c1dOhQzZs3T4MHD3Zab%2B3atXrjjTc0e/ZsNWrUSNu3b9cTTzyhypUrq1OnToqLi1NWVpaio6NVqlQpU%2BZqvaMHAABgkqNHj2rr1q0aPXq0PD09VaNGDQ0bNkwLFizIs25iYqKGDBmixo0by2azKTg4WCEhIdq2bZskaefOnbrjjjtMC2kSjRoAADCTCY1aUlKSTp065TTm4%2BMjX1/f624bGxsrb29vValSxTEWEBCghIQEXbhwQbfccotjPPcUZ67k5GRt27ZNY8eOlXQlqF26dEkPPfSQTpw4oYCAAI0aNUpNmjQx8vKcENQAAIB5TAhqCxcu1MyZM53Ghg8froiIiOtum5qaKk9PT6ex3O/T0tKcgtpfnTp1Sk8%2B%2BaTq16%2BvLl26SJI8PDzUsGFDRUZGqnz58lqwYIEGDRqk5cuXq0aNGn/npeVBUAMAAEVKeHi4QkNDncZ8fHwKtK2Xl5fS09OdxnK/L1OmTL7bbN%2B%2BXZGRkWrWrJmmTp3quOlgzJgxTusNGjRIS5cu1caNG9WvX78Czed6CGoAAMA8JjRqvr6%2BBTrNmZ/AwECdO3dOp0%2BfVuXKlSVJBw8elJ%2Bfn8qVK5dn/cWLF%2Bu1117TiBEj9Pjjjzste/PNN9WpUyfVrVvXMZaZmanSpUv/rbnlh5sJAADATcPf319NmzbVlClTlJKSovj4eL377rvq2bNnnnVXr16tV155RTNmzMgT0iTpwIEDmjx5sk6dOqXMzEzNnDlTKSkpCgsLK7T5EtQAAIB53NwK/8ugd955R9nZ2erQoYN69%2B6tNm3aaNiwYZKk4OBgLV%2B%2BXJI0c%2BZMXb58WSNGjFBwcLDj6%2BWXX5YkTZ06VTVr1lT37t0VEhKirVu3au7cufL29jY8x1w2u91uL7S93ez%2B8rA7WFRwsBQdLTVpIsXEuHo2uAZ7Dr%2BaigKbTeJdxPpc%2Bva0eXPh77Nly8Lfp0XRqAEAAFgUNxMAAADzWPCTCYoSjh4AAIBF0agBAADz0KgZQlADAADmIagZwtEDAACwKBo1AABgHho1Qzh6AAAAFkWjBgAAzEOjZghBDQAAmIegZghHDwAAwKJo1AAAgHlo1Azh6AEAAFgUjRoAADAPjZohBDUAAGAegpohHD0AAACLolEDAADmoVEzhKMHAABgUTRqAADAPDRqhhDUAACAeQhqhnD0AAAALIpGDQAAmIdGzRCCGgAAMA9BzRCOHgAAgEXRqAEAAPPQqBnC0QMAALAoGjUAAGAeGjVDCGoAAMA8BDVDOHoAAAAWRaMGAADMQ6NmCEcPAADAomjUAACAeWjUDCGoAQAA8xDUDOHoAQAAWBSNGgAAMA%2BNmiEcPQAAAIuiUQMAAOahUTOEoAYAAMxDUDOEowcAAGBRNGoAAMA8NGqGcPQAAAAsikYNAACYh0bNEIIaAAAwD0HNEI4eAAC4qSQnJ2vYsGFq1qyZQkJCNHnyZGVnZ%2Be77saNG9W1a1c1btxYnTt31oYNG5yWz549W23btlXjxo316KOP6tChQ4U6V4IaAAAwj5tb4X8ZNHLkSHl5eWnTpk1avHixNm/erHnz5uVZ78iRI4qIiFBkZKR%2B%2B%2B03RUREaOTIkUpMTJQkLVu2TJ999pnmzJmjLVu2qF69ehoxYoTsdrvhOeYiqAEAgJvG0aNHtXXrVo0ePVqenp6qUaOGhg0bpgULFuRZd9myZWrWrJk6duyokiVL6r777lPz5s21cOFCSdJXX32lPn36KDAwUKVLl9aoUaOUkJCgLVu2FNp8uUYNAACYx4Rr1JKSknTq1CmnMR8fH/n6%2Bl5329jYWHl7e6tKlSqOsYCAACUkJOjChQu65ZZbHONxcXEKCgpy2r527drat2%2BfY/mQIUMcy9zd3eXv7699%2B/apRYsWf%2Bu1/S%2BCWiGy5xRe1Qnz2CTZf4929TRwHTY3m6ungOsJDpaio2Vr2kSKiXH1bHAthXgq7oZ/tAr/7/LChQs1c%2BZMp7Hhw4crIiLiutumpqbK09PTaSz3%2B7S0NKeglt%2B6Hh4eSktLK9DywkBQAwAARUp4eLhCQ0Odxnx8fAq0rZeXl9LT053Gcr8vU6aM07inp6cyMjKcxjIyMhzrXW95YSCoAQAA0%2BTkFP4%2BfX19C3SaMz%2BBgYE6d%2B6cTp8%2BrcqVK0uSDh48KD8/P5UrV85p3aCgIO3evdtpLC4uTvXr13fsKzY2Vu3bt5ckZWVl6ciRI3lOlxrBzQQAAOCm4e/vr6ZNm2rKlClKSUlRfHy83n33XfXs2TPPut26ddPWrVu1atUqZWdna9WqVdq6dau6d%2B8uSXrooYc0f/587du3T5cuXdLrr7%2BuypUrq1mzZoU2X5u9MO8hvclxJIsGm40/q6KAa9SKgP%2B7Rk1NuEbN8lz4Sy8rq/D36e5ubPvTp0/r1Vdf1ZYtW%2BTm5qYePXroueeeU4kSJRQcHKyJEyeqW7dukqRNmzYpKipKx44dU7Vq1TR69Gjdc889kiS73a65c%2BdqwYIFOnPmjBo0aKCJEyeqVq1aRl%2BiA0GtEHEkiwaCWtFAUCsCCGpFhwt/6V26VPj7LF268PdpVZz6BAAAsChuJgAAAKYx42aCmwmNGgAAgEXRqAEAANPQqBlDUAMAAKYhqBnDqU8AAACLolEDAACmoVEzhkYNAADAomjUAACAaWjUjCGoAQAA0xDUjOHUJwAAgEXRqAEAANPQqBlDUAMAAKYhqBnDqU8AAACLolEDAACmoVEzhkYNAADAomjUAACAaWjUjCGoAQAA0xDUjOHUJwAAgEXRqAEAANPQqBlDowYAAGBRNGoAAMA0NGrGENQAAIBpCGrGcOoTAADAomjUAACAaWjUjKFRAwAAsCgaNQAAYBoaNWMIagAAwDQENWM49QkAAGBRNGoAAMA0NGrG0KgBAABYFI0aAAAwDY2aMQQ1AABgGoKaMZz6BAAAsCgaNQAAYBoaNWNo1AAAACyKRg0AAJiGRs0YghoAADANQc0YTn0CAABYFI0aAAAwDY2aMTRqAAAAFkWjBgAATEOjZgxBDQAAmIagZgxBDQAA4P%2BkpaVp0qRJWr9%2BvbKzs9WhQwdNmDBBZcqUyXf91atX691331V8fLy8vb314IMPatiwYXJzu3J1WefOnZWQkOD4XpIWL16sgICAAs2HoAYAAExT1Bq1SZMm6eTJk1q9erUuX76skSNHKioqShMmTMiz7q5du/T888/rrbfe0j333KPDhw9ryJAh8vLy0uOPP66UlBQdPnxY69atU7Vq1f7WfLiZAAAAQFJ6erpWrFihESNGyNvbW5UqVdJzzz2npUuXKj09Pc/6J06c0MMPP6z27dvLzc1NAQEBCgsL07Zt2yRdCXLe3t5/O6RJNGoAAMBEZjRqSUlJOnXqlNOYj4%2BPfH19r7ttRkaGEhMT812Wnp6urKwsBQUFOcYCAgKUkZGhI0eO6M4773Rav1OnTurUqZPTvn/88Ud17dpVkrRz5055enqqX79%2Bio2NVbVq1RQREaH27dsX%2BLUS1AAAgGnMCGoLFy7UzJkzncaGDx%2BuiIiI6267Y8cO9e/fP99lkZGRkiQvLy/HmKenpyQpNTX1mvtNSUlRZGSkPDw8NHDgQEmSzWZTgwYN9Oyzz6pq1ar6/vvvFRERofnz56tx48bXnatEUAMAAEVMeHi4QkNDncZ8fHwKtG1ISIj279%2Bf77I9e/bo7bffVnp6uuPmgdxTnmXLlr3qPg8dOqQRI0aoUqVK%2BvTTTx3rDh482Gm9bt266dtvv9Xq1asJagAAwPXMaNR8fX0LdJrzRtWqVUvu7u6Ki4tTo0aNJEkHDx6Uu7u7/P39891m48aNevbZZ9W7d2%2BNGjVKJUv%2BN1rNmTNHdevWVcuWLR1jmZmZKl26dIHnxM0EAAAAunKas3PnzoqKitKZM2d05swZRUVFqUuXLvLw8Miz/vbt2/X0009r7NixeuGFF5xCmiSdPHlSEydOVHx8vLKzs7V48WLFxMTogQceKPCcbHa73W74lUGSxJEsGmw2/qyKApubzdVTwPUEB0vR0VKTJlJMjKtng2tx4S%2B9H34o/H2GhRX%2BPnOlpKRo%2BvTpWr9%2BvbKystShQweNHz/ecd3a/fffr65du%2Bqpp57SU089pR9//NFxHVuupk2b6qOPPlJmZqaioqL03Xff6eLFi6pdu7ZGjx6tkJCQAs%2BHoFaIOJJFA0GtaCCoFQEEtaLDhb/0Vq8u/H3%2B5UbLYo9TnwAAABbFzQQAAMA0Re2TCayGoAYAAExDUDOGU58AAAAWRaMGAABMQ6NmDI0aAACARdGoAQAA09CoGUNQAwAApiGoGcOpTwAAAIuiUQMAAKahUTOGRg0AAMCiaNQAAIBpaNSMIagBAADTENSM4dQnAACARdGoAQAA09CoGUOjBgAAYFE0agAAwDQ0asYQ1AAAgGkIasZw6hMAAMCiaNQAAIBpaNSMoVEDAACwKBo1AABgGho1YwhqAADANAQ1Yzj1CQAAYFE0agAAwDQ0asbQqAEAAFgUjRoAADANjZoxBDUAAGAagpoxnPoEAACwKBo1AABgGho1Y2jUAAAALIpGDQAAmIZGzRiCGgAAMA1BzRhOfQIAAFgUjRoAADANjZoxNGoAAAAWRaMGAABMQ6NmDEENAACYhqBmDKc%2BAQAALIpGDQAAmIZGzRgaNQAAAIuiUQMAAKahUTOGRg0AAJgmJ6fwv8yUlpamsWPHKiQkRE2bNtXzzz%2Bv1NTUq64/YcIE1a9fX8HBwY6vhQsXOpYvW7ZMYWFhaty4sR588EHFxMTc0HwIagAAAP9n0qRJOnnypFavXq01a9bo5MmTioqKuur6O3fu1KRJkxQTE%2BP4Cg8PlyRt2bJFkyZN0rRp07Rt2zZ169ZNQ4cOVXp6eoHnU6SD2vr16/Xwww%2BrZcuWatSokfr166cjR45IklauXKlOnTqpWbNmGjRokMaPH68xY8ZIkux2uz799FPH8j59%2BmjXrl0ufCUAABRPRalRS09P14oVKzRixAh5e3urUqVKeu6557R06dJ8w1VmZqYOHDig%2BvXr57u/RYsW6f7771fTpk3l7u6ugQMHqkKFClq1alWB51Rkg9qff/6pyMhIPfHEE9q8ebN%2B/PFH2e12zZo1SzExMXrhhRf0wgsv6Ndff9XDDz%2BspUuXOrb9/PPPNXfuXL399tvavHmzHnzwQT322GM6ffq0C18RAAAwW0ZGho4ePXrVr6ysLAUFBTnWDwgIUEZGhqMI%2Bqt9%2B/YpOztb77zzjlq1aqVOnTrpww8/VM7/pcm4uDinfUlS7dq1tW/fvgLPt8jeTFCxYkWtXLlSNWvWVEpKiv78809VqFBBiYmJWrJkie69916FhoZKksLCwtSxY0fHtgsWLNCTTz6pOnXqSJJ69uypxYsXa/ny5Xr88ccL9POTkpJ06tQpp7HKlX3k6%2BtbSK8QuMkFB7t6Brie//sd6vgvkA8zGrD83oN9fAr2Hrxjxw71798/32WRkZGSJC8vL8eYp6enJOV7ndrFixd111136dFHH9Ubb7yhvXv36umnn5abm5sGDx6s1NRUx/a5PDw8lJaWdt155iqyQc3d3V3ffvutvvzyS9lsNgUFBSklJUUlS5bUyZMnVbduXaf1a9So4WjMTpw4oenTpzudc87Ozr5qdZmfhQsXaubMmU5jTz89XCNGRBh4Vfin2GyungGuKzra1TNAQX3%2BuatnAAszI6jl9x48fPhwRURc/z04JCRE%2B/fvz3fZnj179Pbbbys9PV1lypSRJMcpz7Jly%2BZZ/%2B6779bdd9/t%2BL5hw4YaMGCAVq1apcGDB8vT01MZGRlO22RkZKhChQrXnWeuIhvUvvvuO82fP19ffPGFbrvtNklXLgA8cOCAqlWrpoSEBKf1ExISVKpUKUmSn5%2BfRowYofvvv9%2Bx/NixY/L29i7wzw8PD3c0drkqV/aR3f43XxD%2BMTab%2BHMqAmxNm7h6CrieOnWuhLQ%2BfaQbOJUDFyhm//DJ7z3Yx8fH8H5r1aold3d3xcXFqVGjRpKkgwcPyt3dXf7%2B/nnWX7t2rU6fPq2HH37YMZaZmSkPDw9JUmBgoGJjY522iYuLU9u2bQs8pyIb1C5evCg3Nzd5eHjIbrdr06ZN%2BvrrrxUYGKhevXqpb9%2B%2B2rRpk1q1aqWff/5Za9asUZcuXSRJvXv31nvvvac6deooICBAmzZt0rBhw/TWW2%2BpQ4cOBfr5vr6%2BeSpW3vyBQnSDt7DDhfbt488LV2VGo5bfe3Bh8PT0VOfOnRUVFaW3335bkhQVFaUuXbo4wtdf2e12TZ06VbfddptatGih7du369NPP9XYsWMlXbm06umnn1bnzp3VtGlTLViwQMnJyQoLCyvwnGx2e9GMF5mZmRo3bpzWr1%2BvEiVK6Pbbb1fLli21YMECbdq0SStXrtTMmTN19uxZNWvWTHa7XX5%2Bfpo0aZIuX76suXPnatGiRUpKSlKVKlU0aNAg9erVy9CciuaRvPnQqBUNNjfOT1tecPCVpqZJE4Ka1bnwl95LLxX%2BPidPLvx95kpJSdH06dO1fv16ZWVlqUOHDho/frzjurX7779fXbt21VNPPSVJ%2BvLLLzV37lwlJiaqcuXKeuyxx9S3b1/H/r755hu99957SkxMVO3atTVu3DhHW1cQRTaoXcvhw4eVk5OjgIAAx1hERIRuv/12PfPMM6b93OJ3JIsnglrRQFArAghqRQdBrcgqso/nuJa4uDgNGDBAx44dk3TlgXObNm3SPffc4%2BKZAQBwcylKz1GzoiJ7jdq1hIWFKS4uTv3799f58%2BdVrVo1TZo0SU2acHEyAAAoOoplUJOkoUOHaujQoa6eBgAAN7WbrQErbMU2qAEAANcjqBlTLK9RAwAAKA5o1AAAgGlo1IyhUQMAALAoGjUAAGAaGjVjCGoAAMA0BDVjOPUJAABgUTRqAADANDRqxtCoAQAAWBSNGgAAMA2NmjEENQAAYBqCmjGc%2BgQAALAoGjUAAGAaGjVjaNQAAAAsikYNAACYhkbNGIIaAAAwDUHNGE59AgAAWBSNGgAAMA2NmjE0agAAABZFowYAAExDo2YMQQ0AAJiGoGYMpz4BAAAsikYNAACYhkbNGBo1AAAAi6JRAwAApqFRM4agBgAATENQM4ZTnwAAABZFowYAAExDo2YMjRoAAIBF0agBAADT0KgZQ1ADAACmIagZw6lPAAAAi6JRAwAApqFRM4ZGDQAAwKJo1AAAgGlo1IwhqAEAANMQ1Izh1CcAAIBF0agBAADT0KgZQ1ADAACmIagZw6lPAAAAi6JRAwAApilqjVpaWpomTZqk9evXKzs7Wx06dNCECRNUpkyZPOu%2B/PLLWrFihdNYRkaGWrVqpTlz5kiSOnfurISEBLm5/bcbW7x4sQICAgo0H5vdbrcbeD34C45k0WCz8WdVFNjcbK6eAq4nOFiKjpaaNJFiYlw9G1yLC3/phYUV/j5/%2BKHw95lr7NixOnnypN566y1dvnxZI0eOVO3atTVhwoTrbvvzzz9r1KhRmj9/vgIDA5WSkqJmzZpp3bp1qlat2t%2BaD6c%2BAQCAaXJyCv/LLOnp6VqxYoVGjBghb29vVapUSc8995yWLl2q9PT0a2575swZPffcc3rppZcUGBgoSdq1a5e8vb3/dkiTOPUJAABMZLVTnxkZGUpMTMx3WXp6urKyshQUFOQYCwgIUEZGho4cOaI777zzqvuNiopS/fr11a1bN8fYzp075enpqX79%2Bik2NlbVqlVTRESE2rdvX%2BD5EtQAAECRkpSUpFOnTjmN%2Bfj4yNfX97rb7tixQ/379893WWRkpCTJy8vLMebp6SlJSk1Nveo%2B4%2BPjtXz5ci1atMhp3GazqUGDBnr22WdVtWpVff/994qIiND8%2BfPVuHHj685VIqgBAAATmdGoLVy4UDNnznQaGz58uCIiIq67bUhIiPbv35/vsj179ujtt99Wenq64%2BaB3FOeZcuWveo%2BlyxZouDg4DyN2%2BDBg52%2B79atm7799lutXr2aoAYAAIqn8PBwhYaGOo35%2BPgY3m%2BtWrXk7u6uuLg4NWrUSJJ08OBBubu7y9/f/6rbrVmzRo8//nie8Tlz5qhu3bpq2bKlYywzM1OlS5cu8JwIagAAwDRmNGq%2Bvr4FOs15ozw9PdW5c2dFRUXp7bfflnTl2rMuXbrIw8Mj323Onj2rgwcPqnnz5nmWnTx5UosWLdLs2bN166236uuvv1ZMTIwmTpxY4DkR1AAAgGmsdjPB9UyYMEHTp09X165dlZWVpQ4dOmj8%2BPGO5ffff7%2B6du2qp556SpJ0/PhxSVKVKlXy7Ov555%2BXm5ub%2BvTpo4sXL6p27dr68MMPddtttxV4PjxHrRBxJIsGnqNWNPActSKA56gVHS78pXf33YW/z19%2BKfx9WhWNGgAAME1Ra9SshgfeAgAAWBSNGgAAMA2NmjEENQAAYBqCmjGc%2BgQAALAoGjUAAGAaGjVjaNQAAAAsikYNAACYhkbNGIIaAAAwDUHNGE59AgAAWBSNGgAAMA2NmjE0agAAABZFowYAAExDo2YMQQ0AAJiGoGYMpz4BAAAsikYNAACYhkbNGBo1AAAAi6JRAwAApqFRM4agBgAATENQM4ZTnwAAABZFowYAAExDo2YMjRoAAIBF0agBAADT0KgZQ1ADAACmIagZw6lPAAAAi6JRAwAApqFRM4ZGDQAAwKJo1AAAgGlo1IwhqAEAANMQ1Izh1CcAAIBF0agBAADT0KgZQ1ADAACmIagZw6lPAAAAi6JRAwAApqFRM4ZGDQAAwKJo1AAAgGlo1IwhqAEAANMQ1Izh1CcAAIBF0agBAADT0KgZQ6MGAABgUTRqAADANDRqxhDUAACAaQhqxnDqEwAAwKIIagAAwDQ5OYX/9U9IT09XeHi4li5des31duzYoV69eik4OFihoaFatGiR0/Jly5YpLCxMjRs31oMPPqiYmJgbmgdBDQAA4C9iY2PVt29fbd%2B%2B/ZrrnT9/Xk888YR69Oihbdu2afLkyZo6dar%2B%2BOMPSdKWLVs0adIkTZs2Tdu2bVO3bt00dOhQpaenF3guBDUAAGCaotaobd68WQMGDNADDzygqlWrXnPdNWvWyNvbW3379lXJkiXVsmVLde3aVQsWLJAkLVq0SPfff7%2BaNm0qd3d3DRw4UBUqVNCqVasKPB9uJgAAAKax2s0EGRkZSkxMzHeZj4%2BP6tSpow0bNqh06dKaO3fuNfcVGxuroKAgp7HatWtr8eLFkqS4uDg99NBDeZbv27evwPMlqAEAgCIlKSlJp06dchrz8fGRr6/vdbfdsWOH%2Bvfvn%2B%2ByWbNmqWPHjgWeR2pqqjw9PZ3GPDw8lJaWVqDlBUFQK0Q2m6tngOtJSkrSwoULFR4eXqC/0HAhu93VM8B1JCUlaeGMGQr//nv%2BPuGqzPirPGPGQs2cOdNpbPjw4YqIiLjutiEhIdq/f3%2BhzMPT01MXL150GsvIyFCZMmUcyzMyMvIsr1ChQoF/BkENN5VTp05p5syZCg0N5Y0FMIi/T3CV8PBwhYaGOo35%2BPj84/MICgrSL7/84jQWFxenwMBASVJgYKBiY2PzLG/btm2BfwY3EwAAgCLF19dX9erVc/pyxT8WwsLCdPr0ac2bN09ZWVn69ddftWLFCsd1aT179tSKFSv066%2B/KisrS/PmzVNycrLCwsIK/DMIagAAAAV0//336/3335ckVahQQR9//LG%2B//57hYSEaNy4cRo3bpxatGghSWrZsqUmTJigV155RXfddZdWrlyp2bNny9vbu8A/j1OfAAAA%2BVi/fn2esZUrVzp936BBA3355ZdX3Uf37t3VvXv3vz0HGjXcVHx8fDR8%2BHCXXMsAFDf8fQLMZ7PbubUKAADAimjUAAAALIqgBgAAYFEENQAAAIsiqAEAAFgUQQ0AAMCiCGoAAAAWRVADAACwKIIaAACARRHUAAAALIqgBgAAYFF8KDuKrdDQUNlstmuus27dun9oNkDxcebMGS1fvlwnTpxQZGSktm3bpvbt27t6WkCxRKOGYisiIkLDhw9X%2B/btlZOTowEDBmj8%2BPEaPHiwSpQooXvvvdfVUwSKnN27d%2Btf//qXvv/%2Bey1evFhnz55VZGSklixZ4uqpAcUSH8qOYq9bt2568803FRAQ4Bg7evSonnjiCa1evdqFMwOKnn79%2BunBBx/Ugw8%2BqObNm2vbtm3atGmTpk6dqlWrVrl6ekCxQ6OGYi8%2BPl41a9Z0GqtSpYqSkpJcNCOg6Dpw4IC6d%2B8uSY5LC9q0aaPExERXTgsotghqKPbq16%2Bv6dOnKzMzU5KUnp6uSZMmqWnTpi6eGVD0VKxYUYcOHXIaO3TokCpXruyiGQHFGzcToNibOHGinnzySX355ZeqUKGCzp49q1q1aunDDz909dSAIqdPnz568skn9dRTTyk7O1urVq3Se%2B%2B9p/DwcFdPDSiWuEYNN4Xs7GxFR0crKSlJfn5%2BatKkidzcKJSBv2PBggX6/PPPdeLECVWpUkXh4eEaOHAgf6cAExDUUGz9%2Beef8vPzU0JCwlXXqVq16j84IwAAbgxBDcVWkyZNFB0drTp16uR5nprdbpfNZtPevXtdNDugaBo7dmy%2B4%2B7u7qpYsaLatWunxo0b/7OTAooxrlFDsbVy5UpJBXuobW77BuDa3N3dtXTpUnXs2FE1atRQQkKC1qxZo1atWuncuXP65JNPNHnyZN13332unipQLNCoAfpv%2Bwbg2oYMGaLw8HB17NjRMbZx40Z98cUXev/997Vlyxa99tprWrFihQtnCRQfXPkJ6MqpUADXt2PHDoWGhjqNtWnTRr/99pskKSQkRCdOnHDF1IBiiaAGSNf9TFAAV1SsWFGbNm1yGtu8ebO8vb0lXXnAdPny5V0wM6B44ho1AECB5X6G7r333qvq1avr%2BPHjWrt2rSZOnKhDhw5pwIAB6tevn6unCRQbXKMGiGvUgBuxfft2LVmyRH/%2B%2BaeqVq2q3r1764477tDx48cVFxfndP0aAGMIaoAIakBBHTt2TLNmzVJiYqJycnIkSVlZWTp8%2BLB%2B/fVXF88OKH64Rg0AUGAvvfSSTpw4oXLlyuny5csKCgpSbGwspzsBkxDUUOz99ttvjn/5X02pUqX%2BodkARduuXbs0a9YsDRs2TGXLltW4ceP0xhtvaPPmza6eGlAsEdRQ7D399NO6dOnSNdfhlA1QMJ6enipfvrxq1qypAwcOSJLatm2rQ4cOuXhmQPFEUEOxV6NGDe3cudPV0wCKhZo1a2rjxo0qU6aMcnJyFB8fr8TERGVnZ7t6akCxxOM5UOyVL19ejz32mKpXry5fX1%2BnZ6Z9%2BumnLpwZUPQ88cQTGjFihL799luFh4fr4YcfVokSJdShQwdXTw0olrjrE8XezJkzr7ps%2BPDh/%2BBMgOIhMTFRlSpVUsmSJbVq1SqlpKSoR48eXOsJmICgBgAAYFFco4abwldffaWuXbsqJCRECQkJGjFihFJTU109LQAAromghmJv3rx5mjNnjh599FFdvnxZZcqUUWJioqZOnerqqQEAcE0ENRR7X3zxhd5991317t1bbm5uKl%2B%2BvGbMmKENGza4emoAAFwTQQ3F3tmzZ1WrVi1JUu4lmZUqVeJxAgAAyyOoodirU6eOFi5cKEmOR3OsWrVKgYGBrpwWAADXxV2fKPZ2796tgQMHKiAgQLt27VLLli21fft2ffTRR2rUqJGrpwcAwFUR1HBTSEpK0jfffKOEhAT5%2Bfmpa9euqlq1qqunBQDANRHUUOyNGTNG9957r1q3bs0DOQEARQrXqKHYq1Chgv7973%2BrRYsWGjlypFatWsUz1AAARQKNGm4ahw4d0rp167Rhwwbt3btXISEhev/99109LQAAropGDTeNS5cuyWazydPTUzk5OTp8%2BLCrpwQAwDXRqKHYe/bZZ7V582bl5OTorrvu0t13363WrVurevXqrp4aAADXVNLVEwDMtn37dqWnp6tz585q06aNWrVqJW9vb1dPCwCA66JRw03h8OHD%2Bvnnn7Vp0yZFR0fL399frVu31siRI109NQAAroqghptKXFyc1q1bp3nz5ik1NVV//PGHq6cEAMBVEdRQ7K1bt04//fSTNm3apAsXLqhNmzbq2LGj7rnnHpUtW9bV0wMA4KoIaij27r77boWGhqpjx45q2bIlD70FABQZBDUUe3a7XTabTefPn1d8fLzq1q2r7OxsAhsAwPJ4jhqKvfT0dI0aNUohISHq16%2Bfjhw5orCwMB06dMjVUwMA4JoIaij2pk%2BfrrS0NH333Xdyd3dXjRo11L59e02ePNnVUwMA4Jp4jhqKvQ0bNmjFihUqX768bDab3N3dNWbMGLVt29bVUwMA4Jpo1FDs5eTkOK5Hy70k869jAABYFUENxV6LFi306quvKj09XTabTZL01ltv6a677nLxzAAAuDbu%2BkSxl5ycrKFDh2rPnj26fPmyPDw85O/vr/fff19VqlRx9fQAALgqghpuCna7XTt37tSJEyfk5%2Benhg0bqkSJEq6eFgAA10RQQ7H1559/ys/PTwkJCVddp2rVqv/gjAAAuDEENRRbTZo0UXR0tOrUqSObzeZ48K3034fg7t2718WzBADg6ng8B4qtlStXSpLuuecejRw5UrfccouLZwQAwI0hqKHYuvXWWyVJXl5eeuSRR3T77berV69e6tKli8qVK%2Bfi2QEAcH2c%2BsRN4eLFi1qxYoW%2B/vprHThwQPfee6969eql5s2bu3pqAABcFUENN53NmzfrpZde0smTJ7lGDQBgaZz6xE0hNTVV33//vb7%2B%2Bmv98ccfateunSZNmuTqaQEAcE00aij2Ro0apfXr18vPz0%2B9evVSjx49VLFiRVdPCwCA66JRQ7FXsmRJzZ49W82aNXP1VAAAuCE0agAAABbFh7IDAABYFEENAADAoghqAAAAFkVQAwAAsCiCGgAAgEUR1AAAACyKoAYAAGBRBDUAAACL%2Bv9T0iX3APHgdwAAAABJRU5ErkJggg%3D%3D" class="center-img">
</div>
    <div class="row headerrow highlight">
        <h1>Sample</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-12" style="overflow:scroll; width: 100%%; overflow-y: hidden;">
        <table border="1" class="dataframe sample">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>viewer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1918165</td>
      <td>Female</td>
      <td>39</td>
      <td>Dallas</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27662619</td>
      <td>Female</td>
      <td>28</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5493662</td>
      <td>Female</td>
      <td>53</td>
      <td>Detroit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14441247</td>
      <td>Male</td>
      <td>41</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25595927</td>
      <td>Male</td>
      <td>53</td>
      <td>Seattle</td>
    </tr>
  </tbody>
</table>
    </div>
</div>
</div>




```python
pp.ProfileReport(test_data[test_data.columns[1:]])
```




<meta charset="UTF-8">

<style>

        .variablerow {
            border: 1px solid #e1e1e8;
            border-top: hidden;
            padding-top: 2em;
            padding-bottom: 2em;
            padding-left: 1em;
            padding-right: 1em;
        }

        .headerrow {
            border: 1px solid #e1e1e8;
            background-color: #f5f5f5;
            padding: 2em;
        }
        .namecol {
            margin-top: -1em;
            overflow-x: auto;
        }

        .dl-horizontal dt {
            text-align: left;
            padding-right: 1em;
            white-space: normal;
        }

        .dl-horizontal dd {
            margin-left: 0;
        }

        .ignore {
            opacity: 0.4;
        }

        .container.pandas-profiling {
            max-width:975px;
        }

        .col-md-12 {
            padding-left: 2em;
        }

        .indent {
            margin-left: 1em;
        }

        .center-img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block;
        }

        /* Table example_values */
            table.example_values {
                border: 0;
            }

            .example_values th {
                border: 0;
                padding: 0 ;
                color: #555;
                font-weight: 600;
            }

            .example_values tr, .example_values td{
                border: 0;
                padding: 0;
                color: #555;
            }

        /* STATS */
            table.stats {
                border: 0;
            }

            .stats th {
                border: 0;
                padding: 0 2em 0 0;
                color: #555;
                font-weight: 600;
            }

            .stats tr {
                border: 0;
            }

            .stats td{
                color: #555;
                padding: 1px;
                border: 0;
            }


        /* Sample table */
            table.sample {
                border: 0;
                margin-bottom: 2em;
                margin-left:1em;
            }
            .sample tr {
                border:0;
            }
            .sample td, .sample th{
                padding: 0.5em;
                white-space: nowrap;
                border: none;

            }

            .sample thead {
                border-top: 0;
                border-bottom: 2px solid #ddd;
            }

            .sample td {
                width:100%;
            }


        /* There is no good solution available to make the divs equal height and then center ... */
            .histogram {
                margin-top: 3em;
            }
        /* Freq table */

            table.freq {
                margin-bottom: 2em;
                border: 0;
            }
            table.freq th, table.freq tr, table.freq td {
                border: 0;
                padding: 0;
            }

            .freq thead {
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;

            }

            td.fillremaining{
                width:auto;
                max-width: none;
            }

            td.number, th.number {
                text-align:right ;
            }

        /* Freq mini */
            .freq.mini td{
                width: 50%;
                padding: 1px;
                font-size: 12px;

            }
            table.freq.mini {
                 width:100%;
            }
            .freq.mini th {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 5em;
                font-weight: 400;
                text-align:right;
                padding-right: 0.5em;
            }

            .missing {
                color: #a94442;
            }
            .alert, .alert > th, .alert > td {
                color: #a94442;
            }


        /* Bars in tables */
            .freq .bar{
                float: left;
                width: 0;
                height: 100%;
                line-height: 20px;
                color: #fff;
                text-align: center;
                background-color: #337ab7;
                border-radius: 3px;
                margin-right: 4px;
            }
            .other .bar {
                background-color: #999;
            }
            .missing .bar{
                background-color: #a94442;
            }
            .tooltip-inner {
                width: 100%;
                white-space: nowrap;
                text-align:left;
            }

            .extrapadding{
                padding: 2em;
            }

            .pp-anchor{

            }

</style>

<div class="container pandas-profiling">
    <div class="row headerrow highlight">
        <h1>Overview</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-6 namecol">
        <p class="h4">Dataset info</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Number of variables</th>
                <td>8 </td>
            </tr>
            <tr>
                <th>Number of observations</th>
                <td>418026 </td>
            </tr>
            <tr>
                <th>Total Missing (%)</th>
                <td>1.6% </td>
            </tr>
            <tr>
                <th>Total size in memory</th>
                <td>25.5 MiB </td>
            </tr>
            <tr>
                <th>Average record size in memory</th>
                <td>64.0 B </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-6 namecol">
        <p class="h4">Variables types</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Numeric</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Categorical</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Boolean</th>
                <td>3 </td>
            </tr>
            <tr>
                <th>Date</th>
                <td>1 </td>
            </tr>
            <tr>
                <th>Text (Unique)</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Rejected</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Unsupported</th>
                <td>0 </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-12" style="padding-left: 1em;">

        <p class="h4">Warnings</p>
        <ul class="list-unstyled"><li><a href="#pp_var_tv_provider"><code>tv_provider</code></a> has 52720 / 12.6% missing values <span class="label label-default">Missing</span></li><li>Dataset has 233004 duplicate rows <span class="label label-warning">Warning</span></li> </ul>
    </div>
</div>
    <div class="row headerrow highlight">
        <h1>Variables</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_date">date<br/>
            <small>Date</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>5</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>
        </div>
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Minimum</th>
                    <td>2018-01-15 00:00:00</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>2018-01-19 00:00:00</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram2120813809216214877">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAFSElEQVR4nO3bT0jTDRzH8Y85PYhBCU4PXjwlhKWwg4jk35ZbrUEXiYmgBxURPHQweBARfLCLB0USPHRJQcHUpcE6CB7ECp%2BDHqSCOoQyZCsXNhNN%2Bz4HcfR7pt%2BHXNt%2B1Od12/z95evbn78fW4qICIjoROeSfQBEZmZJ9gH8l%2B0v30%2Bv88/ftXE4EiJeQYhUpruCJIpZr1RmPa6z%2BB3O5Y8N5E/2O/ziJgr/xSJSMBAiBQMhUjAQIgUDIVIwECIFAyFSMBAiBQMhUjAQIgUDIVIwECIFAyFSMBAiBQMhUjAQIgUDIVIwECIFAyFSMBAiBQMhUjAQIgUDIVIwECIFAyFSMBAiBQMhUjAQIgUDIVIwECIFAyFSMBAiBQMhUjAQIgUDIVIwECIFAyFSMBAiBQMhUjAQIgUDIVIwECIFAyFSpIiIJPsgjgUCAUxMTKCurg5WqzXZh0MJZsb5m%2BoKEgwGMTQ0hGAwmOxDoSQw4/xNFQiR2TAQIgUDIVKYKpDs7Gy0t7cjOzs72YdCSWDG%2BZvqKRaR2ZjqCkJkNgyESMFAiBQMhEjBQIgUDIRIwUCIFDEHMjs7C6fTCbvdjtHRUcPPDg4O0NjYiFevXqnbCIfDcLlc2NjYiLzX39%2BPqqoquN1uuN1ujI2NRa3n9/vh8XhQW1uL1tZWhMNhAMD29jaam5vhcDjg8XgQCARiPU06QbxmPzU1BYfDAZfLhd7eXhwcHEStl7DZSww2NzeloqJCtra2ZGdnR1wul7x580ZERN69eyd3796VwsJCefny5anbWFlZkdu3b8vly5dlfX098n5DQ4Osra2p%2B29ubhav1ysiIkNDQ9LX1yciIj09PfLw4UMREZmenpb29vZYTpNOEK/Zv3//XsrKymRzc1NERLq7u%2BXRo0dR6yZq9jFdQZaWllBSUoKLFy8iIyMDN27cwPPnzwEAk5OTaGpqwtWrV9VtjI%2BPo6ury/D5fxHB69evMTg4GPkrsr%2B/b1jv27dvWF5ehsPhAADcuXMnsu%2BFhQW43W4AwK1bt7C4uBi1PsUmXrN/%2B/YtiouLkZOTAwCorKzE/Py8Yb1Ezj6mQAKBgOHkrFZr5LP8nZ2dqKmp%2Bd9t9PX1wWazGd4LhUIoKipCZ2cnpqensb29jeHh4ahlMjMzkZaWBuDoczzH%2B/7xuCwWC86fP49QKHT2E6Uo8Zp9QUEBVldX4ff7cXh4CJ/PF/X9kETOPqZAvn//jpSUlMhrETG8PqusrCyMjIwgPz8fFosFjY2NWFhYMCxz0r5O27eI4Nw5Po/4leI1%2B/z8fNy7dw9tbW3weDy4dOlSJARtX/GafUy/Nbm5uYa6g8Gg%2BlXJ%2Bfn5yE33wMDAqct9%2BPABMzMzkdeHh4dITU01LJOVlYUvX75EbuB%2B3LfVasXHjx8BHN0s7uzs4MKFCz97eqSI1%2Bz39vZw5coVzMzMYHx8HFarFXl5eYZlEjn7mAIpLS3Fixcv8OnTJ3z9%2BhU%2Bnw/Xrl07dfnq6mp4vV54vV50dHSculx6ejoePHgAv98PEcHo6CiuX79uWCYtLQ02mw3Pnj0DcPTk43jf5eXlmJqaAgDMzc3BZrNF/RWi2MRr9ru7u2hoaEA4HMb%2B/j4eP34cudc4ltDZx3SLLyJPnz4Vp9MpdrtdRkZGon5eX1%2BvPsk4VllZaXiKNTc3Jzdv3hS73S7379%2BXvb29qHU2Njakvr5eHA6HNDU1yefPn0VEJBQKSUtLizidTqmrqzNsl36deM3%2ByZMn4nQ6paamRgYGBk5cJ1Gz5/dBiBS8cyVSMBAiBQMhUvwLW3Gp5Jgise4AAAAASUVORK5CYII%3D">
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives2120813809216214877,#minihistogram2120813809216214877"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives2120813809216214877">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAvBUlEQVR4nO3df1TVVb7/8Zf8UEBEMBKari27/KhphnUlVIQxKpS0m7%2BGMKahH1ajXaEfekWrkUlDQc2mDB2oS8soY9JRc8qbmdV1pWPmr0xvTho4XbXLKEigCKLC%2BXz/6HK%2BcwTzYPvIOafnYy3/OHvvsz/7/dls18tzjodulmVZAgAAgDE%2BXb0AAAAAb0PAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACG%2BXX1An4samoajM/p49NNffr01LffNspms4zP31W8tS7Je2vz1rokavNE3lqX5L21ubKuK6/sZXQ%2BZ/EKlgfz8emmbt26ycenW1cvxShvrUvy3tq8tS6J2jyRt9YleW9t3lgXAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPPr6gXghxk4c31XL8Fp7035RVcvAQCAy4JXsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGL/sGbiA2xdt6eoldAq/TNt1POmXqkv8LADugIAFAICX8qR/KO4sGNnVSzCKgAUA6DKeFAAkXh2E8/gMFgAAgGEELAAAAMMIWAAAAIa5bcDat2%2BfsrKyNHDgQA0dOlRz587V2bNnJUl79uzR%2BPHjFR8fr9TUVK1cudLhuWvWrFFaWpoGDBig9PR07d69297X2tqqBQsWKDk5WfHx8Zo8ebKqq6vt/bW1tcrOztbAgQOVmJiogoICtbS0XJ6iAQCAV3DLgGWz2fTwww9rxIgR2r59u1atWqW//OUvKi0t1YkTJzRp0iSNGzdOO3bsUEFBgebNm6e9e/dKkrZt26Y5c%2BZo/vz52rFjh8aMGaPJkyfr9OnTkqSSkhJt2bJFq1ev1ubNmxUQEKC8vDz7tadMmaKgoCBt3rxZq1at0tatW1VWVtYVtwEAAHgotwxYJ06cUE1NjWw2myzLkiT5%2BPgoMDBQGzZsUGhoqLKysuTn56ekpCSNHj1a5eXlkqSVK1fqjjvuUEJCgvz9/TVhwgSFhYVp3bp19v6JEyfqqquuUnBwsGbOnKlNmzbpyJEjOnTokLZv367p06crMDBQ/fr1U3Z2tn1uAAAAZ7hlwAoLC9OECRO0YMECxcXF6eabb1b//v01YcIEVVRUKDY21mF8dHS09u/fL0mqrKy8YH9DQ4OOHj3q0B8eHq7evXvrwIEDqqioUGhoqCIiIuz9UVFRqqqq0smTJ11YMQAA8CZu%2BT1YNptNAQEB%2Bt3vfqeMjAwdOnRIjzzyiIqKitTY2KjAwECH8QEBAWpqapKk7%2B1vbGyUJAUFBbXrb%2Bs7/7ltj5uamhQSEuLU%2Bqurq1VTU%2BPQ5ucXpL59%2Bzr1fGf5%2BrplPr4gPz/n1ttWl6fV19Wcvb%2Bu4M175ok1cdZcpyvPmeT9e%2BZNdbllwPrggw/0/vvva/367349RUxMjHJyclRQUKDRo0eroaHBYXxzc7N69uwp6btA1Nzc3K4/LCzMHpbaPo91/vMty2rX1/a4bX5nrFixQkuWLHFoy8nJ0WOPPeb0HN4oLMz5eyhJISGBFx8Eu87eX1dgz9wDZ8113OGcSd67Z95Ul1sGrL///e/2/zHYxs/PT/7%2B/oqNjdWWLY7f/FtZWamYmBhJ34WxioqKdv0pKSnq3bu3IiIiHN5GrKmpUX19vWJjY2Wz2VRfX6/jx48rPDxcknTw4EFFRkaqV69eTq8/MzNTqamp560/SHV1jU7P4QxPS/rO1u/r66OQkECdPHlara02F6/Ke5j%2B%2BeoMb94zTztnEmfNlbrynEnev2euqKurQrFbBqyhQ4fq97//vV566SVNnDhRVVVVKikp0ejRo5WWlqaFCxeqrKxMWVlZ2rVrl9auXavi4mJJUkZGhnJycnT77bcrISFB5eXlqq2tVVpamiQpPT1dJSUliouLU1hYmAoLCzV48GBdc801kqSEhAQVFhYqPz9fdXV1Ki4uVkZGRqfW37dv33ZvB9bUNKilxfsOQ2d0tv7WVtuP/p51hjvcK/bMPXDWXMdd7pO37pk31eWWASs6Olovv/yyFi1apFdeeUW9evXSmDFjlJOTo%2B7du2vp0qUqKChQUVGR%2BvTpo7y8PA0ZMkSSlJSUpFmzZmn27Nk6duyYoqOjVVpaqtDQUEnfvVXX0tKirKwsNTY2KjExUYsWLbJfu6ioSPn5%2BRo2bJh8fHw0btw4ZWdnd8FdAAAAnsotA5YkJScnKzk5ucO%2BuLg4LV%2B%2B/ILPHTt2rMaOHdthn7%2B/v3Jzc5Wbm9thf3h4uIqKijq/YAAAgP/jeR8uAAAAcHMELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMc9uAVV9frxkzZigxMVGDBg1Sdna2qqurJUl79uzR%2BPHjFR8fr9TUVK1cudLhuWvWrFFaWpoGDBig9PR07d69297X2tqqBQsWKDk5WfHx8Zo8ebJ9Xkmqra1Vdna2Bg4cqMTERBUUFKilpeXyFA0AALyC2wasRx99VE1NTfrggw%2B0ceNG%2Bfr66ne/%2B51OnDihSZMmady4cdqxY4cKCgo0b9487d27V5K0bds2zZkzR/Pnz9eOHTs0ZswYTZ48WadPn5YklZSUaMuWLVq9erU2b96sgIAA5eXl2a87ZcoUBQUFafPmzVq1apW2bt2qsrKyrrgFAADAQ7llwPriiy%2B0Z88ezZ8/XyEhIQoODtacOXOUm5urDRs2KDQ0VFlZWfLz81NSUpJGjx6t8vJySdLKlSt1xx13KCEhQf7%2B/powYYLCwsK0bt06e//EiRN11VVXKTg4WDNnztSmTZt05MgRHTp0SNu3b9f06dMVGBiofv36KTs72z43AACAM9wyYO3du1fR0dH605/%2BpLS0NA0dOlQLFizQlVdeqYqKCsXGxjqMj46O1v79%2ByVJlZWVF%2BxvaGjQ0aNHHfrDw8PVu3dvHThwQBUVFQoNDVVERIS9PyoqSlVVVTp58qQLKwYAAN7Er6sX0JETJ07owIED%2BvnPf641a9aoublZM2bM0BNPPKHw8HAFBgY6jA8ICFBTU5MkqbGx8YL9jY2NkqSgoKB2/W195z%2B37XFTU5NCQkKcWn91dbVqamoc2vz8gtS3b1%2Bnnu8sX1%2B3zMcX5Ofn3Hrb6vK0%2Brqas/fXFbx5zzyxJs6a63TlOZO8f8%2B8qS63DFjdu3eXJM2cOVM9evRQcHCwpkyZorvuukvp6elqbm52GN/c3KyePXtK%2Bi4QddQfFhZmD0ttn8c6//mWZbXra3vcNr8zVqxYoSVLlji05eTk6LHHHnN6Dm8UFub8PZSkkJDAiw%2BCXWfvryuwZ%2B6Bs%2BY67nDOJO/dM2%2Bqyy0DVnR0tGw2m86dO6cePXpIkmw2myTppz/9qf74xz86jK%2BsrFRMTIwkKSYmRhUVFe36U1JS1Lt3b0VERDi8jVhTU6P6%2BnrFxsbKZrOpvr5ex48fV3h4uCTp4MGDioyMVK9evZxef2ZmplJTUx3a/PyCVFfX2Im7cHGelvSdrd/X10chIYE6efK0WlttLl6V9zD989UZ3rxnnnbOJM6aK3XlOZO8f89cUVdXhWK3DFjJycnq16%2Bffvvb32revHk6c%2BaMXnjhBQ0fPlyjRo1SUVGRysrKlJWVpV27dmnt2rUqLi6WJGVkZCgnJ0e33367EhISVF5ertraWqWlpUmS0tPTVVJSori4OIWFhamwsFCDBw/WNddcI0lKSEhQYWGh8vPzVVdXp%2BLiYmVkZHRq/X379m33dmBNTYNaWrzvMHRGZ%2BtvbbX96O9ZZ7jDvWLP3ANnzXXc5T556555U11u%2BU8zf39/LVu2TL6%2BvhoxYoRGjBihyMhIFRYWKiwsTEuXLtX69euVmJiovLw85eXlaciQIZKkpKQkzZo1S7Nnz9bgwYP17rvvqrS0VKGhoZK%2Be6vu5ptvVlZWlm6%2B%2BWadOXNGixYtsl%2B7qKhILS0tGjZsmO666y7ddNNNys7O7oK7AAAAPJVbvoIlSREREXrhhRc67IuLi9Py5csv%2BNyxY8dq7NixHfb5%2B/srNzdXubm5HfaHh4erqKio8wsGAAD4P275ChYAAIAnI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYJjxgNXa2mp6SgAAAI9iPGClpKTo2WefVWVlpempAQAAPILxgPXII4/os88%2B06hRozR%2B/HgtX75cDQ0Npi8DAADgtowHrLvvvlvLly/X%2BvXrlZycrNLSUg0dOlTTpk3TJ598YvpyAAAAbsdlH3Lv37%2B/pk6dqvXr1ysnJ0cfffSRHnroIaWmpurVV1/ls1oAAMBr%2Bblq4j179ujPf/6z1q1bp7NnzyotLU3p6ek6duyYXnzxRf33f/%2B3nn/%2BeVddHgAAoMsYD1jFxcV6%2B%2B23dejQIcXFxWnq1KkaNWqUgoOD7WN8fX319NNPm740AACAWzAesN544w2NGTNGGRkZio6O7nBMVFSUcnNzTV8aAADALRgPWJs2bdKpU6dUX19vb1u3bp2SkpIUFhYmSbrhhht0ww03mL40AACAWzD%2BIfe//vWvGjFihFasWGFvW7hwoUaPHq2vvvrK9OUAAADcjvGA9eyzz%2Bq2227T1KlT7W0ffvihUlJSNH/%2BfNOXAwAAcDvGA9a%2Bffs0adIkde/e3d7m6%2BurSZMm6fPPPzd9OQAAALdjPGAFBwfr8OHD7dqPHj2qgIAA05cDAABwO8YD1ogRIzR79mx98sknOnXqlBobG/Xpp58qPz9faWlppi8HAADgdoz/L8Jp06bpyJEjevDBB9WtWzd7e1pammbMmGH6cgAAAG7HeMAKDAzUyy%2B/rK%2B//loHDhyQv7%2B/oqKi1L9/f9OXAgAAcEsu%2B1U51157ra699lpXTQ8AAOC2jAesr7/%2BWvn5%2Bdq1a5fOnTvXrv/LL780fUkAAAC3YjxgzZ49W1VVVcrNzVWvXr1MTw8AAOD2jAes3bt367XXXlN8fLzpqQEAADyC8a9pCAsLU8%2BePU1PCwAA4DGMB6x7771Xzz//vBoaGkxPDQAA4BGMv0X48ccf6/PPP1diYqKuuOIKh1%2BZI0kfffSR6UsCAAC4FeMBKzExUYmJiaanBQAA8BjGA9YjjzxiekoAAACPYvwzWJK0f/9%2BPfXUU/rVr36lY8eOqby8XNu2bXPFpQAAANyO8YD1xRdfaPz48frmm2/0xRdf6OzZs/ryyy/14IMPauPGjaYvBwAA4HaMB6znnntODz74oJYtWyZ/f39J0ty5c3XfffdpyZIlpi8HAADgdlzyCta4cePatd99993629/%2BZvpyAAAAbsd4wPL399epU6fatVdVVSkwMND05QAAANyO8YA1fPhw/f73v1ddXZ297eDBgyooKNAtt9xi%2BnIAAABux3jAeuKJJ9Tc3Kzk5GSdPn1a6enpGjVqlPz8/DRjxgzTlwMAAHA7xr8HKzg4WMuXL9fWrVv117/%2BVTabTbGxsbrpppvk4%2BOSb4UAAABwK8YDVpukpCQlJSW5anoAAAC3ZTxgpaamqlu3bhfs53cRAgAAb2c8YP3yl790CFjnzp3ToUOHtGnTJk2ZMsX05QAAANyO8YD16KOPdtj%2BxhtvaNeuXbrvvvtMXxIAAMCtXLZPnd966636%2BOOPL9flAAAAusxlC1jbt29Xjx49LtflAAAAuozxtwjPfwvQsiydOnVKBw4c4O1BAADwo2A8YP3kJz9p978I/f39df/992v06NGmLwcAAOB2jAes%2BfPnm54SAADAoxgPWDt27HB67KBBg0xfHgAAoMsZD1gTJkyQZVn2P23a3jZsa%2BvWrZu%2B/PJL05cHAADocsYD1uLFizVv3jw98cQTGjJkiPz9/bVnzx7Nnj1bv/71r3XrrbeaviQAAIBbMf41DQsWLNCsWbM0fPhwBQcHq0ePHho8eLDy8/O1dOlSXX311fY/AAAA3sh4wKqurtZVV13Vrj04OFh1dXWmLwcAAOB2jAesAQMG6Pnnn9epU6fsbfX19Vq4cKGSkpJMXw4AAMDtGP8MVl5enu6//36lpKSof//%2BkqSvv/5aV155pV5//XXTlwMAAHA7xl/BioqK0rp16zRt2jQNGDBAAwYM0MyZM/X2228rMjKy0/O1trbq3nvv1ZNPPmlv27Nnj8aPH6/4%2BHilpqZq5cqVDs9Zs2aN0tLSNGDAAKWnp2v37t0O8y1YsEDJycmKj4/X5MmTVV1dbe%2Bvra1Vdna2Bg4cqMTERBUUFKilpeUS7gQAAPixcsnvIgwJCdH48eN1zz336KmnntLYsWMVGBh4SXMtWbJEO3futD8%2BceKEJk2apHHjxmnHjh0qKCjQvHnztHfvXknStm3bNGfOHM2fP187duzQmDFjNHnyZJ0%2BfVqSVFJSoi1btmj16tXavHmzAgIClJeXZ59/ypQpCgoK0ubNm7Vq1Spt3bpVZWVll34zAADAj47xgGVZlp577jkNGjRIo0aN0tGjR/XEE0/oqaee0rlz5zo119atW7Vhwwbddttt9rYNGzYoNDRUWVlZ8vPzU1JSkkaPHq3y8nJJ0sqVK3XHHXcoISFB/v7%2BmjBhgsLCwrRu3Tp7/8SJE3XVVVcpODhYM2fO1KZNm3TkyBEdOnRI27dv1/Tp0xUYGKh%2B/fopOzvbPjcAAIAzjH8Ga9myZXr77bc1a9Ys5efnS5KGDx%2BuZ555RldccYVyc3Odmqe2tlYzZ85UcXGxwytIFRUVio2NdRgbHR2tVatWSZIqKyt15513tuvfv3%2B/GhoadPToUYfnh4eHq3fv3jpw4IAkKTQ0VBEREfb%2BqKgoVVVV6eTJkwoJCXFq7dXV1aqpqXFo8/MLUt%2B%2BfZ16vrN8fV3yAqTL%2BPk5t962ujytvq7m7P11BW/eM0%2BsibPmOl15ziTv3zNvqst4wFqxYoWefvpppaWlac6cOZKkf/3Xf1X37t1VUFDgVMCy2WyaPn26HnjgAV1//fUOfY2Nje3ebgwICFBTU9NF%2BxsbGyVJQUFB7frb%2Bs5/btvjpqYmpwPWihUrtGTJEoe2nJwcPfbYY04931uFhfXs1PiQkEt7W/nHqrP31xXYM/fAWXMddzhnkvfumTfVZTxgffPNN/rpT3/arv26667T8ePHnZrj5ZdfVvfu3XXvvfe26wsMDFRDQ4NDW3Nzs3r27Gnvb25ubtcfFhZmD0ttn8c6//mWZbXra3vcNr8zMjMzlZqa6tDm5xekurpGp%2BdwhqclfWfr9/X1UUhIoE6ePK3WVpuLV%2BU9TP98dYY375mnnTOJs%2BZKXXnOJO/fM1fU1VWh2HjAuvrqq7V371790z/9k0P7xx9/rH79%2Bjk1x9tvv63q6moNHDhQkuyB6cMPP9SMGTO0ZcsWh/GVlZWKiYmRJMXExKiioqJdf0pKinr37q2IiAhVVlba3yasqalRfX29YmNjZbPZVF9fr%2BPHjys8PFySdPDgQUVGRqpXr15O34O%2Bffu2ezuwpqZBLS3edxg6o7P1t7bafvT3rDPc4V6xZ%2B6Bs%2BY67nKfvHXPvKku4/80e%2Bihh/TMM8/o1VdflWVZ2rp1qxYuXKhnn322w1ekOrJ%2B/Xp99tln2rlzp3bu3KlRo0Zp1KhR2rlzp9LS0nT8%2BHGVlZXp3Llz%2BvTTT7V27Vr7564yMjK0du1affrppzp37pzKyspUW1urtLQ0SVJ6erpKSkp05MgRnTp1SoWFhRo8eLCuueYa9e/fXwkJCSosLNSpU6d05MgRFRcXKyMjw/RtAgAAXsz4K1h33nmnWlpaVFJSoubmZj399NO64oorNHXqVN19990/eP6wsDAtXbpUBQUFKioqUp8%2BfZSXl6chQ4ZIkpKSkjRr1izNnj1bx44dU3R0tEpLSxUaGirpu89CtbS0KCsrS42NjUpMTNSiRYvs8xcVFSk/P1/Dhg2Tj4%2BPxo0bp%2Bzs7B%2B8bgAA8ONhPGC98847GjlypDIzM/Xtt9/KsixdccUVP2jO%2BfPnOzyOi4vT8uXLLzh%2B7NixGjt2bId9/v7%2Bys3NveCH7cPDw1VUVHTpiwUAAD96xt8inDt3rv3D7H369PnB4QoAAMDTGA9Y/fv3t3%2BnFAAAwI%2BR8bcIY2JilJubq1deeUX9%2B/dXjx49HPrnzZtn%2BpIAAABuxXjAOnz4sBISEiSp3beZAwAA/BgYCVjz5s3T448/rqCgIC1btszElAAAAB7LyGewXn/99XbfgP7QQw%2BpurraxPQAAAAexUjAsiyrXdtnn32mM2fOmJgeAADAo3jeL9kCAABwcwQsAAAAw4wFrG7dupmaCgAAwKMZ%2B5qGuXPnOnzn1blz57Rw4UL17NnTYRzfgwUAALydkYA1aNCgdt95FR8fr7q6OtXV1Zm4BAAAgMcwErD47isAAID/jw%2B5AwAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhrltwNq/f78eeOABDR48WL/4xS80Y8YMffvtt5KkPXv2aPz48YqPj1dqaqpWrlzp8Nw1a9YoLS1NAwYMUHp6unbv3m3va21t1YIFC5ScnKz4%2BHhNnjxZ1dXV9v7a2lplZ2dr4MCBSkxMVEFBgVpaWi5P0QAAwCu4ZcBqbm7Wb37zG8XHx%2Bsvf/mL/vM//1P19fX67W9/qxMnTmjSpEkaN26cduzYoYKCAs2bN0979%2B6VJG3btk1z5szR/PnztWPHDo0ZM0aTJ0/W6dOnJUklJSXasmWLVq9erc2bNysgIEB5eXn2a0%2BZMkVBQUHavHmzVq1apa1bt6qsrKwrbgMAAPBQbhmwqqqqdP311ysnJ0fdu3dXWFiYMjMztWPHDm3YsEGhoaHKysqSn5%2BfkpKSNHr0aJWXl0uSVq5cqTvuuEMJCQny9/fXhAkTFBYWpnXr1tn7J06cqKuuukrBwcGaOXOmNm3apCNHjujQoUPavn27pk%2BfrsDAQPXr10/Z2dn2uQEAAJzhlgHrn//5n/XKK6/I19fX3vb%2B%2B%2B/rZz/7mSoqKhQbG%2BswPjo6Wvv375ckVVZWXrC/oaFBR48edegPDw9X7969deDAAVVUVCg0NFQRERH2/qioKFVVVenkyZOuKBUAAHghv65ewMVYlqVFixZp48aNeuONN/T6668rMDDQYUxAQICampokSY2NjRfsb2xslCQFBQW162/rO/%2B5bY%2BbmpoUEhLi1Jqrq6tVU1Pj0ObnF6S%2Bffs69Xxn%2Bfq6ZT6%2BID8/59bbVpen1dfVnL2/ruDNe%2BaJNXHWXKcrz5nk/XvmTXW5dcA6deqUnnrqKe3bt09vvPGGrrvuOgUGBqqhocFhXHNzs3r27Cnpu0DU3Nzcrj8sLMwelto%2Bj3X%2B8y3LatfX9rhtfmesWLFCS5YscWjLycnRY4895vQc3igszPl7KEkhIYEXHwS7zt5fV2DP3ANnzXXc4ZxJ3rtn3lSX2wasw4cPa%2BLEifrJT36iVatWqU%2BfPpKk2NhYbdmyxWFsZWWlYmJiJEkxMTGqqKho15%2BSkqLevXsrIiLC4W3Empoa1dfXKzY2VjabTfX19Tp%2B/LjCw8MlSQcPHlRkZKR69erl9NozMzOVmprq0ObnF6S6usbO3YSL8LSk72z9vr4%2BCgkJ1MmTp9XaanPxqryH6Z%2BvzvDmPfO0cyZx1lypK8%2BZ5P175oq6uioUu2XAOnHihO6//34NGTJEBQUF8vH5/3/BpaWlaeHChSorK1NWVpZ27dqltWvXqri4WJKUkZGhnJwc3X777UpISFB5eblqa2uVlpYmSUpPT1dJSYni4uIUFhamwsJCDR48WNdcc40kKSEhQYWFhcrPz1ddXZ2Ki4uVkZHRqfX37du33duBNTUNamnxvsPQGZ2tv7XV9qO/Z53hDveKPXMPnDXXcZf75K175k11uWXAeuutt1RVVaX33ntP69evd%2BjbvXu3li5dqoKCAhUVFalPnz7Ky8vTkCFDJElJSUmaNWuWZs%2BerWPHjik6OlqlpaUKDQ2V9N1bdS0tLcrKylJjY6MSExO1aNEi%2B/xFRUXKz8/XsGHD5OPjo3Hjxik7O/tylQ4AALyAWwasBx54QA888MAF%2B%2BPi4rR8%2BfIL9o8dO1Zjx47tsM/f31%2B5ubnKzc3tsD88PFxFRUWdWzAAAMA/8LwPFwAAALg5AhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwOlBbW6vs7GwNHDhQiYmJKigoUEtLS1cvCwAAeAgCVgemTJmioKAgbd68WatWrdLWrVtVVlbW1csCAAAegoB1nkOHDmn79u2aPn26AgMD1a9fP2VnZ6u8vLyrlwYAADwEAes8FRUVCg0NVUREhL0tKipKVVVVOnnyZBeuDAAAeAq/rl6Au2lsbFRgYKBDW9vjpqYmhYSEXHSO6upq1dTUOLT5%2BQWpb9%2B%2B5hYqydfXs/Kxn59z622ry9Pq62rO3l9X8OY988SaOGuu05XnTPL%2BPfOmughY5wkKCtLp06cd2toe9%2BzZ06k5VqxYoSVLlji0PfLII3r00UfNLPL/VFdX6/7ICmVmZhoPb12purpar732SpfXtbNgpPE5q6urtWLFii6vzTR32TNX8NZzJrnHvnHOOqeze%2BaK%2B%2BsK1dXVWrx4sVftmfdERUNiYmJUX1%2Bv48eP29sOHjyoyMhI9erVy6k5MjMz9dZbbzn8yczMNL7WmpoaLVmypN2rZZ7OW%2BuSvLc2b61LojZP5K11Sd5bmzfWxStY5%2Bnfv78SEhJUWFio/Px81dXVqbi4WBkZGU7P0bdvX69J4AAAoPN4BasDRUVFamlp0bBhw3TXXXfppptuUnZ2dlcvCwAAeAhewepAeHi4ioqKunoZAADAQ/EKlge78sor9cgjj%2BjKK6/s6qUY5a11Sd5bm7fWJVGbJ/LWuiTvrc0b6%2BpmWZbV1YsAAADwJryCBQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgOUCtbW1ys7O1sCBA5WYmKiCggK1tLS0G7d7927FxcU5Pe/p06eVmZmpt956y6H9P/7jP/Szn/1M8fHx9j8vvPDCBedZs2aN0tLSNGDAAKWnp2v37t32vtbWVi1YsEDJycmKj4/X5MmTVV1d7TG1tXn11Vd17733OrTV1dXpySef1C9%2B8QsNGjRI999/v7788ssuqet///d/Ha73xBNPqLGx8YLzeNKedba2Nu68Zy%2B99JLDz2B8fLxuuOEGjRgx4oLzeMqeXUptbTq7Z5e7Nkn65JNPlJ6erhtvvFG33nqrlixZou/7fu1L3Td3r6uNJ%2BzZ559/rl/96le68cYbNWLECK1YseJ75/khZ82lLBh3zz33WNOmTbOampqsw4cPW3fccYdVWlpq77fZbNbKlSutAQMGWLGxsU7N%2BdVXX1m//OUvrdjYWGv16tUOfY8%2B%2Bqi1ePFip%2Bb59NNPrfj4eGvnzp3W2bNnrVdffdVKTEy0mpqaLMuyrMWLF1ujR4%2B2qqqqrIaGBmvKlCnWxIkTPaI2y7KsxsZGa968eVZsbKx1zz33OPRNnjzZmjRpkvXtt99aZ86csRYtWmQlJydbjY2Nl7WuM2fOWLfddpv1zDPPWE1NTVZtba2VmZlpPfPMMx3O40l71tnaLMsz9ux8X375pTVo0CBr69atHfZ70p51tjbLuvQ9u9y1ffvtt9a//Mu/WB988IFlWZZVWVlpDRkyxFqzZk2H8/yQfXPnuizLc/bs73//u3XjjTdaRUVF1pkzZ6wDBw5YN910k/XWW291OM8PPWuuRMAy7H/%2B53%2Bs2NhY6%2BjRo/a2d99917rlllvsj5988knrrrvuspYuXerUD%2BMnn3xiJSUlWa%2B//rp1yy23tPvL8ZZbbrE2btzo1PqmTZtm5eXlObSNHDnSWrVqlWVZlpWSkmK988479r6amhrruuuusw4fPuz2tVmWZQ0fPtz693//d2vWrFkOf4nYbDYrOzvb2rdvn72toaHBio2NtT788MPLWtf69eutW265xWppabG3HTt2zPr66687nMuT9qyztVmWZ%2BzZPzpz5ow1YsQIq7i4%2BIJzedKedbY2y7q0Pdu3b99lr%2B2LL76wYmNjrffff9%2By2WxWZWWllZSUZK1fv77DuS5137Zu3erWdVmW5%2BzZH//4R2vYsGEO419%2B%2BWUrIyOjw7l%2ByFlzNb/L8zrZj0dFRYVCQ0MVERFhb4uKilJVVZVOnjypkJAQPf7444qMjNS2bducmvP666/Xxo0b1aNHD7366qsOfbW1taqqqtKf/vQn5eXlqXv37ho5cqQef/xx9ejRo91clZWVuvPOOx3aoqOjtX//fjU0NOjo0aOKjY2194WHh6t37946cOCAJLl1bZK0bNkyRUZGavHixTp48KC9vVu3bvrDH/7gMHb9%2BvUKCgrSmTNnLmtde/fu1fXXX68XX3xR77zzjiRpxIgRmjp1aodzedKedbY2yTP27B%2BVlpbK399fkyZNuuAYT9qzztYmXdqeXXvttdqyZctlre2GG27QyJEj9eijj8rX11etra269957L/j256Xum7vXJXnOntlsNgUGBjq0%2Bfj46G9/%2B1uHc/2Qs9avXz%2Bn1nup%2BAyWYY2Nje1%2BONoeNzU1SZIiIyM7NWdYWNgFA0VNTY0GDhyo9PR0/dd//ZdKS0u1efNmzZ8/3%2Bn1BQQEqKmpyf45maCgoHb9jY2Nbl9bZ67/0Ucfae7cuZo1a5bOnTt3Wes6ceKENm3apB49euj999/XsmXLtG3bNj377LMdjvekPetsbZ25flfuWZtTp07ptdde05QpU%2BTr63vBcZ60Z22cra0z1//HPQsMDLzstZ09e1ahoaF68cUXtWfPHr355pt69913tXLlyg7HX%2Bq%2BnTx50q3r6sz1u3rPUlJSdPjwYb3xxhs6e/asvvrqKy1fvlxnzpzpcPwPOWuuRsAyLCgoSKdPn3Zoa3vcs2fPiz7/Hz9o%2Bpvf/Oai46%2B//nqVl5dr%2BPDh6t69u6KiopSdna1169Z1OD4wMFDNzc0Obc3NzerZs6f9h/T89bf1u3ttzrAsS8XFxcrNzVVhYaHGjRt32evq3r27wsPDlZOTox49eqhfv356%2BOGH9d5773U43pP2rLO1OcMd9qzNe%2B%2B9p5CQEKWmpn7vOE/aszbO1uaMjvZMuvx/P5aXl%2Bubb77RyJEj5e/vrxtvvFH33Xef3nzzzQ7HX%2Bq%2B9erVy63rcoa77Fm/fv300ksv6c9//rOGDh2quXPnKiMjQyEhIR2O/yFnzdV4i9CwmJgY1dfX6/jx4woPD5ckHTx4UJGRkerVq9dFn/%2BP//vBGdu3b9fu3bv18MMP29vOnj2rgICAC66voqLCoa2yslIpKSnq3bu3IiIiVFlZaX9JtaamRvX19YqNjZXNZnPr2i7m9OnTmjp1qioqKlReXq4bbrhB0uXfs6ioKK1fv142m00%2BPt/9G8dms13wfwB50p51traLcZc9a7NhwwaNHj1a3bp1%2B95xnrRnna3tYi60Z9Ll37eqqiqdPXvWoc3Pz0/%2B/v4djr/UfUtOTlZpaanb1nUx7rRnjY2NCgkJ0apVq%2BxtCxcu1M9//vMOx/%2BQs%2BZqvIJlWP/%2B/ZWQkKDCwkKdOnVKR44cUXFxsTIyMlxyvcDAQC1evFhr166VzWZTRUWFiouLlZmZ2eH4jIwMrV27Vp9%2B%2BqnOnTunsrIy1dbWKi0tTZKUnp6ukpISHTlyRKdOnVJhYaEGDx6sa665xu1ru5ipU6fq6NGjWr16tcNfIJe7rttvv12tra0qLCzU2bNn9c033%2Bill17S2LFjOxzvSXvW2douxl32TPruX/i7d%2B/WoEGDLjrWk/ass7VdzIX2TLr8%2B5aamqpdu3ZpzZo1sixL%2B/fv17JlyzRmzJgOx1/qviUnJ7t1XRfjTnvW0NCgzMxMbdmyRTabTZ988olWrFih%2B%2B67r8PxP%2BSsuRqvYLlAUVGR8vPzNWzYMPn4%2BGjcuHHKzs52ybXi4uL0/PPP6w9/%2BIOefvpp9erVS3fddZf%2B7d/%2BrcPxSUlJmjVrlmbPnq1jx44pOjpapaWlCg0NlSTl5OSopaVFWVlZamxsVGJiohYtWuQRtX2fffv2aePGjerevbtuvfVWh77S0tLLWlefPn305ptvat68eUpJSZEkjRkzRtOmTetwvCftWWdr%2Bz7utGfSd98V1NDQ4PBh3wvxpD2TOlfb97nYng0cOPCy1pacnKznnntOL730kubMmaPw8HA9%2BOCD%2BvWvf93h%2BB%2Byb%2B5c1/dxtz2LjIzU888/r7lz5%2Bro0aO6%2BuqrlZ%2Bfr6FDh3Y4/oeeNVfqZl3qa/cAAADoEG8RAgAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACG/T82Mx%2B4WU96QQAAAABJRU5ErkJggg%3D%3D">
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_tv_make">tv_make<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>4</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable8446675513253182467">
    <table class="mini freq">
        <tr class="">
    <th>Sony</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 65.0%">
            271922
        </div>

    </td>
</tr><tr class="">
    <th>Toshiba</th>
    <td>
        <div class="bar" style="width:23%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 15.0%">
            62499
        </div>

    </td>
</tr><tr class="">
    <th>Philips</th>
    <td>
        <div class="bar" style="width:16%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 10.0%">
            &nbsp;
        </div>
        41836
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable8446675513253182467, #minifreqtable8446675513253182467"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable8446675513253182467">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">Sony</td>
        <td class="number">271922</td>
        <td class="number">65.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Toshiba</td>
        <td class="number">62499</td>
        <td class="number">15.0%</td>
        <td>
            <div class="bar" style="width:23%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Philips</td>
        <td class="number">41836</td>
        <td class="number">10.0%</td>
        <td>
            <div class="bar" style="width:16%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">LG</td>
        <td class="number">41769</td>
        <td class="number">10.0%</td>
        <td>
            <div class="bar" style="width:16%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_tv_size">tv_size<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>8</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>51.874</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>32</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>70</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram996793026437943608">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABDklEQVR4nO3cwQkCQRAAQU8MySDMybc5GYQ5jQlIg4LeIlX/Y%2BdYmvntNjNzAF467j0ArOy09wB7OV/vb3/zuF2%2BMAkrs0EgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAICz3LtY/vVf1q3/55JxVrXaXNggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQNhmZvYeAlZlg0AQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUB4Au/CFJESOqCqAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives996793026437943608,#minihistogram996793026437943608"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives996793026437943608">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles996793026437943608"
                                                  aria-controls="quantiles996793026437943608" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram996793026437943608" aria-controls="histogram996793026437943608"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common996793026437943608" aria-controls="common996793026437943608"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme996793026437943608" aria-controls="extreme996793026437943608"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles996793026437943608">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>32</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>32</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>40</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>55</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>65</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>70</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>70</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>38</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>25</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>12.225</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.23567</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.1847</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>51.874</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>10.631</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>-0.099146</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>21684727</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>149.46</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>3.2 MiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram996793026437943608">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAu%2BklEQVR4nO3df1jVdZ7//4f8MA4ichwEbS673BGYarORhUTUbMLIHH%2BsQxjbsJbV6K5QLl0TWmnpYKCO1RrjBdvY5TApOxqaObZW1myjDJKSuTo16YDbmK2jIALyQ5If7%2B8ffTmfjpQivhDO%2B9xv1%2BXldV6v9/t9ns/363R6cN5vjv0sy7IEAAAAY3x6uwAAAAC7IWABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMP8ersAb1FVVd/bJfQYH59%2BGjx4gM6ebVR7u9Xb5fQYb%2BjTG3qUvKNPerQPb%2BizJ3scMmSg0eN1FZ9g4ar5%2BPRTv3795OPTr7dL6VHe0Kc39Ch5R5/0aB/e0KcdeyRgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhfr1dANBXTVlT0tslXJG3Msb3dgkAgP8fn2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADOv1gHX27FklJiZq3759rrFDhw5p1qxZio6OVkJCgoqKitz22bZtmxITEzV69GglJSXp4MGDrrm2tjatWrVK48aNU3R0tObPn6/KykrXfHV1tdLS0hQbG6u4uDhlZ2ertbW1y88NAABwOb0asA4cOKCUlBR9/vnnrrG6ujrNmzdPM2fOVFlZmbKzs7VixQodPnxYkrRv3z4tX75cK1euVFlZmWbMmKH58%2Bfr/PnzkqT8/HyVlJRo69atKi4uVkBAgJYsWeI6fkZGhgIDA1VcXKwtW7aotLRUBQUFXXpuAACArui1gLVt2zY98cQTevzxx93Gd%2B3apZCQEKWmpsrPz0/x8fGaPn26CgsLJUlFRUWaOnWqYmJi5O/vrzlz5sjpdGrnzp2u%2Bblz52rYsGEKCgrS4sWLtWfPHp04cULHjx/X/v37lZmZKYfDoeHDhystLc117Ms9NwAAQFf0WsCaMGGC3n33Xf3oRz9yGy8vL1dUVJTbWEREhI4cOSJJqqio%2BNb5%2Bvp6nTp1ym0%2BNDRUgwYN0tGjR1VeXq6QkBCFh4e75keOHKmTJ0/q3Llzl31uAACArui1f%2Bx5yJAh3zje2Ngoh8PhNhYQEKCmpqbLzjc2NkqSAgMDO813zF28b8fjjv0v9dxdVVlZqaqqKrcxP79AhYWFXdFxPIWvr4/b3%2Bgdfn5Xf/69ZS29oU96tA9v6NOOPfZawPo2DodD9fX1bmPNzc0aMGCAa765ubnTvNPpdIWjjvuxLt7fsqxOcx2PBwwYcNnn7qrNmzdr7dq1bmPp6elasGDBFR3H0wQHOy6/EXqM03llr9NL8Za19IY%2B6dE%2BvKFPO/XY5wJWVFSUSkpK3MYqKioUGRkpSYqMjFR5eXmn%2BYkTJ2rQoEEKDw93u4xYVVWl2tpaRUVFqb29XbW1tTpz5oxCQ0MlSceOHdPQoUM1cODAyz53V6WkpCghIcFtzM8vUDU1jVd0HE/h6%2Buj4GCHzp07r7a29t4ux2uZeH15y1p6Q5/0aB/e0GdP9mjyh88r0ecCVmJiolavXq2CggKlpqbqwIED2rFjh/Ly8iRJycnJSk9P15QpUxQTE6PCwkJVV1crMTFRkpSUlKT8/HyNGjVKTqdTOTk5GjNmjG644QZJUkxMjHJycpSVlaWamhrl5eUpOTm5S8/dVWFhYZ0uB1ZV1au11Z7/YXRoa2u3fY99mclz7y1r6Q190qN9eEOfduqxzwUsp9Op9evXKzs7W7m5uRo8eLCWLFmisWPHSpLi4%2BO1dOlSLVu2TKdPn1ZERITWrVunkJAQSV9dimttbVVqaqoaGxsVFxenNWvWuI6fm5urrKwsTZo0ST4%2BPpo5c6bS0tK69NwAAABd0c%2ByLKu3i/AGVVX1l9/IQ/n5%2BcjpHKCamkbb/OQhSVPWlFx%2Boz7krYzxV30Mu67lxbyhT3q0D2/osyd7HDJkoNHjdZV9btcHAADoIwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYX02YH3yySdKTU1VbGysJkyYoOeee04XLlyQJB06dEizZs1SdHS0EhISVFRU5Lbvtm3blJiYqNGjRyspKUkHDx50zbW1tWnVqlUaN26coqOjNX/%2BfFVWVrrmq6urlZaWptjYWMXFxSk7O1utra3XpmkAAGALfTJgtbe361/%2B5V80efJk7d%2B/X1u2bNEf//hHrVu3TnV1dZo3b55mzpypsrIyZWdna8WKFTp8%2BLAkad%2B%2BfVq%2BfLlWrlypsrIyzZgxQ/Pnz9f58%2BclSfn5%2BSopKdHWrVtVXFysgIAALVmyxPXcGRkZCgwMVHFxsbZs2aLS0lIVFBT0xmkAAAAeqk8GrLq6OlVVVam9vV2WZUmSfHx85HA4tGvXLoWEhCg1NVV%2Bfn6Kj4/X9OnTVVhYKEkqKirS1KlTFRMTI39/f82ZM0dOp1M7d%2B50zc%2BdO1fDhg1TUFCQFi9erD179ujEiRM6fvy49u/fr8zMTDkcDg0fPlxpaWmuYwMAAHRFnwxYTqdTc%2BbM0apVqzRq1CjdcccdGjFihObMmaPy8nJFRUW5bR8REaEjR45IkioqKr51vr6%2BXqdOnXKbDw0N1aBBg3T06FGVl5crJCRE4eHhrvmRI0fq5MmTOnfuXA92DAAA7MSvtwv4Ju3t7QoICNAzzzyj5ORkHT9%2BXI8%2B%2Bqhyc3PV2Ngoh8Phtn1AQICampok6ZLzjY2NkqTAwMBO8x1zF%2B/b8bipqUnBwcFdqr%2ByslJVVVVuY35%2BgQoLC%2BvS/p7G19fH7W/0Dj%2B/qz//3rKW3tAnPdqHN/Rpxx77ZMB699139c477%2Bjtt9%2BWJEVGRio9PV3Z2dmaPn266uvr3bZvbm7WgAEDJH0ViJqbmzvNO51OV1jquB/r4v0ty%2Bo01/G44/hdsXnzZq1du9ZtLD09XQsWLOjyMTxRcLDj8huhxzidXX%2BNXo63rKU39EmP9uENfdqpxz4ZsP72t7%2B5fmOwg5%2Bfn/z9/RUVFaWSkhK3uYqKCkVGRkr6KoyVl5d3mp84caIGDRqk8PBwt8uIVVVVqq2tVVRUlNrb21VbW6szZ84oNDRUknTs2DENHTpUAwcO7HL9KSkpSkhIuKj%2BQNXUNHb5GJ7E19dHwcEOnTt3Xm1t7b1djtcy8frylrX0hj7p0T68oc%2Be7NHkD59Xok8GrAkTJuiFF17Qf/zHf2ju3Lk6efKk8vPzNX36dCUmJmr16tUqKChQamqqDhw4oB07digvL0%2BSlJycrPT0dE2ZMkUxMTEqLCxUdXW1EhMTJUlJSUnKz8/XqFGj5HQ6lZOTozFjxuiGG26QJMXExCgnJ0dZWVmqqalRXl6ekpOTr6j%2BsLCwTpcDq6rq1dpqz/8wOrS1tdu%2Bx77M5Ln3lrX0hj7p0T68oU879dgnA1ZERIRefvllrVmzRq%2B88ooGDhyoGTNmKD09Xf3799f69euVnZ2t3NxcDR48WEuWLNHYsWMlSfHx8Vq6dKmWLVum06dPKyIiQuvWrVNISIikry7Vtba2KjU1VY2NjYqLi9OaNWtcz52bm6usrCxNmjRJPj4%2BmjlzptLS0nrhLAAAAE/Vz%2Br4HgT0qKqq%2Bstv5KH8/HzkdA5QTU2jbX7ykKQpa0ouv1Ef8lbG%2BKs%2Bhl3X8mLe0Cc92oc39NmTPQ4Z0vVbfEyyz%2B36AAAAfQQBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMCwbgWstrY203UAAADYRrcC1sSJE/WLX/xCFRUVpusBAADweN0KWI8%2B%2Bqg%2B%2BugjTZs2TbNmzdKmTZtUX19vujYAAACP1K2Adf/992vTpk16%2B%2B23NW7cOK1bt04TJkzQz372M%2B3du9d0jQAAAB7lqm5yHzFihB5//HG9/fbbSk9P1%2B9//3s98sgjSkhI0K9//Wvu1QIAAF7J72p2PnTokN544w3t3LlTFy5cUGJiopKSknT69Gm99NJL%2BtOf/qQXX3zRVK0AAAAeoVsBKy8vT9u3b9fx48c1atQoPf7445o2bZqCgoJc2/j6%2BurZZ581VigAAICn6FbA2rhxo2bMmKHk5GRFRER84zYjR47UE088cVXFAQAAeKJuBaw9e/aooaFBtbW1rrGdO3cqPj5eTqdTknTzzTfr5ptvNlIkAACAJ%2BnWTe5//vOfNXnyZG3evNk1tnr1ak2fPl1/%2BctfjBUHAADgiboVsH7xi1/o7rvv1uOPP%2B4ae%2B%2B99zRx4kStXLnSWHEAAACeqFsB65NPPtG8efPUv39/15ivr6/mzZun//mf/zFVGwAAgEfqVsAKCgrS559/3mn81KlTCggIuOqiAAAAPFm3AtbkyZO1bNky7d27Vw0NDWpsbNQHH3ygrKwsJSYmmq4RAADAo3Trtwh/9rOf6cSJE3r44YfVr18/13hiYqIWLlxorDgAAABP1K2A5XA49PLLL%2Buzzz7T0aNH5e/vr5EjR2rEiBHGCqutrVVOTo52796t9vZ23XbbbVq2bJnCwsJ06NAhPffcc6qoqJDT6dT8%2BfM1a9Ys177btm1TXl6eqqqq9L3vfU/PPPOMoqOjJUltbW16/vnntX37dp0/f15jx47Vz3/%2Bc4WFhUmSqqur9cwzz2j//v3y9fXVjBkztGjRIvn5XdWX3gMAAC9yVf8W4d/93d/pnnvu0aRJk4yGK0l67LHH1NTUpHfffVfvv/%2B%2BfH199cwzz6iurk7z5s3TzJkzVVZWpuzsbK1YsUKHDx%2BWJO3bt0/Lly/XypUrVVZWphkzZmj%2B/Pk6f/68JCk/P18lJSXaunWriouLFRAQoCVLlrieNyMjQ4GBgSouLtaWLVtUWlqqgoICo70BAAB769bHMp999pmysrJ04MABtbS0dJr/9NNPr6qojz/%2BWIcOHdLevXtd//zO8uXLVVVVpV27dikkJESpqamSpPj4eE2fPl2FhYW69dZbVVRUpKlTpyomJkaSNGfOHG3evFk7d%2B7Uvffeq6KiIj3xxBMaNmyYJGnx4sWaMGGCTpw4ofb2du3fv1979uyRw%2BHQ8OHDlZaWptWrV%2BunP/3pVfUEAAC8R7cC1rJly3Ty5Ek98cQTGjhwoOmadPjwYUVEROi1117Tb3/7W50/f1633367Fi1apPLyckVFRbltHxERoS1btkiSKioqdO%2B993aaP3LkiOrr63Xq1Cm3/UNDQzVo0CAdPXpUkhQSEqLw8HDX/MiRI3Xy5EmdO3dOwcHBXaq/srJSVVVVbmN%2BfoGuy5B24%2Bvr4/Y3eoef39Wff29ZS2/okx7twxv6tGOP3QpYBw8e1G9%2B8xvXfU2m1dXV6ejRo7rlllu0bds2NTc3a%2BHChVq0aJFCQ0PlcDjctg8ICFBTU5MkqbGx8VvnGxsbJUmBgYGd5jvmLt6343FTU1OXA9bmzZu1du1at7H09HQtWLCgS/t7quBgx%2BU3Qo9xOgcYO5a3rKU39EmP9uENfdqpx24FLKfTqQEDzL2ZX6zjC0wXL16s6667TkFBQcrIyNB9992npKQkNTc3u23f3NzsqsfhcHzjvNPpdIWljvuxLt7fsqxOcx2Pr6TflJQUJSQkuI35%2BQWqpqaxy8fwJL6%2BPgoOdujcufNqa2vv7XK8lonXl7espTf0SY/24Q199mSPJn/4vBLdClizZ8/Wiy%2B%2BqNWrV/fIJcKIiAi1t7erpaVF1113nSSpvf2rE37TTTfpP//zP922r6ioUGRkpCQpMjJS5eXlneYnTpyoQYMGKTw8XBUVFa7LhFVVVaqtrVVUVJTa29tVW1urM2fOKDQ0VJJ07NgxDR069Ir6DAsL63Q5sKqqXq2t9vwPo0NbW7vte%2BzLTJ57b1lLb%2BiTHu3DG/q0U4/duti5e/dulZSUKC4uTrfffrsmTZrk9udqjRs3TsOHD9fTTz%2BtxsZGnT17Vv/%2B7/%2Buu%2B66S9OmTdOZM2dUUFCglpYWffDBB9qxY4frvqvk5GTt2LFDH3zwgVpaWlRQUKDq6mrXF6AmJSUpPz9fJ06cUENDg3JycjRmzBjdcMMNGjFihGJiYpSTk6OGhgadOHFCeXl5Sk5OvuqeAACA9%2BjWJ1hxcXGKi4szXYuLv7%2B/NmzYoJUrV2ry5Mn68ssvlZCQoMWLFys4OFjr169Xdna2cnNzNXjwYC1ZskRjx46V9NVvFS5dulTLli3T6dOnFRERoXXr1ikkJETSV/dCtba2KjU1VY2NjYqLi9OaNWtcz52bm6usrCxNmjRJPj4%2BmjlzptLS0nqsVwAAYD/9LMuyersIb1BVVd/bJfQYPz8fOZ0DVFPTaJuPdiVpypqS3i7hiryVMf6qj2HXtbyYN/RJj/bhDX32ZI9Dhpi/lakruv37kEeOHNFTTz2lf/qnf9Lp06dVWFioffv2mawNAADAI3UrYH388ceaNWuWvvjiC3388ce6cOGCPv30Uz388MN6//33TdcIAADgUboVsJ5//nk9/PDD2rBhg/z9/SVJzz33nB544IFO3/8EAADgbbr9CdbMmTM7jd9///363//936utCQAAwKN1K2D5%2B/uroaGh0/jJkyc7fRM6AACAt%2BlWwLrrrrv0wgsvqKamxjV27NgxZWdn64c//KGp2gAAADxStwLWokWL1NzcrHHjxun8%2BfNKSkrStGnT5Ofnp4ULF5quEQAAwKN064tGg4KCtGnTJpWWlurPf/6z2tvbFRUVpdtvv10%2BPvb5l7ABAAC6o1sBq0N8fLzi4%2BNN1QIAAGAL3QpYCQkJ6tev37fO//73v%2B92QQAAAJ6uWwHrxz/%2BsVvAamlp0fHjx7Vnzx5lZGSYqg0AAMAjdStgPfbYY984vnHjRh04cEAPPPDAVRUFAADgyYzekX7nnXdq9%2B7dJg8JAADgcYwGrP379%2Bu6664zeUgAAACP061LhBdfArQsSw0NDTp69CiXBwEAgNfrVsC6/vrrO/0Wob%2B/vx588EFNnz7dSGEAAACeqlsBa%2BXKlabrAAAAsI1uBayysrIub3vbbbd15ynQRVPWlPR2CV32Vsb43i4BfYQnvW4lz3vtetr5BSTpw%2Bx7ersEo7oVsObMmSPLslx/OnRcNuwY69evnz799FMDZQIAAHiObgWsX/7yl1qxYoUWLVqksWPHyt/fX4cOHdKyZcv0k5/8RHfeeafpOgEAADxGt76mYdWqVVq6dKnuuusuBQUF6brrrtOYMWOUlZWl9evX67vf/a7rDwAAgLfpVsCqrKzUsGHDOo0HBQWppqbmqosCAADwZN0KWKNHj9aLL76ohoYG11htba1Wr16t%2BPh4Y8UBAAB4om7dg7VkyRI9%2BOCDmjhxokaMGCFJ%2BuyzzzRkyBC9%2BuqrJusDAADwON0KWCNHjtTOnTu1Y8cOHTt2TJL0k5/8RFOnTpXD4TBaIAAAgKfpVsCSpODgYM2aNUtffPGFhg8fLumrb3MHAADwdt26B8uyLD3//PO67bbbNG3aNJ06dUqLFi3SU089pZaWFtM1AgAAeJRuBawNGzZo%2B/btWrp0qfr37y9Juuuuu/Tf//3feumll4wWCAAA4Gm6FbA2b96sZ599VklJSa5vb//Rj36k7Oxs/dd//ZfRAgEAADxNtwLWF198oZtuuqnT%2BPe//32dOXPmqosCAADwZN0KWN/97nd1%2BPDhTuO7d%2B923fAOAADgrbr1W4SPPPKIfv7zn%2Bv06dOyLEulpaXatGmTNmzYoKeeesp0jQAAAB6lWwHr3nvvVWtrq/Lz89Xc3Kxnn31W3/nOd/T444/r/vvvN10jAACAR%2BlWwPrd736ne%2B65RykpKTp79qwsy9J3vvMd07UBAAB4pG7dg/Xcc8%2B5bmYfPHgw4QoAAOBruhWwRowYoaNHj5quBQAAwBa6dYkwMjJSTzzxhF555RWNGDFC1113ndv8ihUrjBQHAADgiboVsD7//HPFxMRIkqqqqowWBAAA4Om6HLBWrFihf/u3f1NgYKA2bNjQkzUBAAB4tC7fg/Xqq6/q/PnzbmOPPPKIKisrjRcFAADgybocsCzL6jT20Ucf6csvvzRaEAAAgKfr1m8RAgAA4NsRsAAAAAy7ooDVr1%2B/nqoDAADANq7oaxqee%2B45t%2B%2B8amlp0erVqzVgwAC37fgeLAAA4M26HLBuu%2B22Tt95FR0drZqaGtXU1BgvDAAAwFN1OWDx3VcAAABdw03uAAAAhvX5gNXW1qbZs2frySefdI0dOnRIs2bNUnR0tBISElRUVOS2z7Zt25SYmKjRo0crKSlJBw8edDveqlWrNG7cOEVHR2v%2B/PluX5ZaXV2ttLQ0xcbGKi4uTtnZ2Wptbe35RgEAgG30%2BYC1du1affjhh67HdXV1mjdvnmbOnKmysjJlZ2drxYoVOnz4sCRp3759Wr58uVauXKmysjLNmDFD8%2BfPd30LfX5%2BvkpKSrR161YVFxcrICBAS5YscR0/IyNDgYGBKi4u1pYtW1RaWqqCgoJr2jMAAPBsfTpglZaWateuXbr77rtdY7t27VJISIhSU1Pl5%2Ben%2BPh4TZ8%2BXYWFhZKkoqIiTZ06VTExMfL399ecOXPkdDq1c%2BdO1/zcuXM1bNgwBQUFafHixdqzZ49OnDih48ePa//%2B/crMzJTD4dDw4cOVlpbmOjYAAEBXXNHXNFxL1dXVWrx4sfLy8tw%2BQSovL1dUVJTbthEREdqyZYskqaKiQvfee2%2Bn%2BSNHjqi%2Bvl6nTp1y2z80NFSDBg3S0aNHJUkhISEKDw93zY8cOVInT57UuXPnFBwc3KXaKysrO/3GpZ9foMLCwrq0v135%2BfXpPO/xTJxfX18ft7/xFV67wLVhp/eePhmw2tvblZmZqYceekg33nij21xjY6McDofbWEBAgJqami4739jYKEkKDAzsNN8xd/G%2BHY%2Bbmpq6HLA2b96stWvXuo2lp6drwYIFXdrfrpzOAZffCN1m8vwGBzsuv5EX4bULXBt2eu/pkwHr5ZdfVv/%2B/TV79uxOcw6HQ/X19W5jzc3Nri87dTgcam5u7jTvdDpdYanjfqyL97csq9Ncx%2BOLv0z1UlJSUpSQkOA25ucXqJqaxi4fw468vf%2BeZuL8%2Bvr6KDjYoXPnzqutrd1AVfbAaxe4Nnrivae3fkDqkwFr%2B/btqqysVGxsrCS5AtN7772nhQsXqqSkxG37iooKRUZGSpIiIyNVXl7eaX7ixIkaNGiQwsPDVVFR4bpMWFVVpdraWkVFRam9vV21tbU6c%2BaMQkNDJUnHjh3T0KFDNXDgwC7XHxYW1ulyYFVVvVpbvft/WN7ef08zeX7b2tpZr6/hXADXhp3ee/rkxc63335bH330kT788EN9%2BOGHmjZtmqZNm6YPP/xQiYmJOnPmjAoKCtTS0qIPPvhAO3bscN13lZycrB07duiDDz5QS0uLCgoKVF1drcTERElSUlKS8vPzdeLECTU0NCgnJ0djxozRDTfcoBEjRigmJkY5OTlqaGjQiRMnlJeXp%2BTk5N48HQAAwMP0yU%2BwLsXpdGr9%2BvXKzs5Wbm6uBg8erCVLlmjs2LGSpPj4eC1dulTLli3T6dOnFRERoXXr1ikkJETSV/dCtba2KjU1VY2NjYqLi9OaNWtcx8/NzVVWVpYmTZokHx8fzZw5U2lpab3QKQAA8FT9LMuyersIb1BVVX/5jbphypqSy2/UR7yVMb63S7ginnRuJTPn18/PR07nANXUNPbox/TeeG6vJU87v4AkfZh9T4%2B89wwZ0vVbfEzqk5cIAQAAPBkBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMKzPBqwjR47ooYce0pgxYzR%2B/HgtXLhQZ8%2BelSQdOnRIs2bNUnR0tBISElRUVOS277Zt25SYmKjRo0crKSlJBw8edM21tbVp1apVGjdunKKjozV//nxVVla65qurq5WWlqbY2FjFxcUpOztbra2t16ZpAABgC30yYDU3N%2BunP/2poqOj9cc//lFvvvmmamtr9fTTT6uurk7z5s3TzJkzVVZWpuzsbK1YsUKHDx%2BWJO3bt0/Lly/XypUrVVZWphkzZmj%2B/Pk6f/68JCk/P18lJSXaunWriouLFRAQoCVLlrieOyMjQ4GBgSouLtaWLVtUWlqqgoKC3jgNAADAQ/XJgHXy5EndeOONSk9PV//%2B/eV0OpWSkqKysjLt2rVLISEhSk1NlZ%2Bfn%2BLj4zV9%2BnQVFhZKkoqKijR16lTFxMTI399fc%2BbMkdPp1M6dO13zc%2BfO1bBhwxQUFKTFixdrz549OnHihI4fP679%2B/crMzNTDodDw4cPV1pamuvYAAAAXdEnA9b3vvc9vfLKK/L19XWNvfPOO/r7v/97lZeXKyoqym37iIgIHTlyRJJUUVHxrfP19fU6deqU23xoaKgGDRqko0ePqry8XCEhIQoPD3fNjxw5UidPntS5c%2Bd6olUAAGBDfr1dwOVYlqU1a9bo/fff18aNG/Xqq6/K4XC4bRMQEKCmpiZJUmNj47fONzY2SpICAwM7zXfMXbxvx%2BOmpiYFBwd3qebKykpVVVW5jfn5BSosLKxL%2B9uVn1%2BfzPO2YeL8%2Bvr6uP2Nr/DaBa4NO7339OmA1dDQoKeeekqffPKJNm7cqO9///tyOByqr6932665uVkDBgyQ9FUgam5u7jTvdDpdYanjfqyL97csq9Ncx%2BOO43fF5s2btXbtWrex9PR0LViwoMvHsCOns%2BvnEFfO5PkNDnZcfiMvwmsXuDbs9N7TZwPW559/rrlz5%2Br666/Xli1bNHjwYElSVFSUSkpK3LatqKhQZGSkJCkyMlLl5eWd5idOnKhBgwYpPDzc7TJiVVWVamtrFRUVpfb2dtXW1urMmTMKDQ2VJB07dkxDhw7VwIEDu1x7SkqKEhIS3Mb8/AJVU9N4ZSfBZry9/55m4vz6%2BvooONihc%2BfOq62t3UBV9sBrF7g2euK9p7d%2BQOqTAauurk4PPvigxo4dq%2BzsbPn4/L%2BPDBMTE7V69WoVFBQoNTVVBw4c0I4dO5SXlydJSk5OVnp6uqZMmaKYmBgVFhaqurpaiYmJkqSkpCTl5%2Bdr1KhRcjqdysnJ0ZgxY3TDDTdIkmJiYpSTk6OsrCzV1NQoLy9PycnJV1R/WFhYp8uBVVX1am317v9heXv/Pc3k%2BW1ra2e9voZzAVwbdnrv6ZMB6/XXX9fJkyf11ltv6e2333abO3jwoNavX6/s7Gzl5uZq8ODBWrJkicaOHStJio%2BP19KlS7Vs2TKdPn1aERERWrdunUJCQiR9damutbVVqampamxsVFxcnNasWeM6fm5urrKysjRp0iT5%2BPho5syZSktLu1atAwAAG%2BhnWZbV20V4g6qq%2Bstv1A1T1pRcfqM%2B4q2M8b1dwhXxpHMrmTm/fn4%2BcjoHqKamsUd/ivTGc3stedr5BSTpw%2Bx7euS9Z8iQrt/iY5J9btcHAADoIwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQSsb1BdXa20tDTFxsYqLi5O2dnZam1t7e2yAACAhyBgfYOMjAwFBgaquLhYW7ZsUWlpqQoKCnq7LAAA4CEIWBc5fvy49u/fr8zMTDkcDg0fPlxpaWkqLCzs7dIAAICHIGBdpLy8XCEhIQoPD3eNjRw5UidPntS5c%2Bd6sTIAAOAp/Hq7gL6msbFRDofDbazjcVNTk4KDgy97jMrKSlVVVbmN%2BfkFKiwszFyhHsjPjzzfk0ycX19fH7e/8RVeu8C1Yaf3HgLWRQIDA3X%2B/Hm3sY7HAwYM6NIxNm/erLVr17qNPfroo3rsscfMFPk1H2bfY/yYV6qyslKbN29WSkqKrULkxefWrn1%2BXWVlpX7zm1d6vMfeft3afS0/zL7H9j1K9l/HDt7QZ2VlpX75y1/aqkf7REVDIiMjVVtbqzNnzrjGjh07pqFDh2rgwIFdOkZKSopef/11tz8pKSk9VXKvq6qq0tq1azt9amc33tCnN/QoeUef9Ggf3tCnHXvkE6yLjBgxQjExMcrJyVFWVpZqamqUl5en5OTkLh8jLCzMNgkcAABcOT7B%2Bga5ublqbW3VpEmTdN999%2Bn2229XWlpab5cFAAA8BJ9gfYPQ0FDl5ub2dhkAAMBD8QkWrtqQIUP06KOPasiQIb1dSo/yhj69oUfJO/qkR/vwhj7t2GM/y7Ks3i4CAADATvgECwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQtXpLS0VLNmzdI//MM/aPz48Vq%2BfLmam5slSUuXLtUtt9yi6Oho15/Nmzf3csXd19bWptmzZ%2BvJJ590jR06dEizZs1SdHS0EhISVFRU1IsVXr1v6tFO67hz507dfPPNbr1kZmZKstdaXqpPu6xnbW2tFi5cqLi4ON12221KS0tTZWWlJHut5aX6tMNa/u53v3OrPzo6WrfccotuueUWSfZaS1lAF1VXV1ujRo2ytm7darW1tVmnT5%2B2pk2bZr300kuWZVnWj3/8Y%2Bv111/v5SrNWbNmjXXjjTdaixYtsizLsmpra60xY8ZYGzdutFpaWqy9e/da0dHR1qFDh3q50u67uEfLstc6rly50nryySc7jdttLb%2BtT8uyz3r%2B8z//s5Wenm7V1dVZ9fX11qOPPmrNmzfPdmv5bX1aln3W8utOnTpljR8/3nrjjTdst5b8Y8/ossGDB2vv3r0KCgqSZVmqra3Vl19%2BqcGDB%2BvChQv6y1/%2B4vopxNOVlpZq165duvvuu11ju3btUkhIiFJTUyVJ8fHxmj59ugoLC3Xrrbf2Vqnd9k092m0d//SnP2nKlCmdxu22lt/Wp13W8%2BOPP9ahQ4dc7z%2BStHz5clVVVdlqLS/Vp13W8ussy1JmZqZ%2B%2BMMf6h//8R9VVFRkm7WUuESIK9TxH/0dd9yh6dOna8iQIUpKStKRI0fU2tqq3NxcjRs3TpMnT9avfvUrtbe393LFV666ulqLFy/WCy%2B8IIfD4RovLy9XVFSU27YRERE6cuTItS7xqn1bj3Zax/b2dn3yySf6wx/%2BoDvvvFMTJ07UM888o7q6Olut5aX6tMt6Hj58WBEREXrttdeUmJioCRMmaNWqVRoyZIit1vJSfdplLb9u%2B/btqqiocN2iYKe1lAhY6KZdu3Zpz5498vHx0YIFC1RfX68xY8Zo9uzZ2r17t1avXq0NGzZo/fr1vV3qFWlvb1dmZqYeeugh3XjjjW5zjY2NbmFEkgICAtTU1HQtS7xql%2BrRLusoSWfPntXNN9%2BsyZMna%2BfOndq0aZP%2B%2Bte/KjMz0zZrKV26T7usZ11dnY4ePaq//vWv2rZtm9544w2dPn1aixYtstVaXqpPu6xlh/b2duXn5%2Btf//VfXT%2B422ktJQIWuikgIEDh4eHKzMxUcXGxbrnlFr366qsaM2aM/P39deutt%2BrBBx/Uzp07e7vUK/Lyyy%2Brf//%2Bmj17dqc5h8PhuqG/Q3NzswYMGHCtyjPiUj2OHz/eFusoSaGhoSosLFRycrIcDoeuv/56ZWZmas%2BePbIsyxZrKV26zx/84Ae2WM/%2B/ftLkhYvXqygoCCFhoYqIyNDu3fvttVaXqrP0aNH22ItO%2Bzbt0%2BVlZVKTk52jdnlPbYDAQtd9tFHH%2Bmee%2B7RhQsXXGMXLlyQv7%2B/SkpKtGnTJrftL1y4oICAgGtd5lXZvn279u/fr9jYWMXGxurNN9/Um2%2B%2BqdjYWEVFRam8vNxt%2B4qKCkVGRvZStd1zqR7fe%2B89W6yj9NXlzueff16WZbnGLly4IB8fH9166622WEvp0n3u2bPHFusZERGh9vZ2tbS0uMY6Lo3ddNNNtlnLS/X57rvv2mItO7zzzjtKTExUYGCga8wu77EuvXmHPTxLQ0ODdccdd1g5OTnWl19%2BaX3xxRdWcnKytXTpUmvXrl3Wrbfeau3du9dqb2%2B3PvroIysuLs564403ervsq7Jo0SLXb9idPXvWio2NtX79619bFy5csEpLS63o6GirtLS0l6u8Ol/v0U7r%2BLe//c0aPXq09atf/cpqaWmx/u///s%2B67777rKefftpWa3mpPu2ynhcuXLASExOtxx57zGpoaLCqq6utBx54wEpPT7fVWl6qT7usZYdp06ZZr732mtuYndbSsiyLgIUrUl5ebj300ENWbGysdeedd1ovvvii9eWXX1qWZVm//e1vrbvvvtv6wQ9%2BYE2aNMnauHFjL1d79b4ePizLsg4fPmylpKRY0dHR1qRJk6ytW7f2YnVmXNyjndZx3759rvUaO3astXz5cqu5udmyLHut5aX6tMt6njp1ysrIyLDGjx9vxcbGWgsXLrTq6uosy7LXWl6qT7uspWVZ1ujRo60//OEPncbttJb9LOtrnysDAADgqnEPFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABg2P8H0a4e/H7650YAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common996793026437943608">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">40</td>
        <td class="number">52388</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">65</td>
        <td class="number">52305</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">32</td>
        <td class="number">52286</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">70</td>
        <td class="number">52277</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">55</td>
        <td class="number">52254</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">60</td>
        <td class="number">52240</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">43</td>
        <td class="number">52190</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">50</td>
        <td class="number">52086</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme996793026437943608">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">32</td>
        <td class="number">52286</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">40</td>
        <td class="number">52388</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">43</td>
        <td class="number">52190</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">50</td>
        <td class="number">52086</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">55</td>
        <td class="number">52254</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">50</td>
        <td class="number">52086</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">55</td>
        <td class="number">52254</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">60</td>
        <td class="number">52240</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">65</td>
        <td class="number">52305</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">70</td>
        <td class="number">52277</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_uhd_capable">uhd_capable<br/>
            <small>Boolean</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr class="">
                    <th>Distinct count</th>
                    <td>2</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
            </table>
        </div>
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Mean</th>
                    <td>0.20086</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minifreqtable1410013967178828781">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 79.9%">
            334062
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:25%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 20.1%">
            83964
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable1410013967178828781, #minifreqtable1410013967178828781"
        aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable1410013967178828781">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">334062</td>
        <td class="number">79.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">83964</td>
        <td class="number">20.1%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_tv_provider">tv_provider<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>6</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="alert">
            <th>Missing (%)</th>
            <td>12.6%</td>
        </tr>
        <tr class="alert">
            <th>Missing (n)</th>
            <td>52720</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable5199773232418152377">
    <table class="mini freq">
        <tr class="">
    <th>Comcast</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 26.3%">
            109796
        </div>

    </td>
</tr><tr class="">
    <th>Time Warner Cable</th>
    <td>
        <div class="bar" style="width:99%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 26.2%">
            109714
        </div>

    </td>
</tr><tr class="">
    <th>Cox</th>
    <td>
        <div class="bar" style="width:66%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 17.4%">
            72809
        </div>

    </td>
</tr><tr class="other">
    <th>Other values (2)</th>
    <td>
        <div class="bar" style="width:66%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 17.5%">
            72987
        </div>

    </td>
</tr><tr class="missing">
    <th>(Missing)</th>
    <td>
        <div class="bar" style="width:48%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 12.6%">
            52720
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable5199773232418152377, #minifreqtable5199773232418152377"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable5199773232418152377">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">Comcast</td>
        <td class="number">109796</td>
        <td class="number">26.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Time Warner Cable</td>
        <td class="number">109714</td>
        <td class="number">26.2%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Cox</td>
        <td class="number">72809</td>
        <td class="number">17.4%</td>
        <td>
            <div class="bar" style="width:66%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">DirecTV</td>
        <td class="number">36738</td>
        <td class="number">8.8%</td>
        <td>
            <div class="bar" style="width:34%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">Dish Network</td>
        <td class="number">36249</td>
        <td class="number">8.7%</td>
        <td>
            <div class="bar" style="width:33%">&nbsp;</div>
        </td>
</tr><tr class="missing">
        <td class="fillremaining">(Missing)</td>
        <td class="number">52720</td>
        <td class="number">12.6%</td>
        <td>
            <div class="bar" style="width:48%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_total_time_watched">total_time_watched<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>95</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>10.04</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0.25</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>23.75</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-262188564383134968">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABMUlEQVR4nO3bwW2DQABFwRC5pBSRnnJ2Ty7CPW0aiJ7AEmYDM3dLe3n6wMrLGGN8AH/6PPoAMLPb0Qc4ytfPY/NvnvfvHU7CzCwIBIFAEAiE6d5BvBswEwsCYZntHuSVBZmVZfv/LAgEgUCY7iX9TN71uOhRbj8WBIJAIAgEgkAgCASCQCAIBIJAILgoPIGtF5IuFtezIBAEAkEgEAQCwUv6Bflb83oWBIJAIAgEgkAgCASCr1isctUvXxYEggVhN2dYHQsCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJhGWOMow8Bs7IgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEH4BW70klq1xxY8AAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-262188564383134968,#minihistogram-262188564383134968"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-262188564383134968">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-262188564383134968"
                                                  aria-controls="quantiles-262188564383134968" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-262188564383134968" aria-controls="histogram-262188564383134968"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-262188564383134968" aria-controls="common-262188564383134968"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-262188564383134968" aria-controls="extreme-262188564383134968"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-262188564383134968">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0.25</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1.25</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>9.5</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>14.5</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>21.25</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>23.75</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>23.5</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>9.5</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>6.1797</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.61553</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.84981</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>10.04</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>5.2026</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.33999</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>4196900</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>38.189</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>3.2 MiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-262188564383134968">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAw1klEQVR4nO3dfVBUV4L//4/QKg0INCLoTGk5ETCTRwlGomacDQnjGqNDEMNmqEzMg24JiYsV0E0kDwMBdUwcdS2omWQSJglbsui6apbJmMxmImMQGOOoyQQFN6umKOVBQGgg8tC/P/Kjv9OiCcYj3eD7VdVF3XPuPef0qdvdH%2B69fXuEw%2BFwCAAAAMZ4uXsAAAAAww0BCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYZnH3AK4X9fWtV7W9l9cIBQf76dw5u3p7HYZGhW/CnA8%2B5nzwMeeDjzkfXOPGjXFLvxzBGiK8vEZoxIgR8vIa4e6hXDeY88HHnA8%2B5nzwMefXBwIWAACAYR4bsJqbm7Vq1SrFxMTozjvvVEpKiurq6iRJhw8f1uLFixUVFaXY2FgVFxe7bLtz507FxcVp2rRpSkhI0KFDh5x1PT09Wr9%2BvWbNmqWoqCgtX77c2a4kNTY2KiUlRdOnT1dMTIxycnLU3d09OE8aAAAMCx4bsJ5%2B%2Bmm1t7fr/fff14cffihvb289//zzamlp0bJlyxQfH6/Kykrl5ORo7dq1OnLkiCSpvLxc2dnZWrdunSorK7Vw4UItX75cHR0dkqT8/Hzt379fO3bsUGlpqXx8fJSZmensNy0tTb6%2BviotLdX27dtVVlamgoICd0wBAAAYojwyYH366ac6fPiw1q1bp4CAAPn7%2Bys7O1vp6enau3evgoKClJycLIvFopkzZ2rBggUqLCyUJBUXF2v%2B/PmKjo7WyJEjtWTJEtlsNpWUlDjrly5dqgkTJsjf319r1qzRvn37dPr0aZ08eVIVFRXKyMiQ1WrVxIkTlZKS4mwbAABgIDwyYB05ckTh4eH6j//4D8XFxenuu%2B/W%2BvXrNW7cOFVXVysyMtJl/fDwcFVVVUmSampqLlvf2tqqM2fOuNSHhIQoMDBQx44dU3V1tYKCghQWFuasnzJlimpra3X%2B/Plr%2BIwBAMBw4pG3aWhpadGxY8d0yy23aOfOners7NSqVau0evVqhYSEyGq1uqzv4%2BOj9vZ2SZLdbr9svd1ulyT5%2Bvr2q%2B%2Bru3jbvuX29nYFBAQMaPx1dXWqr693KbNYfBUaGjqg7S/F29vL5S%2BuPeZ88DHng485H3zM%2BfXBIwPWqFGjJElr1qzR6NGj5e/vr7S0ND300ENKSEhQZ2eny/qdnZ3y8/OT9HUgulS9zWZzhqW%2B67Eu3t7hcPSr61vua38gioqKtHXrVpey1NRUrVixYsBtXE5AgPXbV4JRzPngY84HH3M%2B%2BJjz4c0jA1Z4eLh6e3vV1dWl0aNHS5J6e3slST/84Q/17//%2B7y7r19TUKCIiQpIUERGh6urqfvVz5sxRYGCgwsLCXE4j1tfXq7m5WZGRkert7VVzc7MaGhoUEhIiSTpx4oTGjx%2BvMWMGfqOypKQkxcbGupRZLL5qarJfwSy48vb2UkCAVefPd6inp/c7t4OBY84HH3M%2B%2BJjzwcecDy6bbeAHSEzyyIA1a9YsTZw4Uc8995zWrl2rr776Sr/61a9033336YEHHtCWLVtUUFCg5ORkHTx4UHv27FFeXp4kKTExUampqZo3b56io6NVWFioxsZGxcXFSZISEhKUn5%2BvW2%2B9VTabTbm5uZoxY4YmTZokSYqOjlZubq6ysrLU1NSkvLw8JSYmXtH4Q0ND%2B50OrK9vVXf31b%2BQenp6jbSDgWPOBx9zPviY88HHnA9vIxwOh0fep//s2bPOWy189dVXio2N1Zo1axQQEKCjR48qJydHx48fV3BwsFJSUpSQkODcdteuXcrPz9fZs2cVHh6uzMxM3X777ZKkrq4ubd68Wbt375bdbldMTIyys7M1duxYSVJDQ4OysrJUXl4uLy8vxcfHKz09Xd7e3lf1fK72p3IsFi/ZbH5qarLzghwkzPngY84HH3M%2B%2BJjzweWun8rx2IA13BCwhh7mfPAx54OPOR98zPng4rcIAQAAhgkCFgAAgGEeeZE74AniXil19xCuyO/TZrt7CACA/x9HsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhmcfcAcHXmbdrv7iEM2O/TZrt7CAAADAqOYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjJ/KAYYJfjYJADwHR7AAAAAM89gjWCUlJUpPT9fo0aOdZffdd582bNigw4cP6%2BWXX1ZNTY1sNpuWL1%2BuxYsXO9fbuXOn8vLyVF9frxtuuEHPP/%2B8oqKiJEk9PT165ZVXtGvXLnV0dOiuu%2B7SL37xC4WGhkqSGhsb9fzzz6uiokLe3t5auHChVq9eLYvFY6dqyBhKR1gAALgaHnsE6%2BjRo/rpT3%2BqQ4cOOR8bNmxQS0uLli1bpvj4eFVWVionJ0dr167VkSNHJEnl5eXKzs7WunXrVFlZqYULF2r58uXq6OiQJOXn52v//v3asWOHSktL5ePjo8zMTGe/aWlp8vX1VWlpqbZv366ysjIVFBS4YwoAAMAQ5dEB65ZbbulXvnfvXgUFBSk5OVkWi0UzZ87UggULVFhYKEkqLi7W/PnzFR0drZEjR2rJkiWy2WwqKSlx1i9dulQTJkyQv7%2B/1qxZo3379un06dM6efKkKioqlJGRIavVqokTJyolJcXZNgAAwEB4ZMDq7e3VZ599pj/96U%2B65557NGfOHD3//PNqaWlRdXW1IiMjXdYPDw9XVVWVJKmmpuay9a2trTpz5oxLfUhIiAIDA3Xs2DFVV1crKChIYWFhzvopU6aotrZW58%2Bfv4bPGAAADCceeWHRuXPndNNNN2nu3LnasmWLmpqatHr1amVkZGjcuHGyWq0u6/v4%2BKi9vV2SZLfbL1tvt9slSb6%2Bvv3q%2B%2Bou3rZvub29XQEBAQMaf11dnerr613KLBZf53Ve34W3t5fLX2Aos1guvR%2Bznw8%2B5nzwMefXB48MWCEhIS6n5axWqzIyMvTQQw8pISFBnZ2dLut3dnbKz8/Pue6l6m02mzMs9V2PdfH2DoejX13fcl/7A1FUVKStW7e6lKWmpmrFihUDbuNyAgKs374S4OFstm9%2BPbGfDz7mfPAx58ObRwasqqoqvfvuu3rmmWc0YsQISdKFCxfk5eWl2267Tb/73e9c1q%2BpqVFERIQkKSIiQtXV1f3q58yZo8DAQIWFhbmcRqyvr1dzc7MiIyPV29ur5uZmNTQ0KCQkRJJ04sQJjR8/XmPGjBnw%2BJOSkhQbG%2BtSZrH4qqnJfmUT8Xe8vb0UEGDV%2BfMd6unp/c7tAJ7gcq8F9vPBx5wPPuZ8cH3bP3TXikcGrKCgIBUWFiowMFCPPfaY6urqtGHDBj344IOaO3euXn31VRUUFCg5OVkHDx7Unj17lJeXJ0lKTExUamqq5s2bp%2BjoaBUWFqqxsVFxcXGSpISEBOXn5%2BvWW2%2BVzWZTbm6uZsyYoUmTJkmSoqOjlZubq6ysLDU1NSkvL0%2BJiYlXNP7Q0NB%2BpwPr61vV3X31L6Senl4j7QDu9G37MPv54GPOBx9zPryNcDgcDncP4lIqKiq0ceNGHT9%2BXKNHj9b8%2BfOVkZGh0aNH6%2BjRo8rJydHx48cVHByslJQUJSQkOLfdtWuX8vPzdfbsWYWHhyszM1O33367JKmrq0ubN2/W7t27ZbfbFRMTo%2BzsbI0dO1aS1NDQoKysLJWXl8vLy0vx8fFKT0%2BXt7f3VT2f%2BvrWq9reYvGSzeanpia7ywuSe0thKLrcndwvt5/j2mHOBx9zPrjGjRv4GSiTPDZgDTcELOD/IWB5DuZ88DHng8tdAYuvMAAAABhGwAIAADCMgAUAAGAYAQsAAMAwj7xNA4Dhbah9OeNyF%2BUDwOVwBAsAAMAwAhYAAIBhBCwAAADDuAYLAL4F14wBuFIcwQIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwzOMDVk9Pjx555BH967/%2Bq7Ps8OHDWrx4saKiohQbG6vi4mKXbXbu3Km4uDhNmzZNCQkJOnTokEt769ev16xZsxQVFaXly5errq7OWd/Y2KiUlBRNnz5dMTExysnJUXd397V/ogAAYNjw%2BIC1detW/eUvf3Eut7S0aNmyZYqPj1dlZaVycnK0du1aHTlyRJJUXl6u7OxsrVu3TpWVlVq4cKGWL1%2Bujo4OSVJ%2Bfr7279%2BvHTt2qLS0VD4%2BPsrMzHS2n5aWJl9fX5WWlmr79u0qKytTQUHBoD5nAAAwtHl0wCorK9PevXv1k5/8xFm2d%2B9eBQUFKTk5WRaLRTNnztSCBQtUWFgoSSouLtb8%2BfMVHR2tkSNHasmSJbLZbCopKXHWL126VBMmTJC/v7/WrFmjffv26fTp0zp58qQqKiqUkZEhq9WqiRMnKiUlxdk2AADAQHhswGpsbNSaNWv06quvymq1Osurq6sVGRnpsm54eLiqqqokSTU1NZetb21t1ZkzZ1zqQ0JCFBgYqGPHjqm6ulpBQUEKCwtz1k%2BZMkW1tbU6f/78tXiaAABgGLK4ewCX0tvbq4yMDD322GO68cYbXersdrtL4JIkHx8ftbe3f2u93W6XJPn6%2Bvar76u7eNu%2B5fb2dgUEBAxo/HV1daqvr3cps1h8FRoaOqDtL8Xb28vlLwBcjsUy8PcJ3lsGH3N%2BffDIgPXrX/9ao0aN0iOPPNKvzmq1qrW11aWss7NTfn5%2BzvrOzs5%2B9TabzRmW%2Bq7Hunh7h8PRr65vua/9gSgqKtLWrVtdylJTU7VixYoBt3E5AQHWb18JwHXNZhv4%2B1Uf3lsGH3M%2BvHlkwNq1a5fq6uo0ffp0SXIGpg8%2B%2BECrVq3S/v37XdavqalRRESEJCkiIkLV1dX96ufMmaPAwECFhYW5nEasr69Xc3OzIiMj1dvbq%2BbmZjU0NCgkJESSdOLECY0fP15jxowZ8PiTkpIUGxvrUmax%2BKqpyX4Fs%2BDK29tLAQFWnT/foZ6e3u/cDoDh70rea3hvGXzM%2BeD6Lv9wmOCRAeu9995zWe67RcO6devU1NSkDRs2qKCgQMnJyTp48KD27NmjvLw8SVJiYqJSU1M1b948RUdHq7CwUI2NjYqLi5MkJSQkKD8/X7feeqtsNptyc3M1Y8YMTZo0SZIUHR2t3NxcZWVlqampSXl5eUpMTLyi8YeGhvY7HVhf36ru7qt/IfX09BppB8Dw9V3eI3hvGXzM%2BfDmkQHrm9hsNr3xxhvKycnRli1bFBwcrMzMTN11112SpJkzZ%2BrFF1/USy%2B9pLNnzyo8PFyvvfaagoKCJH19qq67u1vJycmy2%2B2KiYnRpk2bnO1v2bJFWVlZuvfee%2BXl5aX4%2BHilpKS44ZkCAIChaoTD4XC4exDXg/r61m9f6RtYLF6y2fzU1GR3%2BY9n3qb937AVgOvR79NmD3jdy7234NphzgfXuHEDv8THJL7CAAAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGCY8YDV09NjukkAAIAhxXjAmjNnjn75y1%2BqpqbGdNMAAABDgvGA9dRTT%2BmTTz7RAw88oMWLF2vbtm1qbb263%2BEDAAAYSowHrIcffljbtm3Te%2B%2B9p1mzZum1117T3XffrWeeeUYff/yx6e4AAAA8zjW7yH3y5MlauXKl3nvvPaWmpuqPf/yjnnjiCcXGxurNN9/kWi0AADBsWa5Vw4cPH9Z//dd/qaSkRBcuXFBcXJwSEhJ09uxZbd68WUePHtXGjRuvVfcAAABuYzxg5eXladeuXTp58qRuvfVWrVy5Ug888ID8/f2d63h7e%2BuFF14w3TUAAIBHMB6w3nnnHS1cuFCJiYkKDw%2B/5DpTpkxRenq66a4BAJLmbdrv7iEM2O/TZrt7CMA1YTxg7du3T21tbWpubnaWlZSUaObMmbLZbJKkm266STfddJPprgEAADyC8Yvc//a3v2nu3LkqKipylm3YsEELFizQ8ePHTXcHAADgcYwHrF/%2B8pf6yU9%2BopUrVzrLPvjgA82ZM0fr1q0z3R0AAIDHMR6wPvvsMy1btkyjRo1ylnl7e2vZsmX661//aro7AAAAj2M8YPn7%2B%2BvUqVP9ys%2BcOSMfHx/T3QEAAHgc4wFr7ty5eumll/Txxx%2Brra1NdrtdBw4cUFZWluLi4kx3BwAA4HGMf4vwmWee0enTp/X4449rxIgRzvK4uDitWrXKdHcAAAAex3jAslqt%2BvWvf60vvvhCx44d08iRIzVlyhRNnjzZdFcAAAAe6Zr9VM4PfvAD/eAHP7hWzQMAAHgs4wHriy%2B%2BUFZWlg4ePKiurq5%2B9Z9//rnpLgEAADyK8YD10ksvqba2Vunp6RozZozp5gEAADye8YB16NAh/e53v1NUVJTppgEAAIYE47dpsNls8vPzM90sAADAkGE8YD3yyCPauHGjWltbTTcNAAAwJBg/RfjRRx/pr3/9q2JiYjR27FiXn8yRpD/%2B8Y%2BmuwQAAPAoxgNWTEyMYmJiTDcLAAAwZBgPWE899ZTpJgEAAIYU49dgSVJVVZWeffZZ/dM//ZPOnj2rwsJClZeXX4uuAAAAPI7xgPXpp59q8eLF%2BvLLL/Xpp5/qwoUL%2Bvzzz/X444/rww8/NN0dAACAxzEesF555RU9/vjjevvttzVy5EhJ0ssvv6yf//zn2rp1q%2BnuAAAAPM41OYIVHx/fr/zhhx/W//7v/5ruDgAAwOMYD1gjR45UW1tbv/La2lpZrVbT3QEAAHgc4wHrvvvu06uvvqqmpiZn2YkTJ5STk6N/%2BId/MN0dAACAxzEesFavXq3Ozk7NmjVLHR0dSkhI0AMPPCCLxaJVq1aZ7g4AAMDjGL8Plr%2B/v7Zt26aysjL97W9/U29vryIjI/WjH/1IXl7X5K4QAAAAHuWaJZ6ZM2fqiSee0NKlS/XjH//4isNVWVmZFi9erDvuuEOzZ89Wdna2Ojs7JUmHDx/W4sWLFRUVpdjYWBUXF7tsu3PnTsXFxWnatGlKSEjQoUOHnHU9PT1av369Zs2apaioKC1fvlx1dXXO%2BsbGRqWkpGj69OmKiYlRTk6Ouru7r2ImAADA9cZ4wIqNjdW999572cdAnDt3Tv/8z/%2Bshx9%2BWH/5y1%2B0c%2BdOVVRU6De/%2BY1aWlq0bNkyxcfHq7KyUjk5OVq7dq2OHDkiSSovL1d2drbWrVunyspKLVy4UMuXL1dHR4ckKT8/X/v379eOHTtUWloqHx8fZWZmOvtOS0uTr6%2BvSktLtX37dpWVlamgoMD0NAEAgGHM%2BCnCBx98UCNGjHAud3V16eTJk9q3b5/S0tIG1EZwcLA%2B/vhj%2Bfv7y%2BFwqLm5WV999ZWCg4O1d%2B9eBQUFKTk5WdLXR8oWLFigwsJC3XbbbSouLtb8%2BfMVHR0tSVqyZImKiopUUlKiRYsWqbi4WOnp6ZowYYIkac2aNbr77rt1%2BvRp9fb2qqKiQvv27ZPVatXEiROVkpKiDRs26MknnzQ7UQAAYNgyHrCefvrpS5a/8847OnjwoH7%2B858PqB1/f39J0o9//GOdPXtW06dPV0JCgjZt2qTIyEiXdcPDw7V9%2B3ZJUk1NjRYtWtSvvqqqSq2trTpz5ozL9iEhIQoMDNSxY8ckSUFBQQoLC3PWT5kyRbW1tTp//rwCAgIGNHYAwMDM27Tf3UO4Ir9Pm%2B3uIWCIMB6wLueee%2B7Rxo0br3i7vXv3qqWlRenp6VqxYoXCwsL63U/Lx8dH7e3tkiS73X7ZervdLkny9fXtV99Xd/G2fcvt7e0DDlh1dXWqr693KbNYfBUaGjqg7S/F29vL5S8AYPBZLFf/Hsz7%2BfVh0AJWRUWFRo8efcXb%2Bfj4yMfHRxkZGVq8eLEeeeQRtba2uqzT2dkpPz8/SV8Hor6L4f%2B%2B3mazOcNS3/VYF2/vcDj61fUt97U/EEVFRf1%2BFig1NVUrVqwYcBuXExDAzVoBwF1stoF/Fnwb3s%2BHN%2BMB6%2BJTgA6HQ21tbTp27NiATw9%2B8skneu6557R7926NGjVKknThwgWNHDlS4eHh2r/f9ZByTU2NIiIiJEkRERGqrq7uVz9nzhwFBgYqLCxMNTU1ztOE9fX1am5uVmRkpHp7e9Xc3KyGhgaFhIRI%2BvomqePHj9eYMWMGPAdJSUmKjY11KbNYfNXUZB9wGxfz9vZSQIBV5893qKen9zu3AwD47q7mfbwP7%2BeDy2QovhLGA9b3vvc9l4vcpa9/PufRRx/VggULBtTG1KlT1dnZqVdffVXPPPOM6uvrtX79eiUmJmru3Ll69dVXVVBQoOTkZB08eFB79uxRXl6eJCkxMVGpqamaN2%2BeoqOjVVhYqMbGRsXFxUmSEhISlJ%2Bfr1tvvVU2m025ubmaMWOGJk2aJEmKjo5Wbm6usrKy1NTUpLy8PCUmJl7RHISGhvY7HVhf36ru7qt/IfX09BppBwBw5Uy%2B//J%2BPryNcDgcDncP4lJqamqUm5uro0ePasyYMVqwYIFSU1M1atQoHT16VDk5OTp%2B/LiCg4OVkpKihIQE57a7du1Sfn6%2Bzp49q/DwcGVmZur222%2BX9PW3Gjdv3qzdu3fLbrcrJiZG2dnZGjt2rCSpoaFBWVlZKi8vl5eXl%2BLj45Weni5vb%2B%2Brej719a3fvtI3sFi8ZLP5qanJ7vKCHGoXiALAUGbiIvfLvZ/j2hg3buBnoEwyHrAqKysHvO6dd95psmuPRsACgKGPgDX0uCtgGT9FuGTJEjkcDuejT99pw76yESNG6PPPPzfdPQAAgNsZD1j/9m//prVr12r16tW66667NHLkSB0%2BfFgvvfSSfvazn%2Bmee%2B4x3SUAAIBHMX4TjvXr1%2BvFF1/UfffdJ39/f40ePVozZsxQVlaW3njjDX3/%2B993PgAAAIYj4wGrrq7O%2BTM0f8/f319NTU2muwMAAPA4xgPWtGnTtHHjRrW1tTnLmpubtWHDBs2cOdN0dwAAAB7H%2BDVYmZmZevTRRzVnzhxNnjxZkvTFF19o3Lhxeuutt0x3BwAA4HGMB6wpU6aopKREe/bs0YkTJyRJP/vZzzR//vx%2Bv/MHAAAwHF2T3yIMCAjQ4sWL9eWXX2rixImSvr6bOwAAwPXA%2BDVYDodDr7zyiu6880498MADOnPmjFavXq1nn31WXV1dprsDAADwOMYD1ttvv61du3bpxRdfdP5Q83333af/%2BZ//0ebNm013BwAA4HGMB6yioiK98MILSkhIcN69/f7771dOTo7%2B%2B7//23R3AAAAHsd4wPryyy/1wx/%2BsF/51KlT1dDQYLo7AAAAj2M8YH3/%2B9/XkSNH%2BpV/9NFHzgveAQAAhjPj3yJ84okn9Itf/EJnz56Vw%2BFQWVmZtm3bprffflvPPvus6e4AAAA8jvGAtWjRInV3dys/P1%2BdnZ164YUXNHbsWK1cuVIPP/yw6e4AAAA8jvGAtXv3bv3jP/6jkpKSdO7cOTkcDo0dO9Z0NwAAAB7L%2BDVYL7/8svNi9uDgYMIVAAC47hgPWJMnT9axY8dMNwsAADBkGD9FGBERofT0dL3%2B%2BuuaPHmyRo8e7VK/du1a010CAAB4FOMB69SpU4qOjpYk1dfXm24eAADA4xkJWGvXrtW//Mu/yNfXV2%2B//baJJgEAAIYsI9dgvfXWW%2Bro6HApe%2BKJJ1RXV2eieQAAgCHFSMByOBz9yj755BN99dVXJpoHAAAYUox/ixAAAOB6R8ACAAAwzFjAGjFihKmmAAAAhjRjt2l4%2BeWXXe551dXVpQ0bNsjPz89lPe6DBQAAhjsjAevOO%2B/sd8%2BrqKgoNTU1qampyUQXAAAAQ4aRgMW9rwAAAP4fLnIHAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw4zdBwsAgOFu3qb97h7CFfl92mx3D%2BG6xREsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDCPDVhVVVV67LHHNGPGDM2ePVurVq3SuXPnJEmHDx/W4sWLFRUVpdjYWBUXF7tsu3PnTsXFxWnatGlKSEjQoUOHnHU9PT1av369Zs2apaioKC1fvlx1dXXO%2BsbGRqWkpGj69OmKiYlRTk6Ouru7B%2BdJAwCAYcEjA1ZnZ6eefPJJRUVF6c9//rPeffddNTc367nnnlNLS4uWLVum%2BPh4VVZWKicnR2vXrtWRI0ckSeXl5crOzta6detUWVmphQsXavny5ero6JAk5efna//%2B/dqxY4dKS0vl4%2BOjzMxMZ99paWny9fVVaWmptm/frrKyMhUUFLhjGgAAwBDlkQGrtrZWN954o1JTUzVq1CjZbDYlJSWpsrJSe/fuVVBQkJKTk2WxWDRz5kwtWLBAhYWFkqTi4mLNnz9f0dHRGjlypJYsWSKbzaaSkhJn/dKlSzVhwgT5%2B/trzZo12rdvn06fPq2TJ0%2BqoqJCGRkZslqtmjhxolJSUpxtAwAADIRHBqwbbrhBr7/%2Bury9vZ1lf/jDH3TzzTerurpakZGRLuuHh4erqqpKklRTU3PZ%2BtbWVp05c8alPiQkRIGBgTp27Jiqq6sVFBSksLAwZ/2UKVNUW1ur8%2BfPX4unCgAAhiGP/y1Ch8OhTZs26cMPP9Q777yjt956S1ar1WUdHx8ftbe3S5Lsdvtl6%2B12uyTJ19e3X31f3cXb9i23t7crICBgQGOuq6tTfX29S5nF4qvQ0NABbX8p3t5eLn8BAPg2FgufGe7i0QGrra1Nzz77rD777DO98847mjp1qqxWq1pbW13W6%2BzslJ%2Bfn6SvA1FnZ2e/epvN5gxLfddjXby9w%2BHoV9e33Nf%2BQBQVFWnr1q0uZampqVqxYsWA27icgADrt68EAIAkm23gn10wy2MD1qlTp7R06VJ973vf0/bt2xUcHCxJioyM1P79rr9mXlNTo4iICElSRESEqqur%2B9XPmTNHgYGBCgsLczmNWF9fr%2BbmZkVGRqq3t1fNzc1qaGhQSEiIJOnEiRMaP368xowZM%2BCxJyUlKTY21qXMYvFVU5P9yibh73h7eykgwKrz5zvU09P7ndsBAFw/pq95z91DGLD30390Tdp1V8j0yGOHLS0tevTRR3XHHXfot7/9rTNcSVJcXJwaGhpUUFCgrq4uHThwQHv27NGiRYskSYmJidqzZ48OHDigrq4uFRQUqLGxUXFxcZKkhIQE5efn6/Tp02pra1Nubq5mzJihSZMmafLkyYqOjlZubq7a2tp0%2BvRp5eXlKTEx8YrGHxoaqptvvtnlERwcou7u3u/86AtVPT2u5QAADAdX8xn5TQ93GeFwOBxu6/0y3nzzTa1bt05Wq1UjRoxwqTt06JCOHj2qnJwcHT9%2BXMHBwUpJSVFCQoJznV27dik/P19nz55VeHi4MjMzdfvtt0uSurq6tHnzZu3evVt2u10xMTHKzs7W2LFjJUkNDQ3KyspSeXm5vLy8FB8fr/T0dJcL7r%2BL%2BvrWb1/pG1gsXrLZ/NTUZHfZYeZt2v8NWwEAMDT8Pm32NWl33LiBn4EyySMD1nBEwAIA4PKGW8DyyFOEAAAAQxkBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMI8PWOfOnVNcXJzKy8udZYcPH9bixYsVFRWl2NhYFRcXu2yzc%2BdOxcXFadq0aUpISNChQ4ecdT09PVq/fr1mzZqlqKgoLV%2B%2BXHV1dc76xsZGpaSkaPr06YqJiVFOTo66u7uv/RMFAADDhkcHrIMHDyopKUmnTp1ylrW0tGjZsmWKj49XZWWlcnJytHbtWh05ckSSVF5eruzsbK1bt06VlZVauHChli9fro6ODklSfn6%2B9u/frx07dqi0tFQ%2BPj7KzMx0tp%2BWliZfX1%2BVlpZq%2B/btKisrU0FBwaA%2BbwAAMLR5bMDauXOn0tPTtXLlSpfyvXv3KigoSMnJybJYLJo5c6YWLFigwsJCSVJxcbHmz5%2Bv6OhojRw5UkuWLJHNZlNJSYmzfunSpZowYYL8/f21Zs0a7du3T6dPn9bJkydVUVGhjIwMWa1WTZw4USkpKc62AQAABsJjA9bdd9%2Bt999/X/fff79LeXV1tSIjI13KwsPDVVVVJUmqqam5bH1ra6vOnDnjUh8SEqLAwEAdO3ZM1dXVCgoKUlhYmLN%2BypQpqq2t1fnz500/RQAAMExZ3D2Ayxk3btwly%2B12u6xWq0uZj4%2BP2tvbv7XebrdLknx9ffvV99VdvG3fcnt7uwICAgY09rq6OtXX17uUWSy%2BCg0NHdD2l%2BLt7eXyFwCA4cRiGV6fbx4bsC7HarWqtbXVpayzs1N%2Bfn7O%2Bs7Ozn71NpvNGZb6rse6eHuHw9Gvrm%2B5r/2BKCoq0tatW13KUlNTtWLFigG3cTkBAdZvXwkAgCHGZhv45%2BxQMOQCVmRkpPbv3%2B9SVlNTo4iICElSRESEqqur%2B9XPmTNHgYGBCgsLczmNWF9fr%2BbmZkVGRqq3t1fNzc1qaGhQSEiIJOnEiRMaP368xowZM%2BAxJiUlKTY21qXMYvFVU5P9ip9vH29vLwUEWHX%2BfId6enq/czsAAHiiq/mM/CbuCm5D7nhcXFycGhoaVFBQoK6uLh04cEB79uzRokWLJEmJiYnas2ePDhw4oK6uLhUUFKixsVFxcXGSpISEBOXn5%2Bv06dNqa2tTbm6uZsyYoUmTJmny5MmKjo5Wbm6u2tradPr0aeXl5SkxMfGKxhgaGqqbb77Z5REcHKLu7t7v/OgLVT09ruUAAAwHV/MZ%2BU0PdxlyR7BsNpveeOMN5eTkaMuWLQoODlZmZqbuuusuSdLMmTP14osv6qWXXtLZs2cVHh6u1157TUFBQZK%2BPlXX3d2t5ORk2e12xcTEaNOmTc72t2zZoqysLN17773y8vJSfHy8UlJS3PBMAQDAUDXC4XA43D2I60F9feu3r/QNLBYv2Wx%2BamqyuyTyeZv2f8NWAAAMDb9Pm31N2h03buCX%2BJg05E4RAgAAeDoCFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQSsS2hsbFRKSoqmT5%2BumJgY5eTkqLu7293DAgAAQwQB6xLS0tLk6%2Bur0tJSbd%2B%2BXWVlZSooKHD3sAAAwBBBwLrIyZMnVVFRoYyMDFmtVk2cOFEpKSkqLCx099AAAMAQQcC6SHV1tYKCghQWFuYsmzJlimpra3X%2B/Hk3jgwAAAwVFncPwNPY7XZZrVaXsr7l9vZ2BQQEfGsbdXV1qq%2BvdymzWHwVGhr6ncfl7e3l8hcAgOHEYhlen28ErIv4%2Bvqqo6PDpaxv2c/Pb0BtFBUVaevWrS5lTz31lJ5%2B%2BunvPK66ujr97nevKykpySWo/SXnH79zm/hmdXV1Kioq6jfnuHaY88HHnA8%2B5vz6MLziogERERFqbm5WQ0ODs%2BzEiRMaP368xowZM6A2kpKS9J//%2BZ8uj6SkpKsaV319vbZu3drvyBiuHeZ88DHng485H3zM%2BfWBI1gXmTx5sqKjo5Wbm6usrCw1NTUpLy9PiYmJA24jNDSU/0oAALiOcQTrErZs2aLu7m7de%2B%2B9euihh/SjH/1IKSkp7h4WAAAYIjiCdQkhISHasmWLu4cBAACGKI5gDRHjxo3TU089pXHjxrl7KNcN5nzwMeeDjzkffMz59WGEw%2BFwuHsQAAAAwwlHsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbCGgMbGRqWkpGj69OmKiYlRTk6Ouru73T2sYa2kpEQ33XSToqKinI%2BMjAx3D2tYOnfunOLi4lReXu4sO3z4sBYvXqyoqCjFxsaquLjYjSMcfi415y%2B%2B%2BKJuueUWl32%2BqKjIjaMc%2BqqqqvTYY49pxowZmj17tlatWqVz585JYh%2B/HhCwhoC0tDT5%2BvqqtLRU27dvV1lZmQoKCtw9rGHt6NGj%2BulPf6pDhw45Hxs2bHD3sIadgwcPKikpSadOnXKWtbS0aNmyZYqPj1dlZaVycnK0du1aHTlyxI0jHT4uNefS1/t8dna2yz6flJTkplEOfZ2dnXryyScVFRWlP//5z3r33XfV3Nys5557jn38OkHA8nAnT55URUWFMjIyZLVaNXHiRKWkpKiwsNDdQxvWjh49qltuucXdwxjWdu7cqfT0dK1cudKlfO/evQoKClJycrIsFotmzpypBQsWsM8bcLk5v3Dhgo4fP84%2Bb1Btba1uvPFGpaamatSoUbLZbEpKSlJlZSX7%2BHWCgOXhqqurFRQUpLCwMGfZlClTVFtbq/Pnz7txZMNXb2%2BvPvvsM/3pT3/SPffcozlz5uj5559XS0uLu4c2rNx99916//33df/997uUV1dXKzIy0qUsPDxcVVVVgzm8Yelyc15VVaXu7m5t2bJFs2bN0ty5c/Wb3/xGvb29bhrp0HfDDTfo9ddfl7e3t7PsD3/4g26%2B%2BWb28esEAcvD2e12Wa1Wl7K%2B5fb2dncMadg7d%2B6cbrrpJs2dO1clJSXatm2b/u///o9rsAwbN26cLBZLv/JL7fM%2BPj7s7wZcbs5bW1s1Y8YMPfLII/roo4%2B0YcMGvf3223rjjTfcMMrhx%2BFw6Fe/%2BpU%2B/PBDrVmzhn38OtH/lQaP4uvrq46ODpeyvmU/Pz93DGnYCwkJcTlUb7ValZGRoYceekhtbW3y9/d34%2BiGP6vVqtbWVpeyzs5O9vdraPbs2Zo9e7Zz%2BbbbbtOjjz6qkpISPfnkk24c2dDX1tamZ599Vp999pneeecdTZ06lX38OsERLA8XERGh5uZmNTQ0OMtOnDih8ePHa8yYMW4c2fBVVVWlV155RQ6Hw1l24cIFeXl5adSoUW4c2fUhMjJS1dXVLmU1NTWKiIhw04iGvw8%2B%2BEDbtm1zKbtw4YJ8fHzcNKLh4dSpU1q0aJHa2tq0fft2TZ06VRL7%2BPWCgOXhJk%2BerOjoaOXm5qqtrU2nT59WXl6eEhMT3T20YSsoKEiFhYV6/fXX1d3drdraWm3YsEEPPvggAWsQxMXFqaGhQQUFBerq6tKBAwe0Z88eLVq0yN1DG7YcDofWrl2rsrIyORwOHTp0SG%2B99RbfIrwKLS0tevTRR3XHHXfot7/9rYKDg5117OPXhxGOv/83HR6poaFBWVlZKi8vl5eXl%2BLj45Wenu5y8STMqqio0MaNG3X8%2BHGNHj1a8%2BfPV0ZGhkaPHu3uoQ1LU6dO1VtvvaWYmBhJX3%2BLMycnR8ePH1dwcLBSUlKUkJDg5lEOLxfP%2BbZt2/Tmm2/q7NmzCgkJ0WOPPabk5GQ3j3LoevPNN7Vu3TpZrVaNGDHCpe7QoUPs49cBAhYAAIBhnCIEAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAz7/wC4DxGTesqObQAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-262188564383134968">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">9.0</td>
        <td class="number">6263</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7.0</td>
        <td class="number">6138</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.25</td>
        <td class="number">6133</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4.5</td>
        <td class="number">6050</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5.75</td>
        <td class="number">6033</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8.5</td>
        <td class="number">6020</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6.5</td>
        <td class="number">5963</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6.0</td>
        <td class="number">5930</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.5</td>
        <td class="number">5911</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9.25</td>
        <td class="number">5897</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (85)</td>
        <td class="number">357688</td>
        <td class="number">85.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-262188564383134968">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.25</td>
        <td class="number">4796</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:86%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">0.5</td>
        <td class="number">5002</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:89%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">0.75</td>
        <td class="number">5184</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:93%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">4996</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:89%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.25</td>
        <td class="number">5575</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">22.75</td>
        <td class="number">2051</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">23.0</td>
        <td class="number">1474</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:72%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">23.25</td>
        <td class="number">1920</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:93%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">23.5</td>
        <td class="number">2042</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:99%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">23.75</td>
        <td class="number">1969</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:96%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_watched">watched<br/>
            <small>Boolean</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr class="">
                    <th>Distinct count</th>
                    <td>2</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
            </table>
        </div>
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Mean</th>
                    <td>0.054547</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minifreqtable7691840147767586107">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 94.5%">
            395224
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:6%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 5.5%">
            &nbsp;
        </div>
        22802
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable7691840147767586107, #minifreqtable7691840147767586107"
        aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable7691840147767586107">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">395224</td>
        <td class="number">94.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">22802</td>
        <td class="number">5.5%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_test">test<br/>
            <small>Boolean</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr class="">
                    <th>Distinct count</th>
                    <td>2</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
            </table>
        </div>
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Mean</th>
                    <td>0.48879</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minifreqtable3312842657340778298">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 51.1%">
            213699
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:95%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 48.9%">
            204327
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable3312842657340778298, #minifreqtable3312842657340778298"
        aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable3312842657340778298">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">213699</td>
        <td class="number">51.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">204327</td>
        <td class="number">48.9%</td>
        <td>
            <div class="bar" style="width:95%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div>
    <div class="row headerrow highlight">
        <h1>Correlations</h1>
    </div>
    <div class="row variablerow">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAqcAAAJcCAYAAADEsA3eAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAABsI0lEQVR4nO3de3zP9f//8ft7zDbHzWFOEY2RlG0Oc44xkmPIyjmnkE3nVPogROVD5dDBMSEkRJQKSWUjp5yPYU4bRuxkm71/f/h6/3p/tnjzfvF6v7fb9XJ5X9jzddj9/fRue/R4nSxWq9UqAAAAwAV4mB0AAAAAuIHiFAAAAC6D4hQAAAAug%2BIUAAAALoPiFAAAAC6D4hQAAAAug%2BIUAAAALoPiFAAAAC6D4hQAAAAuI6/ZAQDgbjl58qSaNWv2r8s9PT1VsGBBVahQQU2aNFH37t1VsGDBe5gQAPC/LDy%2BFEBO9c/iNDAwMEvhmZ6eroSEBJ06dUqSVKZMGc2ZM0f333//Pc8KALiO4hRAjvXP4nTu3LkKDQ3Ndr2YmBgNHjxYiYmJCg4O1sKFC%2B9lTADAP3DOKYBcLzQ0VC%2B%2B%2BKIkafv27dq9e7fJiQAg96I4BQBJ4eHhtr/v3LnTxCQAkLtxQRQASCpUqJDt70lJSXbLtmzZoi%2B%2B%2BELbtm3TpUuXVLhwYQUFBalHjx6qV69etvu7fPmyFi5cqA0bNujw4cNKTEyUj4%2BPypcvr6ZNm6pnz54qUqSI3TZVqlSRJP32228aP3681q5dKw8PDz300EOaNWuW8ubNq507d2rOnDnau3evzpw5Iy8vL1WsWFHNmzdX165ds72gKzU1VQsXLtTq1at1%2BPBhpaenq2TJkqpfv7769OmjChUq2K0fExOjnj17qkaNGpo/f76%2B%2BOILLV%2B%2BXMePH5enp6ceeugh9ejRQ82bN7%2BTqQaAm6I4BQBJx48ft/29VKlStr9PmDBB06dPlyQVKVJEgYGBio%2BP19q1a7V27Vr169dPr7zyit2%2Bjh07pt69e%2BvMmTPKmzevypcvr7Jly%2BrUqVPas2eP9uzZo1WrVunrr79WgQIFsmSJjIzU9u3bFRgYqISEBJUoUUJ58%2BbVDz/8oBdeeEEZGRny8/NTpUqVlJSUpD///FM7d%2B7UihUrtHDhQrsC9ezZs3rmmWd09OhRSVKFChVUoEABHTlyRIsWLdLy5cs1fvx4Pf7441lypKenq3///tq0aZP8/PwUEBCgv/76S9HR0YqOjtbIkSP19NNPOzfxAPC/rACQQ8XGxloDAwOtgYGB1ujo6Juu%2B%2Bqrr1oDAwOtDz30kPXcuXNWq9Vq/fLLL62BgYHWWrVqWb/55hvbupmZmdZVq1ZZg4KCrIGBgdbFixfb7at79%2B7WwMBAa5cuXaxxcXF22y1btsxatWpVa2BgoHXevHl2293IWr16devmzZutVqvVeu3aNevFixet165dszZo0MAaGBhonT59ujUjI8O23e7du61169a1BgYGWj/99FPbeEZGhrV9%2B/bWwMBAa8uWLa379u2zLbty5Yr1zTfftL3nHTt22JZFR0fbsgQFBVlXrFhhW3b58mVrr169rIGBgdY6depY09PTbzqvAHC7OOcUQK6VmpqqvXv3asSIEVq%2BfLkkqXfv3ipevLjS0tI0efJkSdI777yjdu3a2bazWCx6/PHHbR3TyZMnKyMjQ5J04cIFHTp0SJI0evRo%2Bfv7223XoUMH1alTR5J04MCBbHO1atVKtWvXliR5eHjI19dXCQkJOnfunCSpS5cuypMnj239hx56SC%2B88IKaN28uX19f2/j333%2Bvffv2ycvLS9OnT1fVqlVtywoWLKgxY8aoUaNGSk9P16RJk7LNEhUVpbZt29q%2BLlSokO19X7p0SX/99Ve22wHAneKwPoBcoWfPnrdc58knn9TQoUMlXb9q//z58ypQoMC/3si/Xbt2Gj16tOLi4rR371498sgjKlasmKKjo5Wamipvb%2B8s21y7ds122D01NTXb/dasWTPLmJ%2Bfn4oUKaK///5bL7/8sgYNGqQaNWrIw%2BN6j6FLly7q0qWL3Tbr1q2TJIWFhalcuXLZfq9nnnlGGzdu1ObNm3XlyhW7c28lqWnTplm2CQgIsP398uXL2e4XAO4UxSmAXOF/b8JvsVjk5eUlX19fValSRc2bN1elSpVsy290P9PT09WtW7d/3W%2BePHmUmZmpo0eP6pFHHrGNe3t768yZM9q5c6dOnDih2NhYHTlyRPv27VNycrIkKTMzM9t9lihRItvv8/LLL%2Butt97Shg0btGHDBhUpUkShoaFq0KCBmjRpYneurCRbV/Ohhx761/w3ll27dk3Hjx9X9erV7ZaXLFkyyzb/LLqvXbv2r/sGgDtBcQogVxg%2BfPi/3oQ/O1euXJEkpaWladu2bbdc/58dxKNHj%2Bq9997Thg0b7ArQggULqlatWoqPj9f%2B/fv/dV/ZdVyl693R%2B%2B%2B/X7Nnz9bvv/%2Buv//%2BWz/88IN%2B%2BOEHWSwWNWnSRCNHjrQVqYmJiZKUpRv6T/8s2P/3LgXS9Ue83oyV57gAMBjFKQBkw8fHR9L1zuLSpUsd3u7ChQvq3r27Lly4oDJlyqhLly6qVq2aHnjgAd13332yWCx66aWXblqc3kxoaKhCQ0OVmpqqP/74Q1u2bNHGjRu1Z88erV%2B/XmfOnNHy5ctlsVhsdwK4UWhn559FdXZ3DgCAe43iFACyUbFiRUnXbwuVkZGhvHmz/ri0Wq2KiYlRqVKlVKZMGeXLl09ff/21Lly4IF9fX3399dcqWrRolu3i4uJuO09aWppiY2OVmJioGjVqyNvbWw0bNlTDhg31wgsvaNWqVXrxxRe1f/9%2BHThwQFWrVtUDDzygvXv3as%2BePf%2B63127dkm6fppD%2BfLlbzsXABiNq/UBIBu1a9dWoUKFlJSU9K%2Bd05UrV6pXr15q1aqVzp49K0k6efKkJKlMmTLZFqaHDx/Wjh07JN3e%2BZq//PKLHn/8cQ0YMEBpaWlZltevX9/29xv7vXEx07p16xQbG5vtfufOnStJCgoKUuHChR3OAwB3C8UpAGQjf/78GjBggCRp7Nix%2Bvrrr%2B3OH/3pp580YsQISddv/XSj6/jAAw9Ikvbv3681a9bY1rdarfrll1/Ur18/paenS5JSUlIcztO4cWP5%2Bfnp0qVLeu2113Tp0iXbsqSkJL377ruSpNKlS6ty5cqSpMcee0xVqlTR1atX1b9/f7tTCRITE/XWW2/p119/Vd68efXyyy87nAUA7iYO6wPAv%2Bjfv79iY2O1ePFivfHGG3r//fd13333KS4uTvHx8ZKkkJAQjRkzxrZN586dtWDBAh0/flxRUVEqW7as/Pz8dObMGV24cEGenp6qU6eONm/efFuH9/Ply6cPP/xQffv21erVq7V27VqVL19eHh4eio2NVXJysnx8fDR%2B/Hjly5dPkpQ3b15NmzZN/fv319GjR9W%2BfXu7J0TduN3VqFGjVKtWLWMnDwDuEMUpAPwLi8Wi0aNHq2XLllq4cKF27Nhhu6l9UFCQ2rRpo4iICFsxKF2/%2Bn3JkiWaPn261q9fr5MnT%2Br8%2BfMqVaqUmjRpol69eil//vxq3ry59u/fr9OnT6tMmTIO5QkNDdVXX32l2bNna%2BvWrTp27Jjy5s2rUqVKqWHDhurTp0%2BWfd133336%2Buuv9eWXX%2Br777/XkSNHdPbsWZUuXVqNGjVSt27dVKFCBSOnDQCcYrFyHxAAAAC4CM45BQAAgMugOAUAAIDLoDgFAACAy6A4BQAAyCUSEhIUHh6umJiYf11nw4YNatu2rYKCgtSqVSutX7/ebvn06dPVuHFjBQUFqUePHjp69KihGSlOAQAAcoGtW7cqIiJCJ06c%2BNd1jh07psjISA0dOlR//PGHIiMj9fzzz9tufbds2TJ98cUXmjlzpmJiYvTQQw8pKipKRl5fT3EKAACQwy1btkwvv/yyXnjhhVuuV6tWLTVv3lx58%2BbV448/rtq1a2vRokWSpMWLF6tr166qXLmyvLy89NJLL%2Bn06dM37cTeLopTAAAAFxcfH689e/bYvW48DMQRDRs21I8//qjHH3/8pusdPnxYgYGBdmOVKlWyPWHuf5d7enqqQoUKdk%2BgcxY34YdzLBazEzimYkXp0CGpcmXpr7/MTnNz/3hEJgDgDpn5%2B%2BkufO9FH32kKVOm2I0NGTJEkZGRDm1fokQJh9ZLSkqSj4%2BP3Zi3t7eSk5MdWm4EilPkDr6%2BUp481/%2BEcSwWied4GIf5NB5zajzm1BQREREKCwuzG3O04LwdPj4%2BSk1NtRtLTU1VgQIFHFpuBIpTAAAAI3kYf9akv7%2B//P39Dd/v/woMDNSePXvsxg4fPqzq1atLkipXrqxDhw6padOmkqT09HQdO3Ysy6kAzuCcUwAAAEiS2rVrp82bN2v16tXKyMjQ6tWrtXnzZrVv316S1KlTJ82bN0/79%2B/X1atX9d///lfFixdXrVq1DMtA5xQAAMBId6FzejcFBwdr1KhRateunQICAjR16lRNmDBBb775psqWLavJkyerYsWKkqTOnTvrypUreu6555SQkKCHH35Yn376qTw9PQ3LY7EaeWMq5D7uckFUcLC0bZsUEiJt3252mptzpwuiOPfMWMyn8ZhT47nLnJr5%2B8nAQs0mPd34fbooOqcAAABGcrPOqauhOAUAADASxalTmD0AAAC4DDqnAAAARqJz6hSKUwAAACNRnDqF2QMAAIDLoHMKAABgJDqnTmH2AAAA4DLonAIAABiJzqlTKE4BAACMRHHqFGYPAAAALoPOKQAAgJHonDqF2QMAAIDLoHMKAABgJDqnTqE4BQAAMBLFqVOYPQAAALgMOqcAAABGonPqFGYPAAAALoPOKQAAgJHonDqF4hQAAMBIFKdOYfYAAADgMuicAgAAGInOqVOYPQAAALgMOqcAAABGonPqFIpTAAAAI1GcOoXZAwAAgMugcwoAAGAkOqdOYfYAAADgMuicAgAAGInOqVMoTgEAAIxEceoUZg8AAAAug87pXXb16lVdvHhRpUqVMjsKAAC4F%2BicOoXZu8u6du2q33///a7t/z//%2BY/%2B85//3LX9AwAA3Et0Tu%2Byixcv3tX9v/3223d1/wAA4DbROXUKs3cX9enTR6dPn9aIESP0%2BOOPa%2BLEiXbLn3zySc2YMeOW%2B4mLi1O/fv1Up04dNW7cWEOGDFF8fLwkadiwYRo2bJgkqV27dgoODra9qlevrmrVqiktLU1Wq1Vz585Vy5YtVatWLXXt2lW7d%2B82/k0DAJDbeXgY/8pF6JzeRbNmzVJYWJiGDBkiDw8PffDBB3r%2B%2Befl4eGhI0eOaN%2B%2Bffr4449vuZ%2BJEyeqVKlS%2Bvjjj3X16lVFRUXps88%2B0/Dhw%2B3WW7Fihe3vJ0%2Be1FNPPaVBgwYpX758mj9/vmbPnq2PP/5YAQEB%2Buabb/TMM8/ou%2B%2B%2BU/HixR16P/Hx8Tp37pzdWImKFeXv6%2BvQ9qaqWtX%2BTwAA4JIoTu%2BRxx57TGPHjlVMTIzq1aunpUuX6tFHH3WoMPTy8tKWLVu0atUq1atXTzNmzJDHTf4v6uLFi%2BrXr5/atGmjbt26SZLmz5%2BvZ599VlX/rzjr3LmzlixZohUrVqhPnz4OvYdFixZpypQpdmNDhg5V5NChDm3vEhYsMDtBzmOxmJ0gZ2E%2BjcecGo85vblc1uk0GsXpPeLt7a22bdtq%2BfLlqlOnjlasWKHRo0c7tO3w4cP16aefaubMmRo2bJiqVq2q4cOHq1atWlnWTU1N1aBBg1S5cmW9%2BuqrtvFTp07p3Xff1YQJE2xjGRkZql69usPvISIiQmFhYXZjJdq2lT7/3OF9mKZq1euFadeu0v79Zqe5ua1bzU7gOItFslrNTpFzMJ/GY06N5y5zSgHttihO76EuXbro6aefVnh4uCwWixo1auTQdnv37lVERIQiIyOVkJCgqVOnasiQIYqOjrZbLzMzUy%2B99JIyMzP1/vvv23VXS5UqpaioKLVu3do2duLECfnexiF5f39/%2Bfv72w/%2B9ZfD27uE/ful7dvNTgEAyMnonDqF2bvL8uXLpytXrkiSqlatqgceeEDvvPOOnnjiCeXJk8ehfXzyyScaPXq0EhMTVbhwYfn4%2BMjPzy/LeqNHj9bhw4f1ySefyNvb225Zly5d9PHHH%2BvIkSOSpI0bN6p169basmWLk%2B8QAADY4YIop9A5vcs6d%2B6sSZMmadeuXZowYYK6dOmiESNGqHPnzg7v4%2B2339aoUaPUrFkzpaWlqXr16vrwww/t1jl9%2BrQWLFigggULqmXLlsrIyLAtmz59unr37i2r1arBgwcrPj5eJUuW1H/%2B8x81a9bMsPcKAADgLIvV6g4njsBlucs5PcHB0rZtUkiI6x/Wz8w0O4Hj3OXcM3fBfBqPOTWeu8ypmb%2Bf6tUzfp%2BbNhm/TxeVu/rEAAAAcGkc1jfZmjVrbDfRz07NmjUdulE/AABwES54juiFCxf01ltvafPmzcqTJ4/atWun1157TXnz2peC/fr109b/uWtMcnKyIiIi9PbbbyszM1M1a9aU1WqV5R/d6d9%2B%2B0358%2Bc3JCvFqclatmypli1bmh0DAAAYxQWL0%2Beff14lS5bUxo0bdf78eQ0aNEhz5sxRv3797Nb734bYkiVLNGXKFA0ZMkSSdPjwYaWnp2vbtm3Kly/fXcnqerMHAAAAwxw/flybN2/WK6%2B8Ih8fH5UrV06DBw/W/Pnzb7rd0aNHNXr0aE2YMMF2K8ldu3apSpUqd60wleicAgAAGOsudE6zfYR4iRJZ7z%2BejUOHDsnX11clS5a0jQUEBOj06dO6fPmyChcunO12o0aNUocOHewe%2BrNr1y5dvXpVnTp10qlTpxQQEKCXXnpJISEhd/jOsqI4BQAAMNJdKE6zfYT4kCGKjIy85bZJSUny8fGxG7vxdXJycrbF6R9//KGdO3faPVlSuv7Ey0ceeURDhw5VkSJFNH/%2BfPXt21crVqxQuXLlbvdtZYviFAAAwMVl%2BwjxEiUc2jZ//vxKSUmxG7vxdYECBbLdZtGiRWrVqlWW7/G/F3H37dtXS5cu1YYNG9S9e3eH8twKxSkAAICR7kLnNNtHiDuocuXKunTpks6fP6/ixYtLko4cOaJSpUqpUKFCWdbPyMjQ2rVrNXXq1CzLJk2apJYtW6patWq2sbS0NHl5ed1RtuxwQRQAAEAOVqFCBdWsWVPvvPOOEhMTFRsbq2nTpv3r0yoPHDigq1evZnse6cGDBzV27FidO3dOaWlpmjJlihITExUeHm5YXopTAAAAI3l4GP9y0kcffaSMjAw1a9ZMXbp0UaNGjTR48GBJUnBwsFasWGFbNzY2VkWKFMm2Gzpu3DiVL19e7du3V2hoqDZv3qzZs2fL19fX6Yw38PhSOIfHlxqPx5fmXsyn8ZhT47nLnJr5%2B%2Blu3L98zRrj9%2Bmi6JwCAADAZXBBFAAAgJFc8AlR7oTZAwAAgMugcwoAAGAkOqdOoTgFAAAwEsWpU5g9AAAAuAw6pwAAAEaic%2BoUZg8AAAAug84pAACAkeicOoXiFAAAwEgUp05h9gAAAOAy6JwCAAAYic6pU5g9AAAAuAw6pwAAAEaic%2BoUilMAAAAjUZw6hdkDAACAy6BzCgAAYCQ6p05h9gAAAOAy6JwCAAAYic6pUyhOAQAAjERx6hRmDwAAAC6DzikAAICR6Jw6hdkDAACAy6BzCgAAYCQ6p06hOAUAADASxalTmD0AAAC4DDqncE5mptkJbs/WrWYnuDV3%2BT/u4GBp2zapZk1p%2B3az0/w7d/uMugGrLGZHcJhF7pXXHbjLnJqa0F1%2BjrsoZg8AAAAug84pAACAkeicOoXiFAAAwEgUp05h9gAAAOAy6JwCAAAYic6pU5g9AAAAuAw6pwAAAEaic%2BoUilMAAAAjUZw6hdkDAACAy6BzCgAAYCQ6p06hOAUAADASxalTmD0AAAC4DDqnAAAARqJz6hRmDwAAAC6DzikAAICR6Jw6heIUAADASBSnTmH2AAAAcrgLFy5o8ODBqlWrlkJDQzV27FhlZGRku26/fv308MMPKzg42Pb65ZdfbMunT5%2Buxo0bKygoSD169NDRo0cNzUpxCgAAYCQPD%2BNfTnr%2B%2BeeVP39%2Bbdy4UUuWLNGmTZs0Z86cbNfdvXu3Zs6cqe3bt9tejRs3liQtW7ZMX3zxhWbOnKmYmBg99NBDioqKktVqdTrjDRSnAAAAOdjx48e1efNmvfLKK/Lx8VG5cuU0ePBgzZ8/P8u6sbGx%2Bvvvv1WtWrVs97V48WJ17dpVlStXlpeXl1566SWdPn1aMTExhuWlOAUAADDSXeicxsfHa8%2BePXav%2BPh4h%2BIcOnRIvr6%2BKlmypG0sICBAp0%2Bf1uXLl%2B3W3bVrlwoUKKAXXnhBdevWVZs2bbRkyRLb8sOHDyswMND2taenpypUqKD9%2B/c7OWn/HxdEAQAAGOkuXBC1aNEiTZkyxW5syJAhioyMvOW2SUlJ8vHxsRu78XVycrIKFy5sG09LS1NQUJBeeOEFVa5cWTExMYqMjFSBAgXUqlWrbPfl7e2t5OTkO31rWVCcAgAAuLiIiAiFhYXZjZUoUcKhbfPnz6%2BUlBS7sRtfFyhQwG68Q4cO6tChg%2B3rhg0bqkOHDvruu%2B/UqlUr%2Bfj4KDU11W6b1NTULPtxBsUpAACAke5C59Tf31/%2B/v53tG3lypV16dIlnT9/XsWLF5ckHTlyRKVKlVKhQoXs1l2yZImtS3pDWlqavLy8bPs6dOiQmjZtKklKT0/XsWPH7A71O4tzTgEAAHKwChUqqGbNmnrnnXeUmJio2NhYTZs2TZ07d86ybmJiokaPHq29e/cqMzNTP//8s7799ltFRERIkjp16qR58%2BZp//79unr1qv773/%2BqePHiqlWrlmF56ZwCAAAYyQVvwv/RRx/p7bffVrNmzeTh4aEOHTpo8ODBkqTg4GCNGjVK7dq1U69evZScnKwhQ4bowoULKleunN59911b8dm5c2dduXJFzz33nBISEvTwww/r008/laenp2FZLVYjb0yF3MedPj4Wi3vkdcEfatkKDpa2bZNCQqTt281O8%2B8yM81O4Dg3%2BYxaZTE7gsPcZErdirvMqcXMj%2Bm4ccbv8/XXjd%2Bni3KT34IAAADIDTisDwAAYCR3OQLmopg9AAAAuAw6pwAAAEaic%2BoUilMAAAAjUZw6hdkDAACAy6BzCgAAYCQ6p05h9gAAAOAy6JwCAAAYic6pUyhOAQAAjERx6hRmDwAAAC4j1xSnS5cuVVhY2G1tU6VKFcXExNylRM47efKkqlSpopMnT2a7/E7eMwAAcJKHh/GvXCR3vVsAAAC4tBxVnGbXSZw8ebJ69OghScrIyNCECRPUpEkThYSEaPjw4crIyJAkpaena9y4cQoNDVXdunU1Y8aM2/ree/bsUY8ePRQcHKyGDRvqww8/lNVqlSQtWbJEHTt2VGhoqIKDg/Xss88qISHBlm/w4MGKjIxUUFCQwsLCtGjRItt%2Bjxw5omeffVZNmjTRI488oscff1zr16%2B3%2B97Lly9X8%2BbNVb9%2BfQ0fPlyJiYk3zVi7dm21aNFCc%2BbMsWUEAAAGoXPqlFx1QVRcXJwKFy6sn376SceOHVPnzp1Vt25dtWnTRtOmTdPPP/%2BsJUuWqFixYho5cqTD%2B7106ZL69OmjHj16aObMmTp79qx69OihkiVLqlq1ahozZozmzp2rRx55RGfPnlWvXr00d%2B5cPf/885KktWvXatiwYZo4caJiYmI0cOBAlS9fXvXq1VNkZKSaNWumKVOmyGq1asKECRo5cqSaNm1q%2B/5//PGHFi9erMzMTA0ePFjvvPOO3nnnnSzvvVevXnrhhRc0a9YsHT9%2BXIMHD5a3t7eeeuoph95nfHy8zp07ZzdWonhx%2Bfv7OzxXcEBwsNkJHFO1qv2fAIDrclkxabRcVZwWLFhQ/fv3l8ViUaVKlVS1alWdOHFCkvTNN99o4MCBKleunCRp%2BPDhWrFihUP7Xb9%2Bvby8vPTcc8/JYrGofPnymj17tvLnzy9fX199%2B%2B23uu%2B%2B%2B/T3338rPj5eRYsWVVxcnG37KlWq6JlnnpEkNWzYUC1bttQ333yjevXq6dNPP1XJkiVltVp16tQpFS5c2G5bSRo2bJiKFi0qSYqKitKgQYM0ZswYu3VWrFihgIAAdevWTZJUqVIl9e3bV/PmzXO4OF20aJGmTJliNzbkuecUGRXl0PYuwWIxO8GtbdtmdoLbs2CB2QlyFjf4jLp%2BQntuMKVuhznF3ZSritMiRYrI8o//ojw9PXXt2jVJ17uCpUuXti0rXLiwihQp4tB%2Bz507p9KlS9vt%2B4EHHpAkpaWlae7cuVq5cqXy58%2BvKlWqKDEx0e5weoUKFez2V7p0ae3bt0%2BStH//fg0ePFjnzp1TQECAihYtmuVQ/H333We3bVpami5dumS3zqlTp7Rnzx7VqlXLNpaZmak8efI49B4lKSIiIssFViWKF5fc5dQAi8U9stasaXYCx1Ster0w7dpV2r/f7DT/butWsxM4zk0%2Bo1Y3Kk/dZErdirvMqakFNJ1Tp%2BSo4vRGoZWenm4bu3jxokPblipVSrGxsbavk5OTdeXKFYe3PXPmjKxWq61A/emnn5SYmKj4%2BHj99ttvWrlypYoXLy5JGjhwoN32/9sJPXnypEqXLq24uDgNHTpUU6ZMsRWFa9as0Q8//JBl%2B4IFC9q2zZ8/v62T%2Bs%2BMoaGhmjlzpm3s4sWLSkpKcug9SpK/v3/WQ/ju8BPK3WzfbnaC27N/v/tlBgC4rBxV2hcrVkxFihTRqlWrZLVatWfPHn3//fcObfvkk09qxowZOnLkiK5evarx48fbuqq30qRJE2VkZOiTTz5RWlqaTpw4oXfeeUdXr15VYmKi8ubNK09PT2VkZOibb77Rxo0b7QroHTt26JtvvtG1a9e0YcMGrV27Vp06dVJSUpKuXbsmHx8fSdLhw4c1depUSdc7sje8//77%2Bvvvv3X27Fl9%2BOGHioiIyJKxbdu22rFjh1asWKGMjAzFx8dr4MCBGj9%2BvEPvEQAAOIgLopySozqn%2BfLl0%2BjRo/XRRx9p5syZql69urp06aKtDhzW69%2B/v1JSUtS9e3dlZGSoS5cu8vX1dej7Fi5cWDNnztS4ceM0e/Zs%2Bfj4qFu3boqIiNClS5d08OBBNW3aVF5eXqpWrZq6du2q6Oho2/YPPvig1q5dqzFjxqh48eJ6//33Ffx/F8W8%2BuqreuWVV5SSkqJSpUqpS5cuev/993Xw4EFbvuDgYD322GPy8PBQmzZt9MILL2TJWLZsWc2YMUMTJkzQmDFjlCdPHjVp0kRvvvmmQ%2B8RAAA4KJcVk0azWLmXkKkmT56szZs364svvjA7yp1xp4%2BPu5wo5S4/1IKDr1%2B8FRLi2of1MzPNTuA4N/mMcs5p7uYuc2rqOaezZxu/z/%2B7cDo3yFGdUwAAANO5S5PBRVGcOiA0NNTuHM//tWrVKpUpU%2BYeJgIAAMiZKE4dEBMTc9f2HRkZedf2DQAATEDn1CkUpwAAAEaiOHUKswcAAACXQecUAADASHROncLsAQAAwGXQOQUAADASnVOnUJwCAAAYieLUKcweAAAAXAadUwAAACPROXUKxSkAAICRKE6dwuwBAADAZdA5BQAAMBKdU6cwewAAAHAZdE4BAACMROfUKRSnAAAARqI4dQqzBwAAAJdB5xQAAMBIdE6dwuwBAADAZdA5BQAAMBKdU6dQnAIAABiJ4tQpFKcAAAA53IULF/TWW29p8%2BbNypMnj9q1a6fXXntNefNmLQW//PJLzZkzR/Hx8fL391fPnj3VrVs3SVJmZqZq1qwpq9Uqi8Vi2%2Ba3335T/vz5DclKcQoAAGAkF%2BycPv/88ypZsqQ2btyo8%2BfPa9CgQZozZ4769etnt95PP/2kiRMnavr06apRo4Z27NihAQMGqHjx4mrZsqUOHz6s9PR0bdu2Tfny5bsrWV1v9gAAAGCY48ePa/PmzXrllVfk4%2BOjcuXKafDgwZo/f36WdePi4tS/f38FBQXJYrEoODhYoaGh2rJliyRp165dqlKlyl0rTCU6pwAAAMa6C53T%2BPh4nTt3zm6sRIkS8vf3v%2BW2hw4dkq%2Bvr0qWLGkbCwgI0OnTp3X58mUVLlzYNn7j8P0NFy5c0JYtW/T6669Lul6cXr16VZ06ddKpU6cUEBCgl156SSEhIc68PTsUpwAAAEa6C8XpokWLNGXKFLuxIUOGKDIy8pbbJiUlycfHx27sxtfJycl2xek/nTt3Ts8%2B%2B6yqV6%2BuNm3aSJK8vb31yCOPaOjQoSpSpIjmz5%2Bvvn37asWKFSpXrtydvLUsKE4BAABcXEREhMLCwuzGSpQo4dC2%2BfPnV0pKit3Yja8LFCiQ7TY7duzQ0KFDVatWLY0bN8524dSwYcPs1uvbt6%2BWLl2qDRs2qHv37g7luRWKUwAAACPdhc6pv7%2B/Q4fws1O5cmVdunRJ58%2BfV/HixSVJR44cUalSpVSoUKEs6y9ZskRjxoxRVFSU%2BvTpY7ds0qRJatmypapVq2YbS0tLk5eX1x1lyw4XRAEAAORgFSpUUM2aNfXOO%2B8oMTFRsbGxmjZtmjp37pxl3TVr1mjkyJGaPHlylsJUkg4ePKixY8fq3LlzSktL05QpU5SYmKjw8HDD8lKcAgAAGMnDw/iXkz766CNlZGSoWbNm6tKlixo1aqTBgwdLkoKDg7VixQpJ0pQpU3Tt2jVFRUUpODjY9vrPf/4jSRo3bpzKly%2Bv9u3bKzQ0VJs3b9bs2bPl6%2BvrdMYbLFar1WrY3pD7uNPHx2Jxj7wueH%2B8bAUHS9u2SSEh0vbtZqf5d5mZZidwnJt8Rq2y3HolF%2BEmU%2BpW3GVOLWZ%2BTDdtMn6f9eoZv08X5Sa/BQEAAJAbcEEUAACAkdzlCJiLYvYAAADgMuicAgAAGInOqVMoTgFX404X8EjS1q1mJ7g5d/klceMCs5o1XfsCM0nKdIOrYdyMRe40pxY3yWviFVHu8nPHRTF7AAAAcBl0TgEAAIxE59QpzB4AAABcBp1TAAAAI9E5dQrFKQAAgJEoTp3C7AEAAMBl0DkFAAAwEp1TpzB7AAAAcBl0TgEAAIxE59QpFKcAAABGojh1CrMHAAAAl0HnFAAAwEh0Tp3C7AEAAMBl0DkFAAAwEp1Tp1CcAgAAGIni1CnMHgAAAFwGnVMAAAAj0Tl1CsUpAACAkShOncLsAQAAwGXQOQUAADASnVOnMHsAAABwGXROAQAAjETn1CkUpwAAAEaiOHUKswcAAACXQecUAADASHROncLsAQAAwGXQOQUAADASnVOnUJwCAAAYieLUKcweAAAAXAadUwAAACPROXUKswcAAACXQecUAADASHROnUJxCgAAYCSKU6cwewAAAHAZdE4BAACMROfUKcweAAAAXAadUwAAACPROXXKbc3e1atXdfbsWYfWPXbs2J3kuaVr164pNjb2ruzbDFeuXFFCQoLZMQAAgFE8PIx/5SK39W67du2q33///Zbr7d27V23atHF4v2FhYVq6dKlD677wwgtavny5JOn06dMKDg7W6dOnHf5eriY8PFyHDh265XqTJ09Wjx49DPu%2BS5cuVVhYmGH7AwAAruvChQsaPHiwatWqpdDQUI0dO1YZGRnZrrthwwa1bdtWQUFBatWqldavX2%2B3fPr06WrcuLGCgoLUo0cPHT161NCst1WcXrx40aH1rly5ovT09DsKdDsZypQpo%2B3bt6tMmTJ35XvdC47OKQAAcBMu2Dl9/vnnlT9/fm3cuFFLlizRpk2bNGfOnCzrHTt2TJGRkRo6dKj%2B%2BOMPRUZG6vnnn1dcXJwkadmyZfriiy80c%2BZMxcTE6KGHHlJUVJSsVqvTGW9w%2BN326dNHp0%2Bf1ogRI/T222/rjz/%2BULdu3VSrVi2FhYXpgw8%2BUFpammJjY9W/f39JUnBwsLZv367ExEQNHz5cLVq0UFBQkBo1aqRPPvnktsO%2B%2Beab%2BuOPP/Tpp59q4MCBOnnypKpUqaKTJ09KkqpUqaJFixapZcuWqlGjhgYOHKjdu3frqaeeUnBwsDp16qTjx4/b9rdq1Sq1bdtWNWvWVMeOHfXrr786lCMqKkpjx461fT1s2DA1aNDA9g%2Bzfv16NW3aVJK0bds29ezZUw0bNtTDDz%2Bsjh07aseOHZKkli1bSpL69%2B%2Bv6dOnS5JWrlypNm3aKDg4WK1atdLq1att3ycpKUnDhw9Xw4YNFRoaqkmTJtmWpaWl6cMPP1SzZs1Up04d9e/f3%2B69HjlyRD169FBwcLDatm2rvXv3OjzvAADAfR0/flybN2/WK6%2B8Ih8fH5UrV06DBw/W/Pnzs6y7bNky1apVS82bN1fevHn1%2BOOPq3bt2lq0aJEkafHixeratasqV64sLy8vvfTSSzp9%2BrRiYmIMy%2BvwBVGzZs1SWFiYhgwZoqCgILVv314vv/yyZs%2BerTNnzigyMtJWhE6fPl09e/bU9u3bJUkjR47UyZMntWTJEhUqVEg//PCDoqKi1KpVK91///0Ohx07dqxOnDihOnXqKDIy0laU/tPKlSu1aNEipaWlqXXr1ho8eLBmz56t0qVLq2/fvvrkk080btw4bdiwQSNGjNDHH3%2BskJAQ/fLLL4qMjNTixYtVuXLlm%2BZo3ry5pk6dqjfffFOS9OuvvyoxMVEHDhxQ1apVtW7dOjVv3lypqakaNGiQoqKi9PTTTys1NVVvvPGG3nvvPS1YsEBr1qxRlSpVNH36dIWGhiomJkZvvPGGpkyZokaNGunXX3/V4MGDFRgYKOn66RK9evXS6NGjFRMTo969e6tJkyYKDg7WpEmTFB0drTlz5sjf31/Tp09Xnz59tHr1anl4eOjZZ59V48aNNWPGDJ04cUL9%2B/eXx23%2Bn1h8fLzOnTtnN1aieHH5%2B/vf1n6Aeyo42OwEjqla1f5PAO7rLpwjmu3v4BIlHPodfOjQIfn6%2BqpkyZK2sYCAAJ0%2BfVqXL19W4cKFbeOHDx%2B21R03VKpUSfv377ctv9GElCRPT09VqFBB%2B/fvV926de/ovf2vO7paf%2BXKlapSpYp69eolSbr//vv10ksvKSoqSm%2B88UaW9SMjI5UnTx4VLFhQZ8%2BelZeXl6TrE307xakjunfvLl9fX0lS5cqVVa1aNQUEBEiS6tatq61bt0qS5s2bp6efflq1a9eWJDVt2lRhYWFauHCh3nrrrZt%2BjyZNmuiNN95QbGyskpKS5O3trUceeUSbNm1SlSpVtH79ek2aNEmenp5atGiR7r//fl29elWnTp2Sr6%2Bvdu3ale1%2Bly9frhYtWujRRx%2BVJDVu3FgLFiywfZgqV66s9u3b295L8eLFdeLECQUFBWnhwoX66KOPVK5cOUnSc889p8WLF%2Bvnn3%2BWn5%2Bfzpw5o1dffVVeXl6qXLmynnnmGX3%2B%2Bee3NbeLFi3SlClT7MaGPPecIqOibms/prJYzE6Q87j6nG7bZnaC27NggdkJbsnF/8WzcPWP6HVuEfL/c49JNY31Lvx7Zvs7eMgQRUZG3nLbpKQk%2Bfj42I3d%2BDo5OdmuOM1uXW9vbyUnJzu03Ah3VJxeuHDBVgTdcN999yk1NVUXLlzIdv2xY8dq7969uu%2B%2B%2B1S9enVJUmZm5p18%2B5u6UZhKUp48eVSkSBHb1x4eHrZD76dOndLmzZv15Zdf2pZfu3bNoaq/cOHCqlOnjn755RclJyerfv36CggI0K%2B//qqQkBBZrVbVrFlTHh4eiomJUf/%2B/ZWcnKxKlSopb968/3peRnx8vKpVq2Y39sgjj2T73iQpX758unbtmhISEpScnKyhQ4fadUPT09N16tQppaWlyc/PT97e3rZl5cuXv%2BX7/F8RERFZLqIqUby4ZOB5JneVxeI%2BWd2FO8xpzZpmJ3BM1arXC9OuXaX/61C4KutW9yn43eEjKkkWuUHIG9xmUnNWAZ3t7%2BASJRzaNn/%2B/EpJSbEbu/F1gQIF7MZ9fHyUmppqN5aammpb71bLjXBHxWnZsmX1ww8/2I2dOHFC%2BfLlsysGbxg6dKjCwsI0c%2BZM5c2bVxcvXtTixYvvLPEtWBz8MJYqVUodOnTQgAEDbGOnT5%2B2K%2BBuplmzZvrll1%2BUnp6up556SgEBAfrggw%2B0Zs0aNWvWTB4eHtq5c6dGjx6thQsX2gryWbNm6a%2B//sp2n6VLl85y54FZs2YpKCjopln8/Pzk5eWVZd2jR4%2BqZMmS2rdvnxISEpSUlGT78Dh6S7B/8vf3z3r4wB1%2BQCF3%2B7/Ti9zG/v3ulxmAnbvQe8v%2Bd7CDKleurEuXLun8%2BfMqXry4pOvXopQqVUqFChWyWzcwMFB79uyxGzt8%2BLCtjqlcubIOHTpku7YmPT1dx44dy3IqgDNu66SIfPny6cqVK2rdurWOHDmizz//XGlpaTpx4oQmTpyotm3bKl%2B%2BfLbD9leuXLH96e3trTx58ighIUFjxoyxvaHbdSODs7p06aK5c%2Bfqzz//lCTt2rVLHTt21LfffuvQ9s2bN9fmzZu1Y8cO1a1bVwEBAfL19dWCBQsUHh4u6fr79vDwsBW8O3bs0Ny5c5WWlpbt%2B3niiSf0448/6tdff1VmZqY2btyoyZMnZ/ng/C8PDw917txZ//3vf3X27FllZmZq2bJlatOmjY4fP67g4GBVrFhRY8aMUUpKio4fP65Zs2bd9pwBAAD3U6FCBdWsWVPvvPOOEhMTFRsbq2nTpqlz585Z1m3Xrp02b96s1atXKyMjQ6tXr9bmzZttpxV26tRJ8%2BbN0/79%2B3X16lX997//VfHixVWrVi3D8t5Wcdq5c2dNmjRJH3zwgWbMmKE1a9aofv366tq1qxo0aKD//Oc/kq5X3TVr1lSjRo20YcMGjRs3TqtXr1ZISIg6duyokiVLqlq1ajp48OBtB%2B7QoYO%2B/vprde3a9ba3/afHHntML774ot544w2FhIRo6NCh6t27t8P3Ei1ZsqQqV66swMBA27kaDRo0kKenp%2B3UgAYNGqhr167q1q2bateurVGjRqlHjx5KSEjQ%2BfPnJV1v07/00kuaNGmSatasqXfffVfvvvuuatWqpffee08TJ0685QVakvTaa6%2BpRo0a6tq1q2rVqqU5c%2Bboo48%2BUrVq1ZQnTx599tlnio%2BPV/369dWvXz81a9bsDmcOAADcTGam8S9nffTRR8rIyFCzZs3UpUsXNWrUSIMHD5Z0/e5KK1askHT9QqmpU6fq008/Ve3atTVt2jRNnjxZFStWlHS9Fuzdu7eee%2B451a1bV3v37tWnn34qT09P50P%2BH4vVyBtTIfdxp4%2BPu5wn5U7cYU7d5ckqwcHXL94KCXH5w/rWTBf/N/8Hd/iISpxzeleYeM7p1avG7/P/DkrnCm7yUxsAAAC5wR1dEHW3dOzY8V8vFpKuPy7LyHMaXD0HAABwP3fjgqjchMP6cI47fXzc5VCUO3GHOeWwvuE4rG88DuvfBSYe1v%2BfuzYZ4n9uLZqjuVTnFAAAwN3ROXUOxSkAAICBKE6d4ybHuwAAAJAb0DkFAAAwEJ1T59A5BQAAgMugcwoAAGAgOqfOoTgFAAAwEMWpczisDwAAAJdB5xQAAMBAdE6dQ3EKAABgIIpT53BYHwAAAC6DzikAAICB6Jw6h84pAAAAXAadUwAAAAPROXUOxSkAAICBKE6dw2F9AAAAuAw6pwAAAAaic%2BocOqcAAABwGXROAQAADETn1DkUpwAAAAaiOHUOh/UBAADgMuicAgAAGIjOqXPonAIAAMBl0DkFAAAwEJ1T51CcAgAAGIji1Dkc1gcAAIDLoHMKAABgIDqnzqFzCgAAAJdB5xQAAMBAdE6dQ3EKAABgIIpT53BYHwAAAC6DzikAAICB6Jw6h84pAAAAXAadU8DFWGUxO4LDLHKDvJlWsxM4zCLJunWb2TFuyeLh4v/mNwQHS9u2yVIzRNq%2B3ew0N5WS7B6fU4tF8vaWUq9aZHXxyD4%2B5n1vOqfOoTgFAAAwEMWpczisDwAAAJdB5xQAAMBAdE6dQ%2BcUAAAALoPOKQAAgIHonDqH4hQAAMBAFKfOoTgFAADIxZKTkzV69GitW7dOGRkZatasmUaMGKECBQpku/6aNWs0bdo0xcbGytfXVx07dtTgwYPl4XH9bNFWrVrp9OnTtq8lacmSJQoICHAoD8UpAACAgdytczp69GidOXNGa9as0bVr1/T8889rwoQJGjFiRJZ1d%2B/erVdffVUffPCBHn30Uf3111/q37%2B/8ufPrz59%2BigxMVF//fWX1q5dq7Jly95RHi6IAgAAyKVSUlK0cuVKRUVFydfXV8WKFdPLL7%2BspUuXKiUlJcv6p06d0lNPPaWmTZvKw8NDAQEBCg8P15YtWyRdL159fX3vuDCV6JwCAAAY6m50TuPj43Xu3Dm7sRIlSsjf3/%2BW26ampiouLi7bZSkpKUpPT1dgYKBtLCAgQKmpqTp27JgefPBBu/Vbtmypli1b2u37559/Vtu2bSVJu3btko%2BPj7p3765Dhw6pbNmyioyMVNOmTR1%2BrxSnAAAABrobxemiRYs0ZcoUu7EhQ4YoMjLyltvu3LlTPXv2zHbZ0KFDJUn58%2Be3jfn837Nfk5KSbrrfxMREDR06VN7e3urdu7ckyWKx6OGHH9aLL76oMmXK6Pvvv1dkZKTmzZunoKCgW2aVKE4BAABcXkREhMLCwuzGSpQo4dC2oaGhOnDgQLbL9u7dqw8//FApKSm2C6BuHM4vWLDgv%2B7z6NGjioqKUrFixTR37lzbuv369bNbr127dvr222%2B1Zs0ailMAAAAz3I3Oqb%2B/v0OH8G9XxYoV5enpqcOHD6tGjRqSpCNHjsjT01MVKlTIdpsNGzboxRdfVJcuXfTSSy8pb97/X07OnDlT1apVU7169WxjaWlp8vLycjgTF0QBAADkUj4%2BPmrVqpUmTJighIQEJSQkaMKECWrTpo28vb2zrL9jxw4999xzev311/Xaa6/ZFaaSdObMGY0aNUqxsbHKyMjQkiVLtH37dj3xxBMOZ7JYrVar0%2B8MuZc7fXwsFrfIa5XF7AgOc5MpdRvuMp8WDzf5jAYHS9u2SSEh0vbtZqe5qZRkN/iH1/XPqLe3lJrq%2Bp/V/ztt0hQ//mj8PsPDjd/nDYmJiXr33Xe1bt06paenq1mzZnrrrbds56G2bt1abdu21cCBAzVw4ED9/PPPtvNSb6hZs6ZmzJihtLQ0TZgwQd99952uXLmiSpUq6ZVXXlFoaKjDeShO4Rx3%2Bvi4yW9%2BitPcy13mk%2BLUeBSnxjOzOF2zxvh9/uMC%2BRyPw/oAAABwGVwQBQAAYCB3e0KUq6E4BQAAMBDFqXM4rA8AAACXQecUAADAQHROnUPnFAAAAC6DzikAAICB6Jw6h%2BIUAADAQBSnzuGwPgAAAFwGnVMAAAAD0Tl1Dp1TAAAAuAw6pwAAAAaic%2BocilMAAAADUZw6h8P6AAAAcBl0TgEAAAxE59Q5dE4BAADgMuicAgAAGIjOqXMoTgEAAAxEceocDusDAADAZdA5BQAAMBCdU%2BfQOTVZfHy8kpOTzY4BAADgEihOTXT%2B/Hm1bNlSCQkJt1x32LBhGjZsmGHfe/LkyerRo4dh%2BwMAANdlZhr/yk04rG%2Bi1NRUuqYAAOQwua2YNBqd0zvUsWNHzZkzx/Z1jx499OSTT9q%2Bnjdvnrp166Z169bpqaeeUr169VSjRg11795dx44d07Vr19SmTRtJUps2bbR69WpJ0ueff67w8HAFBwerY8eO2rRpk22fFy5cUFRUlEJDQ9WwYUPNmzfPtiwxMVFvv/22Hn30UdWrV08vvPCCzp8/b1u%2Bbds2derUSUFBQXrqqad08uTJuzU1AAAAd4zO6R0KDw/Xxo0b1bt3byUlJWn37t1KT0/X5cuXVbhwYa1bt04NGjTQ0KFD9eGHHyosLEwXL17UkCFDNHXqVL3//vv69ttv1axZM3377be67777tHTpUk2bNk2ffPKJatSooa%2B//lqDBg3Szz//LEmKjo7Wp59%2Bqg8//FDLly/X66%2B/rvDwcJUsWVJvvPGGkpKStHTpUnl7e2v8%2BPEaMmSIvvzyS126dEnPPvus%2Bvfvr2eeeUZ//vmnBgwYoGrVqt3We46Pj9e5c%2BfsxkoULy5/f3%2BjphWAOwgONjuBY6pWtf/ThVksZidwzI2c7pLXLHROnUNxeoeaN2%2BuadOmKSUlRdHR0XrkkUd06dIlRUdHq379%2Btq8ebPGjBmjNm3aqHz58kpMTNTZs2fl5%2BenuLi4bPe5bNkyRUREKPj/fvA/%2BeSTCggIkLe3tySpQYMGql%2B/viSpdevWGjZsmGJjY5U3b16tWbNG3333nYoVKyZJeuONN1SrVi3t2bNHhw4dko%2BPj/r37y%2BLxaKaNWuqU6dO2rdv322950WLFmnKlCl2Y0Oee06RUVG3tR9TucFPVNdPaM8NptStuMV8bttmdoLbs2CB2QluydvsALfJy8vsBMjJKE7vUOXKlVWmTBnFxMRo48aNatCggc6fP6/ff/9dGRkZqlKlikqXLq2PP/5YCxculMViUWBgoBITE5U3b/bTfu7cOZUpU8ZuLCQkxPZ3X19f29/z5csnSbp27ZpOnTolSerSpYvdtnny5NHJkycVFxen0qVLy/KP33rly5e/7eI0IiJCYWFhdmMliheXrNbb2o9pLBa3yGp1o/LUTabUbbjLfFpqhtx6JVdQter1wrRrV2n/frPT3FTq7%2B5R8Fss1wvTq1dd/7PqbWLFT%2BfUORSnTmjWrJl%2B%2BeUXbdq0SRMnTtSFCxc0duxYJSYmqkWLFvruu%2B80b948ffnll7r//vslSaNHj9bBgwez3V/p0qV15swZu7FJkyapXbt2N81RsmRJSdJ3332nEiVK2MYPHz6scuXK6bvvvtOpU6eUmZkpD4/rpxmfPXv2tt%2Bvv79/1kP4rv7TCYDxtm83O8Ht2b/f5TO7249Sq9X9Mt9LFKfO4YIoJ4SHh2v16tW6fPmyqlWrpjp16uj06dP66aefFB4eritXrsjDw0Pe3t6yWq365ZdftHz5cqWnp0uSvP7vuEhiYqKk6xdZLVq0SH/%2B%2BacyMzP19ddfa/78%2BfLz87tpjpIlS6pJkyYaO3asLl68qPT0dH388cfq3LmzLl%2B%2BrLCwMFmtVk2ePFlpaWnavXu3vvrqq7s7OQAAAHeAzqkTgoKClDdvXoWGhspiscjb21u1atVSfHy8HnjgAd13333aunWrWrdurTx58uiBBx5Qr169NH/%2BfKWlpal48eIKDw9XRESEhg0bpqefflqXL1/WK6%2B8onPnzqlSpUqaPn26ihYtesss7733nv773/%2BqQ4cOSkxMVOXKlTVjxgxbJ3XmzJkaOXKkZs%2Berfvvv18tW7bUX3/9dbenCACAXIfOqXMsViuNeTjBnT4%2BbnJCH%2Bec5l7uMp8WDzf5jAYHX794KyTE5Q/rpyS7wT%2B8rn9Gvb2l1FTX/6z6%2BJj3vadONX6fzz1n/D5dFZ1TAAAAA9E5dQ7FKQAAgIEoTp3DBVEAAABwGXROAQAADETn1Dl0TgEAAOAy6JwCAAAYiM6pcyhOAQAADERx6hwO6wMAAMBl0DkFAAAwEJ1T59A5BQAAgMugcwoAAGAgOqfOoXMKAABgoMxM4193U3Jysl5//XWFhoaqZs2aevXVV5WUlPSv648YMULVq1dXcHCw7bVo0SLb8mXLlik8PFxBQUHq2LGjtm/fflt5KE4BAABysdGjR%2BvMmTNas2aNfvjhB505c0YTJkz41/V37dql0aNHa/v27bZXRESEJCkmJkajR4/W%2BPHjtWXLFrVr106DBg1SSkqKw3koTgEAAAzkTp3TlJQUrVy5UlFRUfL19VWxYsX08ssva%2BnSpdkWlGlpaTp48KCqV6%2Be7f6%2B%2BuortW7dWjVr1pSnp6d69%2B4tPz8/rV692uFMnHMKAACQg6WmpiouLi7bZSkpKUpPT1dgYKBtLCAgQKmpqTp27JgefPBBu/X379%2BvjIwMffTRR9q6dasKFSqkTp06qV%2B/fvLw8NDhw4fVqVMnu20qVaqk/fv3O5yX4hQAAMBAd6PTGR8fr3PnztmNlShRQv7%2B/rfcdufOnerZs2e2y4YOHSpJyp8/v23Mx8dHkrI97/TKlSuqU6eOevTooYkTJ2rfvn167rnn5OHhoX79%2BikpKcm2/Q3e3t5KTk6%2BZc4bKE4BAAAMdDeK00WLFmnKlCl2Y0OGDFFkZOQttw0NDdWBAweyXbZ37159%2BOGHSklJUYECBSTJdji/YMGCWdZv0KCBGjRoYPv6kUceUa9evbR69Wr169dPPj4%2BSk1NtdsmNTVVfn5%2Bt8x5A8UpAACAi4uIiFBYWJjdWIkSJZzeb8WKFeXp6anDhw%2BrRo0akqQjR47I09NTFSpUyLL%2BTz/9pPPnz%2Bupp56yjaWlpcnb21uSVLlyZR06dMhum8OHD6tx48YOZ%2BKCKAAAAAPdjQui/P399dBDD9m9HDmkfys%2BPj5q1aqVJkyYoISEBCUkJGjChAlq06aNreD8J6vVqnHjxmnTpk2yWq3avn275s6da7tav3Pnzlq5cqWio6OVnp6uOXPm6MKFCwoPD3c4E51TAAAAA7nbTfhHjBihd999V23btlV6erqaNWumt956y7a8devWatu2rQYOHKjw8HC9/vrrGjlypOLi4lS8eHFFRkaqffv2kqR69eppxIgRtuWVKlXS9OnT5evr63Aei9VqtRr9JpGLuNPHx2Jxi7xWWcyO4DA3mVK34S7zafFwk89ocLC0bZsUEiLd5k3A77WUZDf4h9f1z6i3t5Sa6vqf1f%2B5JueeevNN4/c5dqzx%2B3RVdE4BAAAM5G6dU1fDOacAAABwGXROAQAADETn1DkUpwAAAAaiOHUOh/UBAADgMuicAgAAGIjOqXPonAIAAMBl0DkFAAAwEJ1T51CcAgAAGIji1Dkc1gcAAIDLoHMKAABgIDqnzqFzCgAAAJdB5xQAAMBAdE6dQ3EKAABgIIpT53BYHwAAAC6DzikAAICB6Jw6h%2BIUQI5mkdXsCLfB4hZ5U5JdP6MkWSySt6TU37fJ6uKRffJbzI7gmOBgads2edcPkbZvNzvNzbn6Pzr%2BFcUpAACAgeicOofiFAAAwEAUp87hgigAAAC4DDqnAAAABqJz6hw6pwAAAHAZdE4BAAAMROfUORSnAAAABqI4dQ6H9QEAAOAy6JwCAAAYiM6pc%2BicAgAAwGXQOQUAADAQnVPnUJwCAAAYiOLUORzWBwAAgMugcwoAAGAgOqfOoXMKAAAAl0HnFAAAwEB0Tp1DcQoAAGAgilPncFgfAAAALoPOKQAAgIHonDqHzikAAABcBp1TAAAAA9E5dQ7FKQAAgIEoTp3DYX0AAAC4DDqnAAAABqJz6hyKUwAAAANRnDqHw/oAAABwGXROAQAADORundPk5GSNHj1a69atU0ZGhpo1a6YRI0aoQIECWdb9z3/%2Bo5UrV9qNpaamqn79%2Bpo5c6YkqVWrVjp9%2BrQ8PP5/D3TJkiUKCAhwKA%2BdUwAAgFxs9OjROnPmjNasWaMffvhBZ86c0YQJE7Jd9%2B2339b27dttr8mTJ6tw4cIaNmyYJCkxMVF//fWXVq9ebbeeo4WpRHEKAABgqMxM4193S0pKilauXKmoqCj5%2BvqqWLFievnll7V06VKlpKTcdNuEhAS9/PLLevPNN1W5cmVJ0u7du%2BXr66uyZcvecSYO6wMAABjI1Q7rp6amKi4uLttlKSkpSk9PV2BgoG0sICBAqampOnbsmB588MF/3e%2BECRNUvXp1tWvXzja2a9cu%2Bfj4qHv37jp06JDKli2ryMhINW3a1OG8FKcAAAAuLj4%2BXufOnbMbK1GihPz9/W%2B57c6dO9WzZ89slw0dOlSSlD9/ftuYj4%2BPJCkpKelf9xkbG6sVK1boq6%2B%2Bshu3WCx6%2BOGH9eKLL6pMmTL6/vvvFRkZqXnz5ikoKOiWWSWKUwAAAEPdjc7pokWLNGXKFLuxIUOGKDIy8pbbhoaG6sCBA9ku27t3rz788EOlpKTYLoC6cTi/YMGC/7rPr7/%2BWsHBwVk6q/369bP7ul27dvr222%2B1Zs0ailMAAICcIiIiQmFhYXZjJUqUcHq/FStWlKenpw4fPqwaNWpIko4cOSJPT09VqFDhX7f74Ycf1KdPnyzjM2fOVLVq1VSvXj3bWFpamry8vBzORHEKAABgoLvROfX393foEP7t8vHxUatWrTRhwgR9%2BOGHkq6fS9qmTRt5e3tnu83Fixd15MgR1a5dO8uyM2fO6KuvvtL06dNVunRpLV%2B%2BXNu3b9eoUaMczkRxCgAAYCBXuyDqVkaMGKF3331Xbdu2VXp6upo1a6a33nrLtrx169Zq27atBg4cKEk6efKkJKlkyZJZ9vXqq6/Kw8NDXbt21ZUrV1SpUiV99tlnuv/%2B%2Bx3OY7FarVYn3xNyM3f6%2BFgsbpHXKovZERzmDlNqkYsH/Cd3mFBJKanu8Rm1WCRvbyk11fWn1Se/e8ypgoOlbdukkBBp%2B3az09ycif/oDRoYv8/ffjN%2Bn66K%2B5y6oatXr%2Brs2bNO7%2BfYsWPOhwEAAHbc6T6nroji1A117dpVv//%2Bu1P7WLdunfr27WtQIgAAAGNwzqkbunjxotP7uHTpkjijAwAA4%2BW2TqfR6Jy6mT59%2Buj06dMaMWKE3n77be3Zs0c9evRQ7dq11aJFC82ZM8dWdMbFxalfv36qU6eOGjdurCFDhig%2BPl4xMTEaMWKETp8%2BreDg4H99agQAALh9HNZ3Dp1TNzNr1iyFhYVpyJAhatCggVq3bq0XXnhBs2bN0vHjxzV48GB5e3vrqaee0sSJE1WqVCl9/PHHunr1qqKiovTZZ59p%2BPDhGjVqlKZMmaJ169Y5/L2zfTpF8eJ35dYWAFyXxU2u3bmR0y3yBgebncAxVava/wncBRSnbmzFihUKCAhQt27dJEmVKlVS3759NW/ePD311FPy8vLSli1btGrVKtWrV08zZsyQh8edN8uzfTrFc88pMirKqfdxT7nBbynXT2jP9afU5QPac/0J1b/c%2BtBl3ca9v82zbZvZCW7PggVmJ3Bpua3TaTSKUzd26tQp7dmzR7Vq1bKNZWZmKk%2BePJKk4cOH69NPP9XMmTM1bNgwVa1aVcOHD7db/3Zk%2B3SK4sVd/x4tN7jJbXq4lZSxuJWU8VKvusdn1GK5Xphever60%2BpdP8TsCI6pWvV6Ydq1q7R/v9lpbs7dCn7YUJy6sVKlSik0NFQzZ860jV28eFFJSUmSrj8vNyIiQpGRkUpISNDUqVM1ZMgQRUdH39H3y/bpFK7%2BEx%2BA4dztP3ur1Q0yu/o9Q//X/v3ul/keonPqHC6IckP58uXTlStX1LZtW%2B3YsUMrVqxQRkaG4uPjNXDgQI0fP16S9Mknn2j06NFKTExU4cKF5ePjIz8/P0mSl5eXUlJSlJGRYeZbAQAgx%2BGCKOdQnLqhzp07a9KkSZo0aZJmzJihRYsWqX79%2Bmrfvr0eeOABW3H69ttvKzMzU82aNVPt2rW1c%2BdO23Nza9eurWLFiql27do6cOCAmW8HAADAhseXwjnu9PFxk/P5OOfUWJxzajweX2o8Hl96F5j4j/7ww8bvc9cu4/fpquicAgAAwGVwQRQAAICBcts5okajOAUAADAQxalzOKwPAAAAl0HnFAAAwEB0Tp1D5xQAAAAug84pAACAgeicOofiFAAAwEAUp87hsD4AAABcBp1TAAAAA9E5dQ6dUwAAALgMOqcAAAAGonPqHIpTAAAAA1GcOofD%2BgAAAHAZdE4BAAAMROfUOXROAQAA4DLonAIAABiIzqlzKE4BAAAMRHHqHA7rAwAAwGXQOQUAADAQnVPnUJwCAAAYiOLUORzWBwAAgMugcwoAAGAgOqfOoXMKAAAAl0HnFAAAwEB0Tp1DcQoAAGAgilPncFgfAAAALoPOKQAAgIHonDqHzikAAABcBp1TAAAAA9E5dQ7FKQAAgIEoTp3DYX0AAAC4DIpTAAAAA2VmGv%2B6F1JSUhQREaGlS5fedL2dO3fqySefVHBwsMLCwvTVV1/ZLV%2B2bJnCw8MVFBSkjh07avv27beVg%2BIUAAAglzt06JC6deumHTt23HS9v//%2BWwMGDFCHDh20ZcsWjR07VuPGjdOff/4pSYqJidHo0aM1fvx4bdmyRe3atdOgQYOUkpLicBaKUwAAAAO5W%2Bd006ZN6tWrl5544gmVKVPmpuv%2B8MMP8vX1Vbdu3ZQ3b17Vq1dPbdu21fz58yVJX331lVq3bq2aNWvK09NTvXv3lp%2Bfn1avXu1wHi6IAgAAMJCrXRCVmpqquLi4bJeVKFFCVatW1fr16%2BXl5aXZs2ffdF%2BHDh1SYGCg3VilSpW0ZMkSSdLhw4fVqVOnLMv379/vcF6KUwAAABcXHx%2Bvc%2BfO2Y2VKFFC/v7%2Bt9x2586d6tmzZ7bLpk6dqubNmzucIykpST4%2BPnZj3t7eSk5Odmi5IyhO4RyLxewEDomPj9eiRYsUERHh0H/IZnKPGXWnOXWPGXWf%2BZT%2B5/eOy4qPj9f06e4xp7JazU7gkPj4eC2aPFkR33/v%2BnNqorvxzzl58iJNmTLFbmzIkCGKjIy85bahoaE6cOCAITl8fHx05coVu7HU1FQVKFDAtjw1NTXLcj8/P4e/B%2BecIlc4d%2B6cpkyZkuX/OnHnmFNjMZ/GY06Nx5ya58ZV9P98RURE3PMcgYGBOnTokN3Y4cOHVblyZUlS5cqVb7rcERSnAAAALs7f318PPfSQ3cuM7nV4eLjOnz%2BvOXPmKD09XdHR0Vq5cqXtPNPOnTtr5cqVio6OVnp6uubMmaMLFy4oPDzc4e9BcQoAAIB/1bp1a33yySeSJD8/P82aNUvff/%2B9QkNDNXz4cA0fPlx169aVJNWrV08jRozQyJEjVadOHa1atUrTp0%2BXr6%2Bvw9%2BPc04BAAAgSVq3bl2WsVWrVtl9/fDDD2vhwoX/uo/27durffv2d5yBzilyhRIlSmjIkCEqUaKE2VFyDObUWMyn8ZhT4zGnuBcsVqubXCIIAACAHI/OKQAAAFwGxSkAAABcBsUpAAAAXAbFKQAAAFwGxSkAAABcBsUpAAAAXAbFKQAAAFwGxSkAAABcBsUpAAAAXAbFKQAAyNa5c%2BeyHT906NA9ToLchMeXIsdKSEjQihUrdOrUKQ0dOlRbtmxR06ZNzY7l1hYvXqwvvvhC8fHxWrZsmcaPH69x48apQIECZkdzK2FhYbJYLDddZ%2B3atfcojfvr0aPHLedz7ty59yhNzhISEqJt27bZjV27dk21a9fOMg4YJa/ZAYC7Yc%2BePXrmmWf0wAMP6MCBA%2BrZs6eGDh2qESNGqFOnTmbHc0tz5szRl19%2Bqb59%2B%2Bq9995TgQIFFB8fr3HjxmnMmDFmx3MrkZGRkq5/TteuXatnnnlG5cuX15kzZzR79mw1a9bM5ITuJTQ0VJJ08uRJ/fTTT%2BrUqZPKly%2Bvs2fPavHixXrsscdMTuhejh8/rr59%2B8pqtSolJSXL5zE1NVVly5Y1KR1yAzqnyJG6d%2B%2Bujh07qmPHjqpdu7a2bNmijRs3aty4cVq9erXZ8dxSy5YtNW3aNAUEBKhOnTravHmz4uPj9cQTT%2Bi3334zO55bateunSZNmqSAgADb2PHjxzVgwACtWbPGxGTuqWvXrnr55ZcVEhJiG9u9e7feeustLVu2zMRk7mf9%2BvW6ePGiRo4cqVGjRtkt8/LyUu3atVWiRAmT0iGno3OKHOngwYNq3769JNkO9zVq1EjPP/%2B8ianc28WLF1WxYkVJ0o3/py1WrJgyMjLMjOXWYmNjVb58ebuxkiVLKj4%2B3qRE7m3fvn2qUaOG3ViVKlV07NgxcwK5sRunQN13332qU6eOyWmQ23BBFHKkokWL6ujRo3ZjR48eVfHixU1K5P6qVq2qRYsWSfr/Bf/q1atVuXJlM2O5terVq%2Bvdd99VWlqaJCklJUWjR49WzZo1TU7mngICAjRnzhy7sU8%2B%2BURVq1Y1J1AOULFiRb3zzjuSpD/%2B%2BEP169dX69atdfjwYZOTISfjsD5ypLlz52rOnDkaOHCgxo8frzFjxujjjz/WE088oT59%2Bpgdzy3t2bNHvXv3VkBAgHbv3q169eppx44dmjFjRpZuFRxz9OhRPfvsszpz5oz8/Pxs3enPPvtMpUuXNjue29m2bZsGDhyo/Pnzq1SpUjp9%2BrQyMzM1c%2BZMValSxex4bikyMlLJycmaMWOGOnXqpJCQEPn4%2BOjPP//U559/bnY85FAUp8ix5s%2BfrwULFujUqVMqVaqUunTpomeeeeaWV/Xi38XFxWnFihU6ffq0SpUqpbZt26pMmTJmx3JrGRkZ2r59u%2BLi4lSqVCmFhITIw4ODWnfq0qVL%2Bvnnn23zGRYWpkKFCpkdy201adJEq1evVmJioh599FH9/vvvKlSokEJDQ7V161az4yGH4pxT5EgZGRnq1q2bunXrZje%2BdetWDpk6oWTJkurfv7/ZMXKUzMxMXbp0SefPn1ezZs108OBBDkM7wdfXV4GBgcqfP7%2BaNGmiK1eumB3JraWkpMjb21s//vijAgMD5efnp8TEROXNS/mAu4dPF3KkoKAgvfnmm3r66aftxvv378%2B9%2BW4T9%2BS8e06cOKE%2BffooPT1dly9f1qOPPqpOnTppypQp3JP3Dly4cEHPPfecdu/eLU9PTy1ZskSdO3fWrFmzFBwcbHY8t/TII49o5MiR2rp1q1q1aqXz58/r7bff5iIp3FUUp8ixpk2bptjYWL366qu2Mc5iuX037skJ440dO1YdO3bUoEGDVKdOHVWsWFFjxozRRx99RHF6B9555x0FBgZq9uzZaty4sQICAjRgwAC99957%2BvLLL82O55bGjh2riRMnqlatWnr22We1d%2B9epaWlcW9j3FWcc4ocKSQkRCtXrlS/fv0UGBio999/X/ny5cv2aSe4fRcvXpSfn5/ZMdxeaGioNm7cqHz58tnuHZuZmak6derojz/%2BMDue22nQoIF%2B%2Bukn%2Bfj42OYzPT1d9evX15YtW8yOB8BBnHWPHKts2bJasGCB4uLi1LNnT128eJHzpJyQlJSk4cOHq0aNGqpfv75CQkL03nvv2W6DhNtXqFAhnT9/3m7s3LlzKlKkiEmJ3Junp6dSU1Ml/f%2BjJElJSTxe10mLFy9W27ZtFRoaqtOnTysqKkpJSUlmx0IORnGKHM3Pz0%2Bff/65ihUrpq5du3JY3wnjx4/XoUOHNG3aNK1atUqTJk1SdHS0Jk2aZHY0t9W2bVsNGTJEv/32mzIzM/Xnn3/q5ZdfVuvWrc2O5pbCwsL0yiuv6NixY7JYLLpw4YJGjRqlRx991OxobmvOnDmaOXOmevTooWvXrqlAgQKKi4vTuHHjzI6GHIzD%2BsiR2rdvr2%2B%2B%2Bcb2tdVq1dtvv60vv/xS%2B/fvNzGZ%2B2rYsKFWrFihokWL2sbOnj2rzp0769dffzUxmftKT0/XxIkTtXDhQqWkpMjLy0udO3fWa6%2B9pnz58pkdz%2B0kJSXp9ddf1w8//CDp%2BsMiHn30Ub3//vvcTuoO8dhimIHiFLnKmTNnuLn5HQoPD9eSJUvsDjlfvnxZjz32mH7//XcTk%2BUMCQkJ8vPz4z68BkhISNDJkydVqlQp%2Bfv7mx3HrdWpU0fR0dHy8PBQ7dq1tWXLFl27dk3169dXTEyM2fGQQ3ECHnKUkSNHauTIkXr99df/dR0OR92e06dPS5I6dOigF154QcOGDVPZsmUVHx%2Bv999/X7179zY3oJvbuXOnTpw4oWvXrtmNd%2BjQwZxAbu7cuXOKjY3VtWvXdPz4cR0/flySVLt2bZOTuacbjy1%2B%2BumneWwx7hmKU%2BQoHAgw3o37nN6Y23bt2tl%2BSVmtVq1fv14DBgwwM6LbmjRpkj777DMVL15cnp6etnGLxUJxegfmz5%2BvMWPGZPk5YLFYtG/fPpNSubdhw4apV69e%2Buabb5ScnKz%2B/fvbHlsM3C0c1gdwU6dOnbrlOmXLlr0HSXKeevXq6YMPPlBoaKjZUXKExo0b68UXX1Tr1q3tin3cuTFjxmjAgAFasWKFTp06pdKlS6tNmzb64IMP9N5775kdDzkUnVPkSOfPn9dnn32mN954Q3/88YeioqJUtGhRffjhhwoICDA7nlu5WeGZkZGhgwcPUpzeoTx58lCYGigtLY2OswHi4uK0adMmSdJXX32l6tWrq3jx4ipevLik60%2BE%2B/HHH82MiByO4hQ50qhRo5ScnCyr1aqxY8fq8ccfl4%2BPj95%2B%2B219/vnnZsdzSz///LNGjRqluLg4u8OmefPm1a5du0xM5r6aNm2qb7/9Vm3atDE7So4QGhqq6Oho1a1b1%2Bwobs3Pz0/z5s1TQkKC0tLS9NFHH9kt9/Ly0pAhQ0xKh9yAw/rIkZo0aaLVq1crMTFRjz76qH7//XcVKlRIoaGh2rp1q9nx3FKbNm3UoEEDFS5cWAcOHFCbNm00depUde7cWT169DA7nlvp0aOHLBaLkpKStG/fPlWqVEm%2Bvr5268ydO9eccG7oxgWQFy5cUExMjOrXr59lPrkQ8s707dtXM2fONDsGchk6p8iRUlJS5O3trR9//FGBgYHy8/NTYmIiT4hyQmxsrF555RWdPHlS0dHRatGihR544AG98MILFKe36Z%2BH8ps2bWpikpylWLFievzxx82OkaNQmMIM/KZGjvTII49o5MiR2rp1q1q1aqXz58/r7bffVp06dcyO5raKFi0qDw8PlSlTRkeOHJEkVapUSWfPnjU5mfv55yHRI0eOqGTJkipYsKC2b9%2BuwoULc170bfpnV/Ty5cvy8vKSl5eXjhw5oqJFi8rPz8/EdABuF48vRY40duxYpaWlqVatWho4cKBOnTqltLQ0jRgxwuxobqtKlSr68MMPJV3vUG3YsEExMTHy8vIyOZn7%2Bu6779ShQwcdO3ZMkrRjxw49%2BeST2rBhg7nB3FR0dLQeffRR222jVq5cqZYtW%2BrPP/80ORmA28E5p8i1BgwYoM8%2B%2B8zsGG7jyJEjioqK0meffaa9e/fq%2BeefV2Zmpl599VU988wzZsdzS61bt9awYcPUqFEj29jGjRv1/vvva8WKFSYmc0%2BdOnXSU089pSeffNI29vXXX%2Burr77SwoULTUwG4HZQnCLXCgkJ0bZt28yO4bbi4%2BOVlJSkihUrmh3FbWX3GbRarapdu7b%2B%2BOMPk1K5r5o1a2a54JH5BNwP55wCcFh0dLS%2B%2BeYbnTt3TmXKlFHnzp3NjuTWypYtq40bN9p1Tjdt2qQyZcqYmMp9FStWTH/%2B%2BaceeeQR29ju3btt9%2BcE4B4oTgE4ZPHixRo9erRatGihBx98UCdPnlSPHj00YcIEhYeHmx3PLQ0YMEDPPfecWrRoobJly%2Br06dP68ccf9e6775odzS1169ZNAwYMUEREhG0%2BFy9ezD05ATfDYX3kWhzWvz3NmzfXqFGj1KBBA9vYhg0b9N5772nVqlUmJnNvMTExWr58uc6dO6fSpUvriSeeUEhIiNmx3NbSpUvt5rNjx4485ABwMxSnyLUoTm9PcHCw/vjjD%2BXJk8c2lpmZqbp162rz5s0mJnNfM2fOVN%2B%2BfbOMf/DBB3r%2B%2BefvfSA3991336lVq1ZZxhctWqSIiAgTEgG4ExzWB%2BCQRo0aad68eerVq5dtbNWqVapfv76JqdxPQkKC7T6xkydPVo0aNeweB3vlyhV9/vnnFKcOSklJ0cWLFyVJb7zxhoKCgrLM5/jx4ylOATdCcYpci4MGt%2BfatWsaP368li1bpvvvv19xcXHauXOnHnzwQfXs2dO2Ho/dvLl8%2BfIpKirKVlB17949y3IKKcclJiaqdevWSk1NlSSFhYXJarXKYrHY/mzevLnJKQHcDg7rI0eaNWuWOnTooKJFi/7rOj/88INatGhxD1O5tylTpji0HhefOO6xxx7T999/b3YMt3fhwgWlpKSobdu2%2Bvbbb%2B2WeXl5cbU%2B4GYoTpEjdenSRfv27VOTJk305JNPqlGjRrJYLGbHAhySkJBw0/%2BxQvYyMzPl4ZH1wYcZGRnKm5cDhYC7oDhFjnXkyBEtXbpUK1eulIeHhzp27KiOHTvqvvvuMzuaW7p48aK%2B%2BOILxcXFKTMzU5KUnp6ugwcP8jSjO/Tnn3/qvffeyzKnCQkJ2r17t8np3M%2BJEyc0derULPP5119/KTo62uR0ABxFcYocLzMzU%2BvXr9fYsWN19uxZ7d271%2BxIbmngwIE6duyYihYtqsTERJUpU0a//vqrunXrptdff93seG6pc%2BfOKleunHx9fRUbG6sGDRpo7ty56tmzJ4%2BEvQM9evSQ1WqVn5%2BfLly4oGrVqmn58uXq3bs3p5sAbiTr8Q8gB4mOjtabb76pV155RUWLFtXIkSPNjuS2tmzZos8//1zDhg1T%2BfLl9cknn2js2LE6evSo2dHc1qFDhzRu3Dh169ZN165d0zPPPKNJkyZp5cqVZkdzS7t379bUqVM1ePBgFSpUSMOHD9fEiRO1adMms6MBuA2chIMc6cYv%2BMTERLVt21YLFixQ1apVzY7l1vLmzauSJUvKx8dHBw4ckCS1bt1a7733nsnJ3FfhwoXl7e2tcuXK6dChQ5KkoKAgnTp1yuRk7snHx0dFihRR3rx5dfDgQUlS48aN9dprr5mcDMDtoHOKHOmnn37S888/r40bN%2Bqtt96yFaavvvqqycncV9myZbV7924VLlxYSUlJSkhIUHJysu0WPrh9DzzwgL788kt5eXkpf/782rdvn44cOcLFe3eofPny2rBhgwoUKKDMzEzFxsYqLi5OGRkZZkcDcBvonCLHiIuLsx2%2BO3nypDIzM/Xdd9/Zll%2B5ckU//vijWfHcXteuXdWjRw%2BtWrVKbdq0Ua9evZQ3b17Vrl3b7Ghua%2BjQoRo0aJAaNGigvn37qkuXLsqTJ4%2Befvpps6O5pWeffVZRUVH69ttvFRERoaeeekp58uRRs2bNzI4G4DZwQRRyjLS0NHXt2lUJCQk6c%2BaMSpcubbfcy8tLnTt3zvZxkXDMn3/%2BqapVq8pisWjOnDlKTExUnz59VKRIEbOjua2rV68qX758slgs2rlzpxITE9WgQQOzY7ml7t27q2HDhurQoYNKlSql1atXKzExUR06dFC%2BfPnMjgfAQXROkWPky5dPS5YskST17dtXM2fONDlRzrN//36VLFlSJUuWlL%2B/v3x9fSlMnRAWFqYWLVqoefPmqlWrlmrUqGF2JLfWqlUrrVu3TlOnTlXVqlUVHh6uFi1aUJgCbobOKQCHfPTRR1q2bJlmz56tChUqaO3atXrnnXf09NNPq1%2B/fmbHc0vr16%2B3vaxWq5o1a6YWLVqobt26ypMnj9nx3FZiYqJ%2B%2BeUXrV%2B/Xj/99JPuu%2B8%2B7oAAuBGKUwAOady4sebPn69y5crZxk6cOKFevXpp/fr1JibLGf7880%2BtWbNGCxYsUL58%2BRQTE2N2JLeUmJio6Oho/fbbb/r999915swZ1apVS7NmzTI7GgAHcVgfgEMSExOznMdbunRpJScnm5QoZzh48KB%2B//13/f7779qyZYv8/Pw45/QO3XhscaVKlRQaGqrhw4erTp068vLyMjsagNtAcQrAIQ899JA%2B%2B%2BwzDR482DY2a9Ys7h/rhIYNGyopKUkNGjRQo0aNNGzYMD3wwANmx3JbXl5e8vT0VJEiRVSsWDEVL16cwhRwQxzWB%2BCQPXv2qE%2BfPvLx8VGpUqV09uxZZWRkaMaMGRSod2jYsGH67bff5O3trQYNGqhhw4aqW7euChYsaHY0t5WcnKzo6Ght3LhRmzZt0pUrV1S/fn29//77ZkcD4CCKUwAO%2B/vvv7V%2B/XrFx8erdOnSatKkiQoVKmRbfvbsWZUqVcrEhO7pwIEDtmJq586dCgwM1IIFC8yO5bauXr2q6Oho/fLLL1q9erU8PT31yy%2B/mB0LgIM4rA/AYUWKFFGHDh3%2Bdfnjjz%2Bubdu23btAOUSBAgXk4%2BMjT09PZWZmKjMz0%2BxIbmnu3Ln65ZdftGXLFpUuXVrNmzfXxx9/rKCgILOjAbgNdE4BGCY4OFjbt283O4bbeOedd7Rx40adPHlSderUUbNmzdS8eXP5%2B/ubHc0tderUSeHh4WrevLkqVapkdhwAd4jOKQDD8Ez423P27FkNHjw4y%2BkR/7R161bVrFnzHidzT19//bXZEQAYgM4pAMOEhIRwWN9gzCmA3MbD7AAAgH9H/wBAbkNxCgAujFMlAOQ2FKcAAABwGRSnAAyTL18%2BsyMAANwcV%2BsDuKktW7bccp3atWtLkqKjo%2B92HABADkdxCuCmevToIcn%2B3MciRYroypUryszMlK%2BvrzZt2mRWPABADkNxCuCm9u/fL0maOXOmDh48qOHDh6tQoUJKTk7W%2BPHjVaRIEZMT5mwVKlQwOwIA3FPc5xSAQ%2BrXr69169bJ29vbNnb16lU1btxYMTExJiZzb0eOHNGXX36ps2fPavTo0Vq1apW6d%2B9udiwAMA0XRAFwSGZmpi5cuGA3dvLkSeXJk8ekRO7vt99%2BU5cuXXTx4kX9/vvvSk1N1dSpU/XZZ5%2BZHQ0ATENxCsAh7du3V9%2B%2BfbVkyRL99ttvWrhwoZ599lk99dRTZkdzWxMnTtTEiRP13//%2BV3ny5FHp0qX12WefadGiRWZHAwDTcM4pAIe88soryp8/vz7%2B%2BGPFxcWpdOnS6tKli/r37292NLd1/PhxNW7cWNL/v%2BDs4Ycf1t9//21mLAAwFcUpAIfkzZtXQ4cO1dChQ82OkmOUKVNG27ZtU82aNW1ju3btUunSpU1MBQDmojgF4JDMzEx9//33OnHihDIyMuyWDRkyxKRU7u3ZZ5/VoEGD9PTTTys9PV3Tp0/XF198oRdffNHsaABgGopTAA4ZMWKEvv32W1WpUkWenp62cZ79fudat26tggULav78%2BSpTpoyio6P15ptvqmXLlmZHAwDTcCspAA6pVauWFi1apICAALOjAAByMDqnABxSqFAhVaxY0ewYOUpsbKw%2B%2BeQTnTp1SpmZmXbL5s6da1IqADAXxSkAh7Rp00azZs1Sv379zI6SY7z44ovy9PRU3bp15eHBnf0AQOKwPoBbCAsLk8ViUUZGhuLi4lSoUCEVLlzYbp21a9ealM69BQcHa9OmTXZP3QKA3I7OKYCbioyMvOlyLoi6c1WrVtXZs2dVoUIFs6MAgMugcwrAIT169PjXQpTzI%2B/Mnj179Nxzz6lFixZZutHcngtAbkXnFIBDQkND7b6%2BePGivv/%2Be0VERJiUyP1NnjxZycnJ2rNnj905p3SjAeRmdE4B3LE9e/bovffe0%2Beff252FLcUHBysH3/8UcWLFzc7CgC4DC4PBXDHHnroIe3evdvsGG7L399fXl5eZscAAJfCYX0ADjl9%2BrTd1%2Bnp6Vq1ahXPgXdC3759NXjwYPXs2VNFihSxO5xfu3ZtE5MBgHk4rA/AIVWrVrUrnqxWq4oUKaIxY8YoPDzcxGTuq2rVqtmOWywW7du37x6nAQDXQHEKwCGnTp2y%2BzpPnjwqVqyYPD09TUoEAMiJKE4B4B47e/asSpUqleVUiX8qU6bMPUwEAK6D4hQA7rGQkBBt27bNdqrEjR/DN/7OYX0AuRnFKQDcY2fOnFHp0qWznCrxT2XLlr2HiQDAdVCcAoBJBg0apI8//jjLePfu3TVv3jwTEgGA%2BbiVFADcQydPntTy5cslSb/%2B%2BqumTJlitzwxMVEHDhwwIRkAuAaKUwC4h8qUKaNDhw4pISFB165dU0xMjN1yLy8vjRgxwqR0AGA%2BDusDgEmGDx%2BuMWPG3HSdrVu3qmbNmvcoEQCYj%2BIUAFzYjSv7ASC38DA7AADg39E/AJDbUJwCgAv75yNjASA3oDgFAACAy6A4BQAAgMugOAUAAIDLoDgFAACAy6A4BQAXVqFCBbMjAMA9xX1OAeAeu/H40pvp0KHDXc8BAK6I4hQA7rGwsLCbLrdYLFq7du09SgMAroXiFAAAAC4jr9kBACA3i42NVVxcnO1JUOnp6Tp48KB69%2B5tbjAAMAmdUwAwyaeffqpJkybZngJltVplsVj04IMPaunSpSanAwBz0DkFAJMsWLBAH330kfLly6d169bpxRdf1OjRo1W6dGmzowGAabiVFACY5PLly2rRooWqVq2q3bt3y9fXV2%2B%2B%2BaZWr15tdjQAMA3FKQCYxN/fX4mJiSpZsqROnjwpq9WqokWL6u%2B//zY7GgCYhsP6AGCS2rVrKyoqSh988IGqVaumiRMnysvLSyVLljQ7GgCYhs4pAJhk2LBhuv/%2B%2B5WRkaE333xTa9eu1eLFi/Xmm2%2BaHQ0ATMPV%2BgBgkp07d6pGjRpZxn/55Rc1btzYhEQAYD46pwBgkmeeeSbLWGJiooYOHWpCGgBwDZxzCgD30PHjx9W6dWtdu3ZNVqtVDz74YJZ1QkJCTEgGAK6Bw/oAcI/t27dPly9f1oABAzR9%2BnTbzfclycvLS4GBgfLx8TE5JQCYg%2BIUAEwSGxurcuXKSZIuXLigIkWKKG9eDmgByN0oTgHAJOnp6Xr//ff11VdfKTU1Vfny5VO7du301ltvKV%2B%2BfGbHAwBTcEEUAJhk2rRpiomJ0QcffKBvv/1WH3zwgXbu3KkPPvjA7GgAYBo6pwBgkubNm2v27Nm2Q/uSdOLECXXr1k0bN240MRkAmIfOKQCY5O%2B//1bp0qXtxkqXLq3U1FSTEgGA%2BShOAcAkVapU0cKFC%2B3GFi5cqMDAQJMSAYD5OKwPACb5448/1KdPH1WtWlXlypXTiRMndPjwYc2cOZN7nQLIteicAoBJvv/%2Be33zzTdq2LChChQooPDwcH377bdZuqkAkJtwQz0AuIfi4uK0adMmSdJXX32l6tWrq3z58ipfvrwkad26dfrxxx/NjAgApuKwPgDcQ2lpaeratasSEhJ05syZLBdEeXl5qXPnzurbt69JCQHAXBSnAGCSvn37aubMmWbHAACXQnEKAAAAl8EFUQAAAHAZFKcAAABwGRSnAAAAcBkUpwAAAHAZFKcAAABwGRSnAAAAcBkUpwAAAHAZFKcAAABwGf8P4Sfoc%2BQY/wwAAAAASUVORK5CYII%3D" class="center-img">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAqcAAAJcCAYAAADEsA3eAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/d3fzzAAAACXBIWXMAAA9hAAAPYQGoP6dpAABuOUlEQVR4nO3de3zO9f/H8ec1dnKc05xFY5YUm8OcY4zkGDI55xSy0VmllHMlyrFyTKhJFFFKCRUjp3I%2B5czGRsw22%2Bz6/eHr%2BnW1xcX14XNd87jfbtet9v4c9rzeu9pevT4ni9VqtQoAAABwAR5mBwAAAACuozgFAACAy6A4BQAAgMugOAUAAIDLoDgFAACAy6A4BQAAgMugOAUAAIDLoDgFAACAy6A4BQAAgMvIaXYAALgde/fu1eLFi7VhwwbFxsbqypUrKliwoCpUqKCGDRuqQ4cO8vHxMTsmAOAWWXh8KQB3M2nSJE2fPl0ZGRnKkyePypQpI09PT509e1anTp2SJBUvXlxTp07Vgw8%2BaHJaAMCtoDgF4Fa%2B/PJLvfrqq8qVK5fGjh2r8PBw5ciRw7b80KFDevXVV7V9%2B3YVKFBAK1euVMGCBU1MDAC4FZxzCsCtfPjhh5Kkl156SY8%2B%2BqhdYSpJAQEBmj59ugoVKqTz589r3rx5ZsQEANwmilMAbuPixYs6duyYJKlKlSr/uV7BggXVpEkTSdIff/xxV7IBAIzBBVEA3EbOnP//K2vNmjWqVKnSf64bGRmp7t27q1ChQraxoUOHaunSpXrllVdUv359TZw4UZs3b1Zqaqruu%2B8%2BPf744%2BrUqZO8vb2z3OfmzZv16aefauvWrbpw4YLy5cunqlWrqlu3bqpdu3aW21y8eFGff/651q5dq4MHDyoxMVG%2Bvr4qU6aMGjVqpO7duyt//vx221SsWFGS9Ouvv2rcuHH68ccf5eHhoQcffFCzZ8/WsGHDtHTpUo0aNUo1atTQ5MmTtXHjRl26dEmlSpXSE088oZ49e8pisej777/XJ598oj179igjI0NBQUEaMGCAHnnkkUxZU1JS9OWXX2r16tXat2%2BfLl68KC8vL5UoUUL16tXTU089paJFi9ptExYWppMnT2rlypWKj4/XzJkztWPHDiUlJalUqVJq3ry5evfurdy5c//nzwoA/olzTgG4lSeffFJbt26VxWJRmzZt1KFDB4WEhGQ6vJ%2BV68Vpu3bttGrVKiUlJalChQpKT0/X4cOHJUnVqlXTRx99pLx589ptO378eM2YMUOSlD9/fpUqVUpxcXE6e/asJKlPnz568cUX7bY5cuSIevbsqdOnTytnzpwqU6aMfH19dfLkSV24cEGSVK5cOX355Zd2xdv14jQkJETbtm1TYGCgEhISFBoaqvfee8/ufXz77bdKT09XQECA4uPjbXn69esni8Wijz76SPny5VPp0qX1119/KSkpSRaLRR9//LEaNGhg%2B54JCQnq0aOH9u/fL4vFojJlyihv3ryKjY217bNQoUJasmSJihUrZtvuenH61FNPae7cufLy8lLZsmX1999/68yZM5Kk4OBgLViwwKGfEQDICgBuZNeuXdaqVataAwMDba%2BQkBBr3759rR999JF1%2B/bt1qtXr2a57csvv2zbplGjRtbdu3fblm3dutVap04da2BgoPX111%2B32%2B6zzz6zBgYGWqtXr279%2BuuvbeMZGRnWFStW2PIsWrTIbruuXbtaAwMDrR07drTGxsbabbd06VJrUFCQNTAw0Dp//ny77a5nrFy5snXTpk1Wq9VqvXr1qvX8%2BfOZ3seTTz5pjYuLs60zdOhQa2BgoDUoKMhasWJF66xZs2zzkZCQYG3btq01MDDQ2rVr1yznJjw83PrXX3/ZLVu3bp21SpUq1sDAQOu4cePsljVq1MiWZejQodaLFy/a3uP8%2BfNty3744YcsfyYA8G%2BccwrArVSqVElffPGFqlWrZhtLTEzU2rVr9d5776ljx46qV6%2BeJk6cqOTk5Cz34eHhoWnTpumBBx6wjQUHB%2Bvtt9%2BWJH3xxReKjY2VJKWmpmry5MmSpDFjxqh169a2bSwWix577DFbx3Ty5MlKT0%2BXJMXHx%2BvAgQOSpJEjR8rf399uu7Zt26pmzZqSpH379mWZs3nz5qpRo4Yts5%2Bfn93ynDlzasKECSpSpIhtnX79%2BkmSMjIy1KZNG/Xq1UseHtd%2B1RcoUEDdu3eXJO3evdu2n/T0dP3%2B%2B%2B%2ByWCx65ZVXVLZsWbvvU79%2BfT322GOSpP3792eZNSgoSGPGjLF1nC0Wi7p06WLrAm/ZsiXL7QDg3yhOAbid8uXLa%2BHChfrqq680aNAgBQcHy9PT07Y8Pj5eH374oVq3bm07tPxPtWrVUlBQUKbxevXqqVSpUsrIyNCaNWskSdu2bdO5c%2BeUO3duNW7cOMs8rVu3loeHh2JjY21FX6FChbRx40bt2LFDgYGBmba5evWq8uTJI%2BnauZ5Z%2BWcBnpWKFSvaHWKXpJIlS9r%2BPavzSq8XyYmJibaxnDlzavXq1dqxY4caNmyYaRur1apcuXLdMGvDhg1lsVgyjd9///2SpEuXLt3wvQDAdVwQBcBtPfDAA3rggQcUGRmp5ORkbd26Vb/88ou%2B/vprxcfH69ixYxo8eLCio6Pttnv44Yf/c58VK1bUiRMndOTIEUmydT/T0tLUpUuX/9wuR44cysjI0OHDh%2B327%2BPjo9OnT2vHjh06duyYjh8/rkOHDmnPnj1KSkqSdK3LmZXrHdH/Urx48UxjXl5etn8vUKBApuX/vKjs37y9vRUfH6/t27fryJEjOnHihA4fPqw9e/bo77//vmHWf3aG/%2Bn6U7quXr36328EAP6B4hRAtuDr66u6deuqbt26Gjx4sF599VWtWLFC27dv165du%2ByeFPXvq%2BP/6XqH8OLFi5L%2Bv%2BOXmpqqrVu33jTH9e0k6fDhw3rnnXe0du1au6IuT548ql69uuLi4rR3797/3NfNHr/q6%2Bt7w%2BXXD%2Bc74uzZs3r77bf13XffKS0tze57PPTQQ7p69eoND83/syjOipVrbwE4iOIUgNt44403tHHjRj3%2B%2BOMaMGDAf67n4%2BOjESNG6Pvvv1daWpr%2B%2Busvu%2BL0escyK9cPd1%2B/BdX1AvDBBx/UkiVLHM4aHx%2Bvrl27Kj4%2BXiVKlFDHjh1VqVIl3X///SpVqpQsFouef/75Gxand8uVK1fUo0cPHTp0SH5%2BfnryySdVuXJlBQQEqEyZMsqRI4cmTpzIeaMA7gqKUwBu48qVKzp69KhWr159w%2BJUutadzJ07ty5cuJDp8aXXD9Vn5XqxWL58eUnXbvUkXbstVHp6epaHxa1Wq2JiYlSsWDGVKFFCXl5e%2BvLLLxUfHy8/Pz99%2BeWXWT5C9fpFV2ZbvXq1Dh06pJw5cyo6OjrTBVGSsjx3FwDuBC6IAuA2rl8pv3Pnzpt2MX/55RdduHBBfn5%2BmZ4mtW7dOtu9O/9pzZo1On36tLy8vBQWFiZJqlGjhvLmzavLly//5/dcvny5evTooebNm9uKuBMnTkiSSpQokWVhevDgQW3fvl2S%2BedjXs%2BaO3fuLAvTc%2BfO6eeff5ZkflYA2R/FKQC3UbduXTVr1kySNGzYMI0ePdpWWF135coVffnllxoyZIgkafDgwZmeTpSUlKSBAwfq9OnTtrGYmBi98sorkq7dwP76LZFy5cpluz3T6NGj9eWXX9qdP7p69WoNHz5c0rVbP5UpU0bS/1%2BlvnfvXq1atcq2vtVq1bp169SnTx/buZ3/dcuru%2BV61r///luffPKJ3fmh27dv11NPPWV7aIDZWQFkfxzWB%2BBWxo8fr1y5cumrr77SvHnzNG/ePJUoUUKFChXSlStXdOTIEaWmpsrT01PPP/%2B8OnfunGkfZcuW1Z49e9SkSRMFBgYqKSnJdnV%2By5Yt9fTTT9ut37dvXx0/flyLFi3Sq6%2B%2BqnfffVelSpVSbGys4uLiJF17mtOoUaNs23To0EELFy7U0aNHFRUVpZIlS6pAgQI6ffq04uPj5enpqZo1a2rTpk2mH94PCwtTcHCwtm3bpjFjxmjGjBkqWrSozp49q9jYWFksFtWpU0e//fab4uLiZLVas7xtFAAYgeIUgFvx8vLSuHHj1KVLF61cuVIxMTGKjY3V3r175evrq3LlyqlevXrq0KGDrSP4bw899JDGjx%2BvSZMmacuWLcqZM6dq1qypJ5980naz%2BX%2ByWCwaOXKkmjVrps8//1zbt2/Xnj175O3trapVq6ply5aKiIiwu2I9T548Wrx4sWbMmKE1a9boxIkTOnfunIoVK6aGDRuqR48eypUrl5o0aaK9e/fq1KlTKlGixB2btxvJkSOH5s6dq08//VQrVqzQ8ePHtX//fhUpUkSPPfaYunTpogcffFChoaG6cOGCtm7detN7sALA7bJYub8HgHvE9WfSt2rVSuPHjzc7DgAgC5xzCgAAAJdBcQoAAACXQXEKAAAAl0FxCgAAcI9ISEhQeHi4YmJi/nOdtWvXqlWrVqpataqaN2%2BuNWvW2C2fMWOGGjRooKpVq6pbt246fPiwoRkpTgHcM8aNG6d9%2B/ZxMRSAe9KWLVsUERGhY8eO/ec6R44cUWRkpAYPHqzff/9dkZGRGjJkiO2Wd0uXLtWnn36qWbNmKSYmRg8%2B%2BKCioqJk5PX1FKcAAADZ3NKlS/XCCy/o2Wefvel61atXV5MmTZQzZ0499thjqlGjhqKjoyVJixYtUufOnVWhQgV5e3vr%2Beef16lTp27Yib1VFKcAAAAuLi4uTrt27bJ7XX8IiCPq1aunH374Ict7Of/TwYMHFRgYaDdWvnx57d27N8vlnp6eKlu2rG25EbgJP5zjLk%2BJKVdOOnBAqlBB%2Busvs9Pc2D8ejQkAuE1m/n26A987etIkTZkyxW5s0KBBioyMdGj7IkWKOLTe5cuX5evrazfm4%2BOjpKQkh5YbgeIU9wY/PylHjmv/hHEsFonneBiH%2BTQec2o85tQUERERCgsLsxtztOC8Fb6%2BvkpJSbEbS0lJUe7cuR1abgSKUwAAACN5GH/WpL%2B/v/z9/Q3f778FBgZq165ddmMHDx5U5cqVJUkVKlTQgQMH1KhRI0lSWlqajhw5kulUAGdwzikAAAAkSa1bt9amTZu0cuVKpaena%2BXKldq0aZPatGkjSWrfvr3mz5%2BvvXv36sqVK3rvvfdUuHBhVa9e3bAMdE4BAACMdAc6p3dScHCw3nrrLbVu3VoBAQGaOnWqxo8fr9dee00lS5bU5MmTVa5cOUlShw4ddOnSJT3zzDNKSEjQQw89pI8%2B%2Bkienp6G5bFYjbwxFe497nJBVHCwtHWrFBIibdtmdpobc6cLojj3zFjMp/GYU%2BO5y5ya%2BffJwELNJi3N%2BH26KDqnAAAARnKzzqmroTgFAAAwEsWpU5g9AAAAuAw6pwAAAEaic%2BoUilMAAAAjUZw6hdkDAACAy6BzCgAAYCQ6p05h9gAAAOAy6JwCAAAYic6pUyhOAQAAjERx6hRmDwAAAC6DzikAAICR6Jw6hdkDAACAy6BzCgAAYCQ6p06hOAUAADASxalTmD0AAAC4DDqnAAAARqJz6hRmDwAAAC6DzikAAICR6Jw6heIUAADASBSnTmH2AAAA4DLonAIAABiJzqlTmD0AAAC4DDqnAAAARqJz6hSKUwAAACNRnDqF2QMAAIDLoHMKAABgJDqnTmH2AAAA4DLonAIAABiJzqlTKE4BAACMRHHqFGYPAAAALoPO6R125coVnT9/XsWKFTM7CgAAuBvonDqF2bvDOnfurN9%2B%2B%2B2O7f%2BNN97QG2%2B8ccf2DwAAcDfROb3Dzp8/f0f3P2LEiDu6fwAAcIvonDqF2buDevXqpVOnTmn48OF67LHHNGHCBLvlTzzxhGbOnHnT/cTGxqpPnz6qWbOmGjRooEGDBikuLk6SNHToUA0dOlSS1Lp1awUHB9telStXVqVKlZSamiqr1ap58%2BapWbNmql69ujp37qydO3ca/6YBALjXeXgY/7qH0Dm9g2bPnq2wsDANGjRIHh4eev/99zVkyBB5eHjo0KFD2rNnj6ZPn37T/UyYMEHFihXT9OnTdeXKFUVFRenjjz/WsGHD7NZbtmyZ7d9PnDihTp06acCAAfLy8tKCBQs0Z84cTZ8%2BXQEBAfr666/11FNP6dtvv1XhwoUdej9xcXE6e/as3ViRcuXk7%2Bfn0PamCgqy/ycAAHBJFKd3yaOPPqrRo0crJiZGtWvX1pIlS/TII484VBh6e3tr8%2BbNWrFihWrXrq2ZM2fK4wb/F3X%2B/Hn16dNHLVu2VJcuXSRJCxYs0NNPP62g/xVnHTp00OLFi7Vs2TL16tXLofcQHR2tKVOm2I0NGjxYkYMHO7S9S1i40OwE2Y/FYnaC7IX5NB5zajzm9MbusU6n0ShO7xIfHx%2B1atVKX331lWrWrKlly5Zp5MiRDm07bNgwffTRR5o1a5aGDh2qoKAgDRs2TNWrV8%2B0bkpKigYMGKAKFSropZdeso2fPHlSb7/9tsaPH28bS09PV%2BXKlR1%2BDxEREQoLC7MbK9KqlfTJJw7vwzRBQdcK086dpb17zU5zY1u2mJ3AcRaLZLWanSL7YD6Nx5waz13mlALabVGc3kUdO3bUk08%2BqfDwcFksFtWvX9%2Bh7Xbv3q2IiAhFRkYqISFBU6dO1aBBg7Rx40a79TIyMvT8888rIyND7777rl13tVixYoqKilKLFi1sY8eOHZPfLRyS9/f3l7%2B/v/3gX385vL1L2LtX2rbN7BQAgOyMzqlTmL07zMvLS5cuXZIkBQUF6f7779eYMWP0%2BOOPK0eOHA7t48MPP9TIkSOVmJiofPnyydfXVwUKFMi03siRI3Xw4EF9%2BOGH8vHxsVvWsWNHTZ8%2BXYcOHZIkrV%2B/Xi1atNDmzZudfIcAAMAOF0Q5hc7pHdahQwdNnDhRf/75p8aPH6%2BOHTtq%2BPDh6tChg8P7GDFihN566y01btxYqampqly5sj744AO7dU6dOqWFCxcqT548atasmdLT023LZsyYoZ49e8pqtWrgwIGKi4tT0aJF9cYbb6hx48aGvVcAAABnWaxWdzhxBC7LXc7pCQ6Wtm6VQkJc/7B%2BRobZCRznLueeuQvm03jMqfHcZU7N/PtUu7bx%2B9ywwfh9uqh7q08MAAAAl8ZhfZOtWrXKdhP9rFSrVs2hG/UDAAAX4YLniMbHx%2Bv111/Xpk2blCNHDrVu3Vovv/yycua0LwX79OmjLf%2B6a0xSUpIiIiI0YsQIZWRkqFq1arJarbL8ozv966%2B/KleuXIZkpTg1WbNmzdSsWTOzYwAAAKO4YHE6ZMgQFS1aVOvXr9e5c%2Bc0YMAAzZ07V3369LFb798NscWLF2vKlCkaNGiQJOngwYNKS0vT1q1b5eXldUeyut7sAQAAwDBHjx7Vpk2b9OKLL8rX11elS5fWwIEDtWDBghtud/jwYY0cOVLjx4%2B33Uryzz//VMWKFe9YYSrROQUAADDWHeicZvkI8SJFMt9/PAsHDhyQn5%2BfihYtahsLCAjQqVOndPHiReXLly/L7d566y21bdvW7qE/f/75p65cuaL27dvr5MmTCggI0PPPP6%2BQkJDbfGeZUZwCAAAY6Q4Up1k%2BQnzQIEVGRt5028uXL8vX19du7PrXSUlJWRanv//%2Bu3bs2GH3ZEnp2hMvH374YQ0ePFj58%2BfXggUL1Lt3by1btkylS5e%2B1beVJYpTAAAAF5flI8SLFHFo21y5cik5Odlu7PrXuXPnznKb6OhoNW/ePNP3%2BPdF3L1799aSJUu0du1ade3a1aE8N0NxCgAAYKQ70DnN8hHiDqpQoYIuXLigc%2BfOqXDhwpKkQ4cOqVixYsqbN2%2Bm9dPT0/Xjjz9q6tSpmZZNnDhRzZo1U6VKlWxjqamp8vb2vq1sWeGCKAAAgGysbNmyqlatmsaMGaPExEQdP35c06ZN%2B8%2BnVe7bt09XrlzJ8jzS/fv3a/To0Tp79qxSU1M1ZcoUJSYmKjw83LC8FKcAAABG8vAw/uWkSZMmKT09XY0bN1bHjh1Vv359DRw4UJIUHBysZcuW2dY9fvy48ufPn2U3dOzYsSpTpozatGmj0NBQbdq0SXPmzJGfn5/TGa/j8aVwDo8vNR6PL713MZ/GY06N5y5zaubfpztx//JVq4zfp4uicwoAAACXwQVRAAAARnLBJ0S5E2YPAAAALoPOKQAAgJHonDqF4hQAAMBIFKdOYfYAAADgMuicAgAAGInOqVOYPQAAALgMOqcAAABGonPqFIpTAAAAI1GcOoXZAwAAgMugcwoAAGAkOqdOYfYAAADgMuicAgAAGInOqVMoTgEAAIxEceoUZg8AAAAug84pAACAkeicOoXZAwAAgMugcwoAAGAkOqdOoTgFAAAwEsWpU5g9AAAAuAw6pwAAAEaic%2BoUZg8AAAAug84pAACAkeicOoXiFAAAwEgUp05h9gAAAOAy6JzCORkZZie4NVu2mJ3g5tzl/7iDg6WtW6Vq1aRt28xO89/c7TPqBqyymB3BYRa5V1534C5zampCd/k97qKYPQAAALgMOqcAAABGonPqFIpTAAAAI1GcOoXZAwAAgMugcwoAAGAkOqdOYfYAAADgMuicAgAAGInOqVMoTgEAAIxEceoUZg8AAAAug84pAACAkeicOoXiFAAAwEgUp05h9gAAAOAy6JwCAAAYic6pU5g9AAAAuAw6pwAAAEaic%2BoUilMAAAAjUZw6hdkDAADI5uLj4zVw4EBVr15doaGhGj16tNLT07Nct0%2BfPnrooYcUHBxse61bt862fMaMGWrQoIGqVq2qbt266fDhw4ZmpTgFAAAwkoeH8S8nDRkyRLly5dL69eu1ePFibdiwQXPnzs1y3Z07d2rWrFnatm2b7dWgQQNJ0tKlS/Xpp59q1qxZiomJ0YMPPqioqChZrVanM15HcQoAAJCNHT16VJs2bdKLL74oX19flS5dWgMHDtSCBQsyrXv8%2BHH9/fffqlSpUpb7WrRokTp37qwKFSrI29tbzz//vE6dOqWYmBjD8lKcAgAAGOkOdE7j4uK0a9cuu1dcXJxDcQ4cOCA/Pz8VLVrUNhYQEKBTp07p4sWLduv%2B%2Beefyp07t5599lnVqlVLLVu21OLFi23LDx48qMDAQNvXnp6eKlu2rPbu3evkpP0/LogCAAAw0h24ICo6OlpTpkyxGxs0aJAiIyNvuu3ly5fl6%2BtrN3b966SkJOXLl882npqaqqpVq%2BrZZ59VhQoVFBMTo8jISOXOnVvNmzfPcl8%2BPj5KSkq63beWCcUpAACAi4uIiFBYWJjdWJEiRRzaNleuXEpOTrYbu/517ty57cbbtm2rtm3b2r6uV6%2Be2rZtq2%2B//VbNmzeXr6%2BvUlJS7LZJSUnJtB9nUJwCAAAY6Q50Tv39/eXv739b21aoUEEXLlzQuXPnVLhwYUnSoUOHVKxYMeXNm9du3cWLF9u6pNelpqbK29vbtq8DBw6oUaNGkqS0tDQdOXLE7lC/szjnFAAAIBsrW7asqlWrpjFjxigxMVHHjx/XtGnT1KFDh0zrJiYmauTIkdq9e7cyMjL0888/65tvvlFERIQkqX379po/f7727t2rK1eu6L333lPhwoVVvXp1w/LSOQUAADCSC96Ef9KkSRoxYoQaN24sDw8PtW3bVgMHDpQkBQcH66233lLr1q3Vo0cPJSUladCgQYqPj1fp0qX19ttv24rPDh066NKlS3rmmWeUkJCghx56SB999JE8PT0Ny2qxGnljKtx73OnjY7G4R14X/KWWpeBgaetWKSRE2rbN7DT/LSPD7ASOc5PPqFUWsyM4zE2m1K24y5xazPyYjh1r/D5fecX4fbooN/krCAAAgHsBh/UBAACM5C5HwFwUswcAAACXQecUAADASHROnUJxCgAAYCSKU6cwewAAAHAZdE4BAACMROfUKcweAAAAXAadUwAAACPROXUKxSkAAICRKE6dwuwBAADAZdwzxemSJUsUFhZ2S9tUrFhRMTExdyiR806cOKGKFSvqxIkTWS6/nfcMAACc5OFh/Osecm%2B9WwAAALi0bFWcZtVJnDx5srp16yZJSk9P1/jx49WwYUOFhIRo2LBhSk9PlySlpaVp7NixCg0NVa1atTRz5sxb%2Bt67du1St27dFBwcrHr16umDDz6Q1WqVJC1evFjt2rVTaGiogoOD9fTTTyshIcGWb%2BDAgYqMjFTVqlUVFham6Oho234PHTqkp59%2BWg0bNtTDDz%2Bsxx57TGvWrLH73l999ZWaNGmiOnXqaNiwYUpMTLxhxho1aqhp06aaO3euLSMAADAInVOn3FMXRMXGxipfvnxavXq1jhw5og4dOqhWrVpq2bKlpk2bpp9//lmLFy9WoUKF9Oabbzq83wsXLqhXr17q1q2bZs2apTNnzqhbt24qWrSoKlWqpFGjRmnevHl6%2BOGHdebMGfXo0UPz5s3TkCFDJEk//vijhg4dqgkTJigmJkb9%2B/dXmTJlVLt2bUVGRqpx48aaMmWKrFarxo8frzfffFONGjWyff/ff/9dixYtUkZGhgYOHKgxY8ZozJgxmd57jx499Oyzz2r27Nk6evSoBg4cKB8fH3Xq1Mmh9xkXF6ezZ8/ajRUpXFj%2B/v4OzxUcEBxsdgLHBAXZ/xMAcM09Vkwa7Z4qTvPkyaO%2BffvKYrGofPnyCgoK0rFjxyRJX3/9tfr376/SpUtLkoYNG6Zly5Y5tN81a9bI29tbzzzzjCwWi8qUKaM5c%2BYoV65c8vPz0zfffKNSpUrp77//VlxcnAoWLKjY2Fjb9hUrVtRTTz0lSapXr56aNWumr7/%2BWrVr19ZHH32kokWLymq16uTJk8qXL5/dtpI0dOhQFSxYUJIUFRWlAQMGaNSoUXbrLFu2TAEBAerSpYskqXz58urdu7fmz5/vcHEaHR2tKVOm2I0NeuYZRUZFObS9S7BYzE5wc1u3mp3g1ixcaHaC7MUNPqOun9CeG0yp22FOcSfdU8Vp/vz5ZfnHf1Genp66evWqpGtdweLFi9uW5cuXT/nz53dov2fPnlXx4sXt9n3//fdLklJTUzVv3jwtX75cuXLlUsWKFZWYmGh3OL1s2bJ2%2BytevLj27NkjSdq7d68GDhyos2fPKiAgQAULFsx0KL5UqVJ226ampurChQt265w8eVK7du1S9erVbWMZGRnKkSOHQ%2B9RkiIiIjJdYFWkcGHJXU4NsFjcI2u1amYncExQ0LXCtHNnae9es9P8ty1bzE7gODf5jFrdqDx1kyl1K%2B4yp6YW0HROnZKtitPrhVZaWppt7Pz58w5tW6xYMR0/ftz2dVJSki5duuTwtqdPn5bVarUVqKtXr1ZiYqLi4uL066%2B/avny5SpcuLAkqX///nbb/7sTeuLECRUvXlyxsbEaPHiwpkyZYisKV61ape%2B//z7T9nny5LFtmytXLlsn9Z8ZQ0NDNWvWLNvY%2BfPndfnyZYfeoyT5%2B/tnPoTvDr%2Bh3M22bWYnuDV797pfZgCAy8pWpX2hQoWUP39%2BrVixQlarVbt27dJ3333n0LZPPPGEZs6cqUOHDunKlSsaN26crat6Mw0bNlR6ero%2B/PBDpaam6tixYxozZoyuXLmixMRE5cyZU56enkpPT9fXX3%2Bt9evX2xXQ27dv19dff62rV69q7dq1%2BvHHH9W%2BfXtdvnxZV69ela%2BvryTp4MGDmjp1qqRrHdnr3n33Xf399986c%2BaMPvjgA0VERGTK2KpVK23fvl3Lli1Tenq64uLi1L9/f40bN86h9wgAABzEBVFOyVadUy8vL40cOVKTJk3SrFmzVLlyZXXs2FFbHDis17dvXyUnJ6tr165KT09Xx44d5efn59D3zZcvn2bNmqWxY8dqzpw58vX1VZcuXRQREaELFy5o//79atSokby9vVWpUiV17txZGzdutG3/wAMP6Mcff9SoUaNUuHBhvfvuuwr%2B30UxL730kl588UUlJyerWLFi6tixo959913t37/fli84OFiPPvqoPDw81LJlSz377LOZMpYsWVIzZ87U%2BPHjNWrUKOXIkUMNGzbUa6%2B95tB7BAAADrrHikmjWazcS8hUkydP1qZNm/Tpp5%2BaHeX2uNPHx11OlHKXX2rBwdcu3goJce3D%2BhkZZidwnJt8Rjnn9N7mLnNq6jmnc%2BYYv8//XTh9L8hWnVMAAADTuUuTwUVRnDogNDTU7hzPf1uxYoVKlChxFxMBAABkTxSnDoiJiblj%2B46MjLxj%2BwYAACagc%2BoUilMAAAAjUZw6hdkDAACAy6BzCgAAYCQ6p05h9gAAAOAy6JwCAAAYic6pUyhOAQAAjERx6hRmDwAAAC6DzikAAICR6Jw6heIUAADASBSnTmH2AAAA4DLonAIAABiJzqlTmD0AAAC4DDqnAAAARqJz6hSKUwAAACNRnDqF2QMAAIDLoHMKAABgJDqnTmH2AAAA4DLonAIAABiJzqlTKE4BAACMRHHqFIpTAACAbC4%2BPl6vv/66Nm3apBw5cqh169Z6%2BeWXlTNn5lLws88%2B09y5cxUXFyd/f391795dXbp0kSRlZGSoWrVqslqtslgstm1%2B/fVX5cqVy5CsFKcAAABGcsHO6ZAhQ1S0aFGtX79e586d04ABAzR37lz16dPHbr3Vq1drwoQJmjFjhqpUqaLt27erX79%2BKly4sJo1a6aDBw8qLS1NW7dulZeX1x3J6nqzBwAAAMMcPXpUmzZt0osvvihfX1%2BVLl1aAwcO1IIFCzKtGxsbq759%2B6pq1aqyWCwKDg5WaGioNm/eLEn6888/VbFixTtWmEp0TgEAAIx1BzqncXFxOnv2rN1YkSJF5O/vf9NtDxw4ID8/PxUtWtQ2FhAQoFOnTunixYvKly%2Bfbfz64fvr4uPjtXnzZr3yyiuSrhWnV65cUfv27XXy5EkFBATo%2BeefV0hIiDNvzw7FKQAAgJHuQHEaHR2tKVOm2I0NGjRIkZGRN9328uXL8vX1tRu7/nVSUpJdcfpPZ8%2Be1dNPP63KlSurZcuWkiQfHx89/PDDGjx4sPLnz68FCxaod%2B/eWrZsmUqXLn07by0TilMAAAAXFxERobCwMLuxIkWKOLRtrly5lJycbDd2/evcuXNnuc327ds1ePBgVa9eXWPHjrVdODV06FC79Xr37q0lS5Zo7dq16tq1q0N5bobiFAAAwEh3oHPq7%2B/v0CH8rFSoUEEXLlzQuXPnVLhwYUnSoUOHVKxYMeXNmzfT%2BosXL9aoUaMUFRWlXr162S2bOHGimjVrpkqVKtnGUlNT5e3tfVvZssIFUQAAANlY2bJlVa1aNY0ZM0aJiYk6fvy4pk2bpg4dOmRad9WqVXrzzTc1efLkTIWpJO3fv1%2BjR4/W2bNnlZqaqilTpigxMVHh4eGG5aU4BQAAMJKHh/EvJ02aNEnp6elq3LixOnbsqPr162vgwIGSpODgYC1btkySNGXKFF29elVRUVEKDg62vd544w1J0tixY1WmTBm1adNGoaGh2rRpk%2BbMmSM/Pz%2BnM15nsVqtVsP2hnuPO318LBb3yOuC98fLUnCwtHWrFBIibdtmdpr/lpFhdgLHucln1CrLzVdyEW4ypW7FXebUYubHdMMG4/dZu7bx%2B3RRbvJXEAAAAPcCLogCAAAwkrscAXNRzB4AAABcBp1TAAAAI9E5dQrFKeBq3OkCHknassXsBDfmLn8krl9gVq2aa19gJkkZbnA1jJuxyJ3m1OImeU28Ispdfu%2B4KGYPAAAALoPOKQAAgJHonDqF2QMAAIDLoHMKAABgJDqnTqE4BQAAMBLFqVOYPQAAALgMOqcAAABGonPqFGYPAAAALoPOKQAAgJHonDqF4hQAAMBIFKdOYfYAAADgMuicAgAAGInOqVOYPQAAALgMOqcAAABGonPqFIpTAAAAI1GcOoXZAwAAgMugcwoAAGAkOqdOoTgFAAAwEsWpU5g9AAAAuAw6pwAAAEaic%2BoUZg8AAAAug84pAACAkeicOoXiFAAAwEgUp05h9gAAAOAy6JwCAAAYic6pU5g9AAAAuAw6pwAAAEaic%2BoUilMAAAAjUZw6hdkDAACAy6BzCgAAYCQ6p05h9gAAAOAy6JwCAAAYic6pUyhOAQAAjERx6hRmDwAAAC6DzikAAICR6Jw6hdkDAACAy6BzCgAAYCQ6p065pdm7cuWKzpw549C6R44cuZ08N3X16lUdP378juzbDJcuXVJCQoLZMQAAgFE8PIx/3UNu6d127txZv/32203X2717t1q2bOnwfsPCwrRkyRKH1n322Wf11VdfSZJOnTql4OBgnTp1yuHv5WrCw8N14MCBm643efJkdevWzbDvu2TJEoWFhRm2PwAA4Lri4%2BM1cOBAVa9eXaGhoRo9erTS09OzXHft2rVq1aqVqlatqubNm2vNmjV2y2fMmKEGDRqoatWq6tatmw4fPmxo1lsqTs%2BfP%2B/QepcuXVJaWtptBbqVDCVKlNC2bdtUokSJO/K97gZH5xQAALgJF%2BycDhkyRLly5dL69eu1ePFibdiwQXPnzs203pEjRxQZGanBgwfr999/V2RkpIYMGaLY2FhJ0tKlS/Xpp59q1qxZiomJ0YMPPqioqChZrVanM17n8Lvt1auXTp06peHDh2vEiBH6/fff1aVLF1WvXl1hYWF6//33lZqaquPHj6tv376SpODgYG3btk2JiYkaNmyYmjZtqqpVq6p%2B/fr68MMPbznsa6%2B9pt9//10fffSR%2BvfvrxMnTqhixYo6ceKEJKlixYqKjo5Ws2bNVKVKFfXv3187d%2B5Up06dFBwcrPbt2%2Bvo0aO2/a1YsUKtWrVStWrV1K5dO/3yyy8O5YiKitLo0aNtXw8dOlR169a1/WDWrFmjRo0aSZK2bt2q7t27q169enrooYfUrl07bd%2B%2BXZLUrFkzSVLfvn01Y8YMSdLy5cvVsmVLBQcHq3nz5lq5cqXt%2B1y%2BfFnDhg1TvXr1FBoaqokTJ9qWpaam6oMPPlDjxo1Vs2ZN9e3b1%2B69Hjp0SN26dVNwcLBatWql3bt3OzzvAADAfR09elSbNm3Siy%2B%2BKF9fX5UuXVoDBw7UggULMq27dOlSVa9eXU2aNFHOnDn12GOPqUaNGoqOjpYkLVq0SJ07d1aFChXk7e2t559/XqdOnVJMTIxheR2%2BIGr27NkKCwvToEGDVLVqVbVp00YvvPCC5syZo9OnTysyMtJWhM6YMUPdu3fXtm3bJElvvvmmTpw4ocWLFytv3rz6/vvvFRUVpebNm%2Bu%2B%2B%2B5zOOzo0aN17Ngx1axZU5GRkbai9J%2BWL1%2Bu6OhopaamqkWLFho4cKDmzJmj4sWLq3fv3vrwww81duxYrV27VsOHD9f06dMVEhKidevWKTIyUosWLVKFChVumKNJkyaaOnWqXnvtNUnSL7/8osTERO3bt09BQUH66aef1KRJE6WkpGjAgAGKiorSk08%2BqZSUFL366qt65513tHDhQq1atUoVK1bUjBkzFBoaqpiYGL366quaMmWK6tevr19%2B%2BUUDBw5UYGCgpGunS/To0UMjR45UTEyMevbsqYYNGyo4OFgTJ07Uxo0bNXfuXPn7%2B2vGjBnq1auXVq5cKQ8PDz399NNq0KCBZs6cqWPHjqlv377yuMX/E4uLi9PZs2ftxooULix/f/9b2g9wVwUHm53AMUFB9v8E4L7uwDmiWf4NLlLEob/BBw4ckJ%2Bfn4oWLWobCwgI0KlTp3Tx4kXly5fPNn7w4EFb3XFd%2BfLltXfvXtvy601ISfL09FTZsmW1d%2B9e1apV67be27/d1tX6y5cvV8WKFdWjRw9J0n333afnn39eUVFRevXVVzOtHxkZqRw5cihPnjw6c%2BaMvL29JV2b6FspTh3RtWtX%2Bfn5SZIqVKigSpUqKSAgQJJUq1YtbdmyRZI0f/58Pfnkk6pRo4YkqVGjRgoLC9Pnn3%2Bu119//Ybfo2HDhnr11Vd1/PhxXb58WT4%2BPnr44Ye1YcMGVaxYUWvWrNHEiRPl6emp6Oho3Xfffbpy5YpOnjwpPz8//fnnn1nu96uvvlLTpk31yCOPSJIaNGighQsX2j5MFSpUUJs2bWzvpXDhwjp27JiqVq2qzz//XJMmTVLp0qUlSc8884wWLVqkn3/%2BWQUKFNDp06f10ksvydvbWxUqVNBTTz2lTz755JbmNjo6WlOmTLEbG/TMM4qMirql/ZjKYjE7Qfbj6nO6davZCW7NwoVmJ7gpF/%2BJZ%2BLqH9Fr3CLk/3OPSTWN9Q78PLP8GzxokCIjI2%2B67eXLl%2BXr62s3dv3rpKQku%2BI0q3V9fHyUlJTk0HIj3FZxGh8fbyuCritVqpRSUlIUHx%2Bf5fqjR4/W7t27VapUKVWuXFmSlJGRcTvf/oauF6aSlCNHDuXPn9/2tYeHh%2B3Q%2B8mTJ7Vp0yZ99tlntuVXr151qOrPly%2BfatasqXXr1ikpKUl16tRRQECAfvnlF4WEhMhqtapatWry8PBQTEyM%2Bvbtq6SkJJUvX145c%2Bb8z/My4uLiVKlSJbuxhx9%2BOMv3JkleXl66evWqEhISlJSUpMGDB9t1Q9PS0nTy5EmlpqaqQIEC8vHxsS0rU6bMTd/nv0VERGS6iKpI4cKSgeeZ3FEWi/tkdRfuMKfVqpmdwDFBQdcK086dpf91KFyVdYv7FPzu8BGVJIvcIOR1bjOp2auAzvJvcJEiDm2bK1cuJScn241d/zp37tx2476%2BvkpJSbEbS0lJsa13s%2BVGuK3itGTJkvr%2B%2B%2B/txo4dOyYvLy%2B7YvC6wYMHKywsTLNmzVLOnDl1/vx5LVq06PYS34TFwQ9jsWLF1LZtW/Xr1882durUKbsC7kYaN26sdevWKS0tTZ06dVJAQIDef/99rVq1So0bN5aHh4d27NihkSNH6vPPP7cV5LNnz9Zff/2V5T6LFy%2Be6c4Ds2fPVtWqVW%2BYpUCBAvL29s607uHDh1W0aFHt2bNHCQkJunz5su3D4%2Bgtwf7J398/8%2BEDd/gFhXvb/04vcht797pfZgB27kDvLeu/wQ6qUKGCLly4oHPnzqlw4cKSrl2LUqxYMeXNm9du3cDAQO3atctu7ODBg7Y6pkKFCjpw4IDt2pq0tDQdOXIk06kAzrilkyK8vLx06dIltWjRQocOHdInn3yi1NRUHTt2TBMmTFCrVq3k5eVlO2x/6dIl2z99fHyUI0cOJSQkaNSoUbY3dKuuZ3BWx44dNW/ePP3xxx%2BSpD///FPt2rXTN99849D2TZo00aZNm7R9%2B3bVqlVLAQEB8vPz08KFCxUeHi7p2vv28PCwFbzbt2/XvHnzlJqamuX7efzxx/XDDz/ol19%2BUUZGhtavX6/Jkydn%2BuD8m4eHhzp06KD33ntPZ86cUUZGhpYuXaqWLVvq6NGjCg4OVrly5TRq1CglJyfr6NGjmj179i3PGQAAcD9ly5ZVtWrVNGbMGCUmJur48eOaNm2aOnTokGnd1q1ba9OmTVq5cqXS09O1cuVKbdq0yXZaYfv27TV//nzt3btXV65c0XvvvafChQurevXqhuW9peK0Q4cOmjhxot5//33NnDlTq1atUp06ddS5c2fVrVtXb7zxhqRrVXe1atVUv359rV27VmPHjtXKlSsVEhKidu3aqWjRoqpUqZL2799/y4Hbtm2rL7/8Up07d77lbf/p0Ucf1XPPPadXX31VISEhGjx4sHr27OnwvUSLFi2qChUqKDAw0HauRt26deXp6Wk7NaBu3brq3LmzunTpoho1auitt95St27dlJCQoHPnzkm61qZ//vnnNXHiRFWrVk1vv/223n77bVWvXl3vvPOOJkyYcNMLtCTp5ZdfVpUqVdS5c2dVr15dc%2BfO1aRJk1SpUiXlyJFDH3/8seLi4lSnTh316dNHjRs3vs2ZAwAAN5KRYfzLWZMmTVJ6eroaN26sjh07qn79%2Bho4cKCka3dXWrZsmaRrF0pNnTpVH330kWrUqKFp06Zp8uTJKleunKRrtWDPnj31zDPPqFatWtq9e7c%2B%2BugjeXp6Oh/yfyxWI29MhXuPO3183OU8KXfiDnPqLk9WCQ6%2BdvFWSIjLH9a3Zrj4z/wf3OEjKnHO6R1h4jmnV64Yv8//HZS%2BJ7jJb20AAADcC27rgqg7pV27dv95sZB07XFZRp7T4Oo5AACA%2B7kTF0TdSzisD%2Be408fHXQ5FuRN3mFMO6xuOw/rG47D%2BHWDiYf1/3bXJEP%2B6tWi25lKdUwAAAHdH59Q5FKcAAAAGojh1jpsc7wIAAMC9gM4pAACAgeicOofOKQAAAFwGnVMAAAAD0Tl1DsUpAACAgShOncNhfQAAALgMOqcAAAAGonPqHIpTAAAAA1GcOofD%2BgAAAHAZdE4BAAAMROfUOXROAQAA4DLonAIAABiIzqlzKE4BAAAMRHHqHA7rAwAAwGXQOQUAADAQnVPn0DkFAACAy6BzCgAAYCA6p86hOAUAADAQxalzOKwPAAAAl0HnFAAAwEB0Tp1D5xQAAAAug84pAACAgeicOofiFAAAwEAUp87hsD4AAABcBp1TAAAAA9E5dQ6dUwAAALgMOqcAAAAGonPqHIpTAAAAA1GcOofD%2BgAAAHAZdE4BAAAMROfUOXROAQAA4DLonAIuxiqL2REcZpEb5M2wmp3AYRZJ1i1bzY5xUxYPF/%2BZXxccLG3dKku1EGnbNrPT3FByknt8Ti0WycdHSrlikdXFI/v6mve96Zw6h%2BIUAADAQBSnzuGwPgAAAFwGnVMAAAAD0Tl1Dp1TAAAAuAw6pwAAAAaic%2BocilMAAAADUZw6h%2BIUAADgHpaUlKSRI0fqp59%2BUnp6uho3bqzhw4crd%2B7cWa6/atUqTZs2TcePH5efn5/atWungQMHysPj2tmizZs316lTp2xfS9LixYsVEBDgUB6KUwAAAAO5W%2Bd05MiROn36tFatWqWrV69qyJAhGj9%2BvIYPH55p3Z07d%2Bqll17S%2B%2B%2B/r0ceeUR//fWX%2Bvbtq1y5cqlXr15KTEzUX3/9pR9//FElS5a8rTxcEAUAAHCPSk5O1vLlyxUVFSU/Pz8VKlRIL7zwgpYsWaLk5ORM6588eVKdOnVSo0aN5OHhoYCAAIWHh2vz5s2SrhWvfn5%2Bt12YSnROAQAADHUnOqdxcXE6e/as3ViRIkXk7%2B9/021TUlIUGxub5bLk5GSlpaUpMDDQNhYQEKCUlBQdOXJEDzzwgN36zZo1U7Nmzez2/fPPP6tVq1aSpD///FO%2Bvr7q2rWrDhw4oJIlSyoyMlKNGjVy%2BL1SnAIAABjoThSn0dHRmjJlit3YoEGDFBkZedNtd%2BzYoe7du2e5bPDgwZKkXLly2cZ8//fs18uXL99wv4mJiRo8eLB8fHzUs2dPSZLFYtFDDz2k5557TiVKlNB3332nyMhIzZ8/X1WrVr1pVoniFAAAwOVFREQoLCzMbqxIkSIObRsaGqp9%2B/ZluWz37t364IMPlJycbLsA6vrh/Dx58vznPg8fPqyoqCgVKlRI8%2BbNs63bp08fu/Vat26tb775RqtWraI4BQAAMMOd6Jz6%2B/s7dAj/VpUrV06enp46ePCgqlSpIkk6dOiQPD09VbZs2Sy3Wbt2rZ577jl17NhRzz//vHLm/P9yctasWapUqZJq165tG0tNTZW3t7fDmbggCgAA4B7l6%2Bur5s2ba/z48UpISFBCQoLGjx%2Bvli1bysfHJ9P627dv1zPPPKNXXnlFL7/8sl1hKkmnT5/WW2%2B9pePHjys9PV2LFy/Wtm3b9PjjjzucyWK1Wq1OvzPcu9zp42OxuEVeqyxmR3CYm0yp23CX%2BbR4uMlnNDhY2rpVCgmRtm0zO80NJSe5wQ9e1z6jPj5SSorrf1b/d9qkKX74wfh9hocbv8/rEhMT9fbbb%2Bunn35SWlqaGjdurNdff912HmqLFi3UqlUr9e/fX/3799fPP/9sOy/1umrVqmnmzJlKTU3V%2BPHj9e233%2BrSpUsqX768XnzxRYWGhjqch%2BIUznGnj4%2Bb/OWnOL13uct8Upwaj%2BLUeGYWp6tWGb/Pf1wgn%2B1xWB8AAAAugwuiAAAADORuT4hyNRSnAAAABqI4dQ6H9QEAAOAy6JwCAAAYiM6pc%2BicAgAAwGXQOQUAADAQnVPnUJwCAAAYiOLUORzWBwAAgMugcwoAAGAgOqfOoXMKAAAAl0HnFAAAwEB0Tp1DcQoAAGAgilPncFgfAAAALoPOKQAAgIHonDqHzikAAABcBp1TAAAAA9E5dQ7FKQAAgIEoTp3DYX0AAAC4DDqnAAAABqJz6hw6pyaLi4tTUlKS2TEAAABcAsWpic6dO6dmzZopISHhpusOHTpUQ4cONex7T548Wd26dTNsfwAA4JqMDONf9xIO65soJSWFrikAANnMvVZMGo3O6W1q166d5s6da/u6W7dueuKJJ2xfz58/X126dNFPP/2kTp06qXbt2qpSpYq6du2qI0eO6OrVq2rZsqUkqWXLllq5cqUk6ZNPPlF4eLiCg4PVrl07bdiwwbbP%2BPh4RUVFKTQ0VPXq1dP8%2BfNtyxITEzVixAg98sgjql27tp599lmdO3fOtnzr1q1q3769qlatqk6dOunEiRN3amoAAABuG53T2xQeHq7169erZ8%2Beunz5snbu3Km0tDRdvHhR%2BfLl008//aS6detq8ODB%2BuCDDxQWFqbz589r0KBBmjp1qt5991198803aty4sb755huVKlVKS5Ys0bRp0/Thhx%2BqSpUq%2BvLLLzVgwAD9/PPPkqSNGzfqo48%2B0gcffKCvvvpKr7zyisLDw1W0aFG9%2Buqrunz5spYsWSIfHx%2BNGzdOgwYN0meffaYLFy7o6aefVt%2B%2BffXUU0/pjz/%2BUL9%2B/VSpUqVbes9xcXE6e/as3ViRwoXl7%2B9v1LQCcAfBwWYncExQkP0/XZjFYnYCx1zP6S55zULn1DkUp7epSZMmmjZtmpKTk7Vx40Y9/PDDunDhgjZu3Kg6depo06ZNGjVqlFq2bKkyZcooMTFRZ86cUYECBRQbG5vlPpcuXaqIiAgF/%2B8X/xNPPKGAgAD5%2BPhIkurWras6depIklq0aKGhQ4fq%2BPHjypkzp1atWqVvv/1WhQoVkiS9%2Buqrql69unbt2qUDBw7I19dXffv2lcViUbVq1dS%2BfXvt2bPnlt5zdHS0pkyZYjc26JlnFBkVdUv7MZUb/EZ1/YT23GBK3YpbzOfWrWYnuDULF5qd4KZ8zA5wi7y9zU6A7Izi9DZVqFBBJUqUUExMjNavX6%2B6devq3Llz%2Bu2335Senq6KFSuqePHimj59uj7//HNZLBYFBgYqMTFROXNmPe1nz55ViRIl7MZCQkJs/%2B7n52f7dy8vL0nS1atXdfLkSUlSx44d7bbNkSOHTpw4odjYWBUvXlyWf/zVK1OmzC0XpxEREQoLC7MbK1K4sGS13tJ%2BTGOxuEVWqxuVp24ypW7DXebTUi3k5iu5gqCga4Vp587S3r1mp7mhlN/co%2BC3WK4VpleuuP5n1cfEip/OqXMoTp3QuHFjrVu3Ths2bNCECRMUHx%2Bv0aNHKzExUU2bNtW3336r%2BfPn67PPPtN9990nSRo5cqT279%2Bf5f6KFy%2Bu06dP241NnDhRrVu3vmGOokWLSpK%2B/fZbFSlSxDZ%2B8OBBlS5dWt9%2B%2B61OnjypjIwMeXhcO834zJkzt/x%2B/f39Mx/Cd/XfTgCMt22b2Qluzd69Lp/Z3X6VWq3ul/luojh1DhdEOSE8PFwrV67UxYsXValSJdWsWVOnTp3S6tWrFR4erkuXLsnDw0M%2BPj6yWq1at26dvvrqK6WlpUmSvP93XCQxMVHStYusoqOj9ccffygjI0NffvmlFixYoAIFCtwwR9GiRdWwYUONHj1a58%2BfV1pamqZPn64OHTro4sWLCgsLk9Vq1eTJk5WamqqdO3fqiy%2B%2BuLOTAwAAcBvonDqhatWqypkzp0JDQ2WxWOTj46Pq1asrLi5O999/v0qVKqUtW7aoRYsWypEjh%2B6//3716NFDCxYsUGpqqgoXLqzw8HBFRERo6NChevLJJ3Xx4kW9%2BOKLOnv2rMqXL68ZM2aoYMGCN83yzjvv6L333lPbtm2VmJioChUqaObMmbZO6qxZs/Tmm29qzpw5uu%2B%2B%2B9SsWTP99ddfd3qKAAC459A5dY7FaqUxDye408fHTU7o45zTe5e7zKfFw00%2Bo8HB1y7eCglx%2BcP6yUlu8IPXtc%2Boj4%2BUkuL6n1VfX/O%2B99Spxu/zmWeM36eronMKAABgIDqnzqE4BQAAMBDFqXO4IAoAAAAug84pAACAgeicOofOKQAAAFwGnVMAAAAD0Tl1DsUpAACAgShOncNhfQAAALgMOqcAAAAGonPqHDqnAAAAcBl0TgEAAAxE59Q5dE4BAAAMlJFh/OtOSkpK0iuvvKLQ0FBVq1ZNL730ki5fvvyf6w8fPlyVK1dWcHCw7RUdHW1bvnTpUoWHh6tq1apq166dtm3bdkt5KE4BAADuYSNHjtTp06e1atUqff/99zp9%2BrTGjx//n%2Bv/%2BeefGjlypLZt22Z7RURESJJiYmI0cuRIjRs3Tps3b1br1q01YMAAJScnO5yH4hQAAMBA7tQ5TU5O1vLlyxUVFSU/Pz8VKlRIL7zwgpYsWZJlQZmamqr9%2B/ercuXKWe7viy%2B%2BUIsWLVStWjV5enqqZ8%2BeKlCggFauXOlwJs45BQAAyMZSUlIUGxub5bLk5GSlpaUpMDDQNhYQEKCUlBQdOXJEDzzwgN36e/fuVXp6uiZNmqQtW7Yob968at%2B%2Bvfr06SMPDw8dPHhQ7du3t9umfPny2rt3r8N5KU4BAAAMdCc6nXFxcTp79qzdWJEiReTv73/TbXfs2KHu3btnuWzw4MGSpFy5ctnGfH19JSnL804vXbqkmjVrqlu3bpowYYL27NmjZ555Rh4eHurTp48uX75s2/46Hx8fJSUl3TTndRSnAAAABroTxWl0dLSmTJliNzZo0CBFRkbedNvQ0FDt27cvy2W7d%2B/WBx98oOTkZOXOnVuSbIfz8%2BTJk2n9unXrqm7duravH374YfXo0UMrV65Unz595Ovrq5SUFLttUlJSVKBAgZvmvI7iFAAAwMVFREQoLCzMbqxIkSJO77dcuXLy9PTUwYMHVaVKFUnSoUOH5OnpqbJly2Zaf/Xq1Tp37pw6depkG0tNTZWPj48kqUKFCjpw4IDdNgcPHlSDBg0czsQFUQAAAAa6ExdE%2Bfv768EHH7R7OXJI/2Z8fX3VvHlzjR8/XgkJCUpISND48ePVsmVLW8H5T1arVWPHjtWGDRtktVq1bds2zZs3z3a1focOHbR8%2BXJt3LhRaWlpmjt3ruLj4xUeHu5wJjqnAAAABnK3m/APHz5cb7/9tlq1aqW0tDQ1btxYr7/%2Bum15ixYt1KpVK/Xv31/h4eF65ZVX9Oabbyo2NlaFCxdWZGSk2rRpI0mqXbu2hg8fbltevnx5zZgxQ35%2Bfg7nsVitVqvRbxL3EHf6%2BFgsbpHXKovZERzmJlPqNtxlPi0ebvIZDQ6Wtm6VQkKkW7wJ%2BN2WnOQGP3hd%2B4z6%2BEgpKa7/Wf3XNTl31WuvGb/P0aON36eronMKAABgIHfrnLoazjkFAACAy6BzCgAAYCA6p86hOAUAADAQxalzOKwPAAAAl0HnFAAAwEB0Tp1D5xQAAAAug84pAACAgeicOofiFAAAwEAUp87hsD4AAABcBp1TAAAAA9E5dQ6dUwAAALgMOqcAAAAGonPqHIpTAAAAA1GcOofD%2BgAAAHAZdE4BAAAMROfUORSnALI1i6xmR7gFFrfIm5zk%2BhklyWKRfCSl/LZVVheP7JvLYnYExwQHS1u3yqdOiLRtm9lpbszVf%2Bj4TxSnAAAABqJz6hyKUwAAAANRnDqHC6IAAADgMuicAgAAGIjOqXPonAIAAMBl0DkFAAAwEJ1T51CcAgAAGIji1Dkc1gcAAIDLoHMKAABgIDqnzqFzCgAAAJdB5xQAAMBAdE6dQ3EKAABgIIpT53BYHwAAAC6DzikAAICB6Jw6h84pAAAAXAadUwAAAAPROXUOxSkAAICBKE6dw2F9AAAAuAw6pwAAAAaic%2BocOqcAAABwGXROAQAADETn1DkUpwAAAAaiOHUOh/UBAADgMuicAgAAGIjOqXMoTgEAAAxEceocDusDAADAZdA5BQAAMJC7dU6TkpI0cuRI/fTTT0pPT1fjxo01fPhw5c6dO9O6b7zxhpYvX243lpKSojp16mjWrFmSpObNm%2BvUqVPy8Pj/HujixYsVEBDgUB46pwAAAPewkSNH6vTp01q1apW%2B//57nT59WuPHj89y3REjRmjbtm221%2BTJk5UvXz4NHTpUkpSYmKi//vpLK1eutFvP0cJUojgFAAAwVEaG8a87JTk5WcuXL1dUVJT8/PxUqFAhvfDCC1qyZImSk5NvuG1CQoJeeOEFvfbaa6pQoYIkaefOnfLz81PJkiVvOxOH9QEAAAzkaof1U1JSFBsbm%2BWy5ORkpaWlKTAw0DYWEBCglJQUHTlyRA888MB/7nf8%2BPGqXLmyWrdubRv7888/5evrq65du%2BrAgQMqWbKkIiMj1ahRI4fzUpwCAAC4uLi4OJ09e9ZurEiRIvL397/ptjt27FD37t2zXDZ48GBJUq5cuWxjvr6%2BkqTLly//5z6PHz%2BuZcuW6YsvvrAbt1gseuihh/Tcc8%2BpRIkS%2Bu677xQZGan58%2BeratWqN80qUZwCAAAY6k50TqOjozVlyhS7sUGDBikyMvKm24aGhmrfvn1ZLtu9e7c%2B%2BOADJScn2y6Aun44P0%2BePP%2B5zy%2B//FLBwcGZOqt9%2BvSx%2B7p169b65ptvtGrVKopTAACA7CIiIkJhYWF2Y0WKFHF6v%2BXKlZOnp6cOHjyoKlWqSJIOHTokT09PlS1b9j%2B3%2B/7779WrV69M47NmzVKlSpVUu3Zt21hqaqq8vb0dzkRxCgAAYKA70Tn19/d36BD%2BrfL19VXz5s01fvx4ffDBB5KunUvasmVL%2Bfj4ZLnN%2BfPndejQIdWoUSPTstOnT%2BuLL77QjBkzVLx4cX311Vfatm2b3nrrLYczUZwCAAAYyNUuiLqZ4cOH6%2B2331arVq2Ulpamxo0b6/XXX7ctb9GihVq1aqX%2B/ftLkk6cOCFJKlq0aKZ9vfTSS/Lw8FDnzp116dIllS9fXh9//LHuu%2B8%2Bh/NYrFar1cn3hHuZO318LBa3yGuVxewIDnOHKbXIxQP%2BkztMqKTkFPf4jFosko%2BPlJLi%2BtPqm8s95lTBwdLWrVJIiLRtm9lpbszEH3rdusbv89dfjd%2Bnq%2BI%2Bp27oypUrOnPmjNP7OXLkiPNhAACAHXe6z6krojh1Q507d9Zvv/3m1D5%2B%2Bukn9e7d26BEAAAAxuCcUzd0/vx5p/dx4cIFcUYHAADGu9c6nUajc%2BpmevXqpVOnTmn48OEaMWKEdu3apW7duqlGjRpq2rSp5s6days6Y2Nj1adPH9WsWVMNGjTQoEGDFBcXp5iYGA0fPlynTp1ScHDwfz41AgAA3DoO6zuHzqmbmT17tsLCwjRo0CDVrVtXLVq00LPPPqvZs2fr6NGjGjhwoHx8fNSpUydNmDBBxYoV0/Tp03XlyhVFRUXp448/1rBhw/TWW29pypQp%2Bumnnxz%2B3lk%2BnaJw4TtyawsArsviJtfuXM/pFnmDg81O4JigIPt/AncAxakbW7ZsmQICAtSlSxdJUvny5dW7d2/Nnz9fnTp1kre3tzZv3qwVK1aodu3amjlzpjw8br9ZnuXTKZ55RpFRUU69j7vKDf5KuX5Ce64/pS4f0J7rT6j%2B49aHLusW7v1tnq1bzU5waxYuNDuBS7vXOp1Gozh1YydPntSuXbtUvXp121hGRoZy5MghSRo2bJg%2B%2BugjzZo1S0OHDlVQUJCGDRtmt/6tyPLpFIULu/49Wq5zk9v0cCspY3ErKeOlXHGPz6jFcq0wvXLF9afVp06I2REcExR0rTDt3Fnau9fsNDfmbgU/bChO3VixYsUUGhqqWbNm2cbOnz%2Bvy5cvS7r2vNyIiAhFRkYqISFBU6dO1aBBg7Rx48bb%2Bn5ZPp3C1X/jAzCcu/1nb7W6QWZXv2fov%2B3d636Z7yI6p87hgig35OXlpUuXLqlVq1bavn27li1bpvT0dMXFxal///4aN26cJOnDDz/UyJEjlZiYqHz58snX11cFChSQJHl7eys5OVnp6elmvhUAALIdLohyDsWpG%2BrQoYMmTpyoiRMnaubMmYqOjladOnXUpk0b3X///bbidMSIEcrIyFDjxo1Vo0YN7dixw/bc3Bo1aqhQoUKqUaOG9u3bZ%2BbbAQAAsOHxpXCOO3183OR8Ps45NRbnnBqPx5caj8eX3gEm/tAfesj4ff75p/H7dFV0TgEAAOAyuCAKAADAQPfaOaJGozgFAAAwEMWpczisDwAAAJdB5xQAAMBAdE6dQ%2BcUAAAALoPOKQAAgIHonDqH4hQAAMBAFKfO4bA%2BAAAAXAadUwAAAAPROXUOnVMAAAC4DDqnAAAABqJz6hyKUwAAAANRnDqHw/oAAABwGXROAQAADETn1Dl0TgEAAOAy6JwCAAAYiM6pcyhOAQAADERx6hwO6wMAAMBl0DkFAAAwEJ1T51CcAgAAGIji1Dkc1gcAAIDLoHMKAABgIDqnzqFzCgAAAJdB5xQAAMBAdE6dQ3EKAABgIIpT53BYHwAAAC6DzikAAICB6Jw6h84pAAAAXAadUwAAAAPROXUOxSkAAICBKE6dw2F9AAAAuAyKUwAAAANlZBj/uhuSk5MVERGhJUuW3HC9HTt26IknnlBwcLDCwsL0xRdf2C1funSpwsPDVbVqVbVr107btm27pRwUpwAAAPe4AwcOqEuXLtq%2BffsN1/v777/Vr18/tW3bVps3b9bo0aM1duxY/fHHH5KkmJgYjRw5UuPGjdPmzZvVunVrDRgwQMnJyQ5noTgFAAAwkLt1Tjds2KAePXro8ccfV4kSJW647vfffy8/Pz916dJFOXPmVO3atdWqVSstWLBAkvTFF1%2BoRYsWqlatmjw9PdWzZ08VKFBAK1eudDgPF0QBAAAYyNUuiEpJSVFsbGyWy4oUKaKgoCCtWbNG3t7emjNnzg33deDAAQUGBtqNlS9fXosXL5YkHTx4UO3bt8%2B0fO/evQ7npTgFAABwcXFxcTp79qzdWJEiReTv73/TbXfs2KHu3btnuWzq1Klq0qSJwzkuX74sX19fuzEfHx8lJSU5tNwRFKdwjsVidgKHxMXFKTo6WhEREQ79h2wm95hRd5pT95hR95lP6V9/d1xWXFycZsxwjzmV1Wp2AofExcUpevJkRXz3nevPqYnuxI9z8uRoTZkyxW5s0KBBioyMvOm2oaGh2rdvnyE5fH19denSJbuxlJQU5c6d27Y8JSUl0/ICBQo4/D045xT3hLNnz2rKlCmZ/q8Tt485NRbzaTzm1HjMqXmuX0X/z1dERMRdzxEYGKgDBw7YjR08eFAVKlSQJFWoUOGGyx1BcQoAAODi/P399eCDD9q9zOheh4eH69y5c5o7d67S0tK0ceNGLV%2B%2B3HaeaYcOHbR8%2BXJt3LhRaWlpmjt3ruLj4xUeHu7w96A4BQAAwH9q0aKFPvzwQ0lSgQIFNHv2bH333XcKDQ3VsGHDNGzYMNWqVUuSVLt2bQ0fPlxvvvmmatasqRUrVmjGjBny8/Nz%2BPtxzikAAAAkST/99FOmsRUrVth9/dBDD%2Bnzzz//z320adNGbdq0ue0MdE5xTyhSpIgGDRqkIkWKmB0l22BOjcV8Go85NR5zirvBYrW6ySWCAAAAyPbonAIAAMBlUJwCAADAZVCcAgAAwGVQnAIAAMBlUJwCAADAZVCcAgAAwGVQnAIAAMBlUJwCAADAZVCcAgAAwGVQnAIAgCydPXs2y/EDBw7c5SS4l/D4UmRbCQkJWrZsmU6ePKnBgwdr8%2BbNatSokdmx3NqiRYv06aefKi4uTkuXLtW4ceM0duxY5c6d2%2BxobiUsLEwWi%2BWG6/z44493KY3769at203nc968eXcpTfYSEhKirVu32o1dvXpVNWrUyDQOGCWn2QGAO2HXrl166qmndP/992vfvn3q3r27Bg8erOHDh6t9%2B/Zmx3NLc%2BfO1WeffabevXvrnXfeUe7cuRUXF6exY8dq1KhRZsdzK5GRkZKufU5//PFHPfXUUypTpoxOnz6tOXPmqHHjxiYndC%2BhoaGSpBMnTmj16tVq3769ypQpozNnzmjRokV69NFHTU7oXo4eParevXvLarUqOTk50%2BcxJSVFJUuWNCkd7gV0TpEtde3aVe3atVO7du1Uo0YNbd68WevXr9fYsWO1cuVKs%2BO5pWbNmmnatGkKCAhQzZo1tWnTJsXFxenxxx/Xr7/%2BanY8t9S6dWtNnDhRAQEBtrGjR4%2BqX79%2BWrVqlYnJ3FPnzp31wgsvKCQkxDa2c%2BdOvf7661q6dKmJydzPmjVrdP78eb355pt666237JZ5e3urRo0aKlKkiEnpkN3ROUW2tH//frVp00aSbIf76tevryFDhpiYyr2dP39e5cqVkyRd/3/aQoUKKT093cxYbu348eMqU6aM3VjRokUVFxdnUiL3tmfPHlWpUsVurGLFijpy5Ig5gdzY9VOgSpUqpZo1a5qcBvcaLohCtlSwYEEdPnzYbuzw4cMqXLiwSYncX1BQkKKjoyX9f8G/cuVKVahQwcxYbq1y5cp6%2B%2B23lZqaKklKTk7WyJEjVa1aNZOTuaeAgADNnTvXbuzDDz9UUFCQOYGygXLlymnMmDGSpN9//1116tRRixYtdPDgQZOTITvjsD6ypXnz5mnu3Lnq37%2B/xo0bp1GjRmn69Ol6/PHH1atXL7PjuaVdu3apZ8%2BeCggI0M6dO1W7dm1t375dM2fOzNStgmMOHz6sp59%2BWqdPn1aBAgVs3emPP/5YxYsXNzue29m6dav69%2B%2BvXLlyqVixYjp16pQyMjI0a9YsVaxY0ex4bikyMlJJSUmaOXOm2rdvr5CQEPn6%2BuqPP/7QJ598YnY8ZFMUp8i2FixYoIULF%2BrkyZMqVqyYOnbsqKeeeuqmV/Xiv8XGxmrZsmU6deqUihUrplatWqlEiRJmx3Jr6enp2rZtm2JjY1WsWDGFhITIw4ODWrfrwoUL%2Bvnnn23zGRYWprx585ody201bNhQK1euVGJioh555BH99ttvyps3r0JDQ7Vlyxaz4yGb4pxTZEvp6enq0qWLunTpYje%2BZcsWDpk6oWjRourbt6/ZMbKVjIwMXbhwQefOnVPjxo21f/9%2BDkM7wc/PT4GBgcqVK5caNmyoS5cumR3JrSUnJ8vHx0c//PCDAgMDVaBAASUmJipnTsoH3Dl8upAtVa1aVa%2B99pqefPJJu/G%2Bfftyb75bxD0575xjx46pV69eSktL08WLF/XII4%2Boffv2mjJlCvfkvQ3x8fF65plntHPnTnl6emrx4sXq0KGDZs%2BereDgYLPjuaWHH35Yb775prZs2aLmzZvr3LlzGjFiBBdJ4Y6iOEW2NW3aNB0/flwvvfSSbYyzWG7d9XtywnijR49Wu3btNGDAANWsWVPlypXTqFGjNGnSJIrT2zBmzBgFBgZqzpw5atCggQICAtSvXz%2B98847%2Buyzz8yO55ZGjx6tCRMmqHr16nr66ae1e/dupaamcm9j3FGcc4psKSQkRMuXL1efPn0UGBiod999V15eXlk%2B7QS37vz58ypQoIDZMdxeaGio1q9fLy8vL9u9YzMyMlSzZk39/vvvZsdzO3Xr1tXq1avl6%2Btrm8%2B0tDTVqVNHmzdvNjseAAdx1j2yrZIlS2rhwoWKjY1V9%2B7ddf78ec6TcsLly5c1bNgwValSRXXq1FFISIjeeecd222QcOvy5s2rc%2BfO2Y2dPXtW%2BfPnNymRe/P09FRKSoqk/z9KcvnyZR6v66RFixapVatWCg0N1alTpxQVFaXLly%2BbHQvZGMUpsrUCBQrok08%2BUaFChdS5c2cO6zth3LhxOnDggKZNm6YVK1Zo4sSJ2rhxoyZOnGh2NLfVqlUrDRo0SL/%2B%2BqsyMjL0xx9/6IUXXlCLFi3MjuaWwsLC9OKLL%2BrIkSOyWCyKj4/XW2%2B9pUceecTsaG5r7ty5mjVrlrp166arV68qd%2B7cio2N1dixY82OhmyMw/rIltq0aaOvv/7a9rXVatWIESP02Wefae/evSYmc1/16tXTsmXLVLBgQdvYmTNn1KFDB/3yyy8mJnNfaWlpmjBhgj7//HMlJyfL29tbHTp00MsvvywvLy%2Bz47mdy5cv65VXXtH3338v6drDIh555BG9%2B%2B673E7qNvHYYpiB4hT3lNOnT3Nz89sUHh6uxYsX2x1yvnjxoh599FH99ttvJibLHhISElSgQAHuw2uAhIQEnThxQsWKFZO/v7/ZcdxazZo1tXHjRnl4eKhGjRravHmzrl69qjp16igmJsbseMimOAEP2cqbb76pN998U6%2B88sp/rsPhqFtz6tQpSVLbtm317LPPaujQoSpZsqTi4uL07rvvqmfPnuYGdHM7duzQsWPHdPXqVbvxtm3bmhPIzZ09e1bHjx/X1atXdfToUR09elSSVKNGDZOTuafrjy1%2B8skneWwx7hqKU2QrHAgw3vX7nF6f29atW9v%2BSFmtVq1Zs0b9%2BvUzM6Lbmjhxoj7%2B%2BGMVLlxYnp6etnGLxUJxehsWLFigUaNGZfo9YLFYtGfPHpNSubehQ4eqR48e%2Bvrrr5WUlKS%2BffvaHlsM3Ckc1gdwQydPnrzpOiVLlrwLSbKf2rVr6/3331doaKjZUbKFBg0a6LnnnlOLFi3sin3cvlGjRqlfv35atmyZTp48qeLFi6tly5Z6//339c4775gdD9kUnVNkS%2BfOndPHH3%2BsV199Vb///ruioqJUsGBBffDBBwoICDA7nlu5UeGZnp6u/fv3U5zephw5clCYGig1NZWOswFiY2O1YcMGSdIXX3yhypUrq3DhwipcuLCka0%2BE%2B%2BGHH8yMiGyO4hTZ0ltvvaWkpCRZrVaNHj1ajz32mHx9fTVixAh98sknZsdzSz///LPeeustxcbG2h02zZkzp/78808Tk7mvRo0a6ZtvvlHLli3NjpIthIaGauPGjapVq5bZUdxagQIFNH/%2BfCUkJCg1NVWTJk2yW%2B7t7a1BgwaZlA73Ag7rI1tq2LChVq5cqcTERD3yyCP67bfflDdvXoWGhmrLli1mx3NLLVu2VN26dZUvXz7t27dPLVu21NSpU9WhQwd169bN7HhupVu3brJYLLp8%2BbL27Nmj8uXLy8/Pz26defPmmRPODV2/ADI%2BPl4xMTGqU6dOpvnkQsjb07t3b82aNcvsGLjH0DlFtpScnCwfHx/98MMPCgwMVIECBZSYmMgTopxw/Phxvfjiizpx4oQ2btyopk2b6v7779ezzz5LcXqL/nkov1GjRiYmyV4KFSqkxx57zOwY2QqFKczAX2pkSw8//LDefPNNbdmyRc2bN9e5c%2Bc0YsQI1axZ0%2BxobqtgwYLy8PBQiRIldOjQIUlS%2BfLldebMGZOTuZ9/HhI9dOiQihYtqjx58mjbtm3Kly8f50Xfon92RS9evChvb295e3vr0KFDKliwoAoUKGBiOgC3iseXIlsaPXq0UlNTVb16dfXv318nT55Uamqqhg8fbnY0t1WxYkV98MEHkq51qNauXauYmBh5e3ubnMx9ffvtt2rbtq2OHDkiSdq%2BfbueeOIJrV271txgbmrjxo165JFHbLeNWr58uZo1a6Y//vjD5GQAbgXnnOKe1a9fP3388cdmx3Abhw4dUlRUlD7%2B%2BGPt3r1bQ4YMUUZGhl566SU99dRTZsdzSy1atNDQoUNVv35929j69ev17rvvatmyZSYmc0/t27dXp06d9MQTT9jGvvzyS33xxRf6/PPPTUwG4FZQnOKeFRISoq1bt5odw23FxcXp8uXLKleunNlR3FZWn0Gr1aoaNWro999/NymV%2B6pWrVqmCx6ZT8D9cM4pAIdt3LhRX3/9tc6ePasSJUqoQ4cOZkdyayVLltT69evtOqcbNmxQiRIlTEzlvgoVKqQ//vhDDz/8sG1s586dtvtzAnAPFKcAHLJo0SKNHDlSTZs21QMPPKATJ06oW7duGj9%2BvMLDw82O55b69eunZ555Rk2bNlXJkiV16tQp/fDDD3r77bfNjuaWunTpon79%2BikiIsI2n4sWLeKenICb4bA%2B7lkc1r81TZo00VtvvaW6devaxtauXat33nlHK1asMDGZe4uJidFXX32ls2fPqnjx4nr88ccVEhJidiy3tWTJErv5bNeuHQ85ANwMxSnuWRSntyY4OFi///67cuTIYRvLyMhQrVq1tGnTJhOTua9Zs2apd%2B/emcbff/99DRky5O4HcnPffvutmjdvnmk8OjpaERERJiQCcDs4rA/AIfXr19f8%2BfPVo0cP29iKFStUp04dE1O5n4SEBNt9YidPnqwqVarYPQ720qVL%2BuSTTyhOHZScnKzz589Lkl599VVVrVo103yOGzeO4hRwIxSnuGdx0ODWXL16VePGjdPSpUt13333KTY2Vjt27NADDzyg7t2729bjsZs35uXlpaioKFtB1bVr10zLKaQcl5iYqBYtWiglJUWSFBYWJqvVKovFYvtnkyZNTE4J4FZwWB/Z0uzZs9W2bVsVLFjwP9f5/vvv1bRp07uYyr1NmTLFofW4%2BMRxjz76qL777juzY7i9%2BPh4JScnq1WrVvrmm2/slnl7e3O1PuBmKE6RLXXs2FF79uxRw4YN9cQTT6h%2B/fqyWCxmxwIckpCQcMP/sULWMjIy5OGR%2BcGH6enpypmTA4WAu6A4RbZ16NAhLVmyRMuXL5eHh4fatWundu3aqVSpUmZHc0vnz5/Xp59%2BqtjYWGVkZEiS0tLStH//fp5mdJv%2B%2BOMPvfPOO5nmNCEhQTt37jQ5nfs5duyYpk6dmmk%2B//rrL23cuNHkdAAcRXGKbC8jI0Nr1qzR6NGjdebMGe3evdvsSG6pf//%2BOnLkiAoWLKjExESVKFFCv/zyi7p06aJXXnnF7HhuqUOHDipdurT8/Px0/Phx1a1bV/PmzVP37t15JOxt6Natm6xWqwoUKKD4%2BHhVqlRJX331lXr27MnpJoAbyXz8A8hGNm7cqNdee00vvviiChYsqDfffNPsSG5r8%2BbN%2BuSTTzR06FCVKVNGH374oUaPHq3Dhw%2BbHc1tHThwQGPHjlWXLl109epVPfXUU5o4caKWL19udjS3tHPnTk2dOlUDBw5U3rx5NWzYME2YMEEbNmwwOxqAW8BJOMiWrv%2BBT0xMVKtWrbRw4UIFBQWZHcut5cyZU0WLFpWvr6/27dsnSWrRooXeeecdk5O5r3z58snHx0elS5fWgQMHJElVq1bVyZMnTU7mnnx9fZU/f37lzJlT%2B/fvlyQ1aNBAL7/8ssnJANwKOqfIllavXq0hQ4Zo/fr1ev31122F6UsvvWRyMvdVsmRJ7dy5U/ny5dPly5eVkJCgpKQk2y18cOvuv/9%2BffbZZ/L29lauXLm0Z88eHTp0iIv3blOZMmW0du1a5c6dWxkZGTp%2B/LhiY2OVnp5udjQAt4DOKbKN2NhY2%2BG7EydOKCMjQ99%2B%2B61t%2BaVLl/TDDz%2BYFc/tde7cWd26ddOKFSvUsmVL9ejRQzlz5lSNGjXMjua2Bg8erAEDBqhu3brq3bu3OnbsqBw5cujJJ580O5pbevrppxUVFaVvvvlGERER6tSpk3LkyKHGjRubHQ3ALeCCKGQbqamp6ty5sxISEnT69GkVL17cbrm3t7c6dOiQ5eMi4Zg//vhDQUFBslgsmjt3rhITE9WrVy/lz5/f7Ghu68qVK/Ly8pLFYtGOHTuUmJiounXrmh3LLXXt2lX16tVT27ZtVaxYMa1cuVKJiYlq27atvLy8zI4HwEF0TpFteHl5afHixZKk3r17a9asWSYnyn727t2rokWLqmjRovL395efnx%2BFqRPCwsLUtGlTNWnSRNWrV1eVKlXMjuTWmjdvrp9%2B%2BklTp05VUFCQwsPD1bRpUwpTwM3QOQXgkEmTJmnp0qWaM2eOypYtqx9//FFjxozRk08%2BqT59%2Bpgdzy2tWbPG9rJarWrcuLGaNm2qWrVqKUeOHGbHc1uJiYlat26d1qxZo9WrV6tUqVLcAQFwIxSnABzSoEEDLViwQKVLl7aNHTt2TD169NCaNWtMTJY9/PHHH1q1apUWLlwoLy8vxcTEmB3JLSUmJmrjxo369ddf9dtvv%2Bn06dOqXr26Zs%2BebXY0AA7isD4AhyQmJmY6j7d48eJKSkoyKVH2sH//fv3222/67bfftHnzZhUoUIBzTm/T9ccWly9fXqGhoRo2bJhq1qwpb29vs6MBuAUUpwAc8uCDD%2Brjjz/WwIEDbWOzZ8/m/rFOqFevni5fvqy6deuqfv36Gjp0qO6//36zY7ktb29veXp6Kn/%2B/CpUqJAKFy5MYQq4IQ7rA3DIrl271KtXL/n6%2BqpYsWI6c%2BaM0tPTNXPmTArU2zR06FD9%2Buuv8vHxUd26dVWvXj3VqlVLefLkMTua20pKStLGjRu1fv16bdiwQZcuXVKdOnX07rvvmh0NgIMoTgE47O%2B//9aaNWsUFxen4sWLq2HDhsqbN69t%2BZkzZ1SsWDETE7qnffv22YqpHTt2KDAwUAsXLjQ7ltu6cuWKNm7cqHXr1mnlypXy9PTUunXrzI4FwEEc1gfgsPz586tt27b/ufyxxx7T1q1b716gbCJ37tzy9fWVp6enMjIylJGRYXYktzRv3jytW7dOmzdvVvHixdWkSRNNnz5dVatWNTsagFtA5xSAYYKDg7Vt2zazY7iNMWPGaP369Tpx4oRq1qypxo0bq0mTJvL39zc7mltq3769wsPD1aRJE5UvX97sOABuE51TAIbhmfC35syZMxo4cGCm0yP%2BacuWLapWrdpdTuaevvzyS7MjADAAnVMAhgkJCeGwvsGYUwD3Gg%2BzAwAA/hv9AwD3GopTAHBhnCoB4F5DcQoAAACXQXEKwDBeXl5mRwAAuDmu1gdwQ5s3b77pOjVq1JAkbdy48U7HAQBkcxSnAG6oW7dukuzPfcyfP78uXbqkjIwM%2Bfn5acOGDWbFAwBkMxSnAG5o7969kqRZs2Zp//79GjZsmPLmzaukpCSNGzdO%2BfPnNzlh9la2bFmzIwDAXcV9TgE4pE6dOvrpp5/k4%2BNjG7ty5YoaNGigmJgYE5O5t0OHDumzzz7TmTNnNHLkSK1YsUJdu3Y1OxYAmIYLogA4JCMjQ/Hx8XZjJ06cUI4cOUxK5P5%2B/fVXdezYUefPn9dvv/2mlJQUTZ06VR9//LHZ0QDANBSnABzSpk0b9e7dW4sXL9avv/6qzz//XE8//bQ6depkdjS3NWHCBE2YMEHvvfeecuTIoeLFi%2Bvjjz9WdHS02dEAwDSccwrAIS%2B%2B%2BKJy5cql6dOnKzY2VsWLF1fHjh3Vt29fs6O5raNHj6pBgwaS/v%2BCs4ceekh///23mbEAwFQUpwAckjNnTg0ePFiDBw82O0q2UaJECW3dulXVqlWzjf35558qXry4iakAwFwUpwAckpGRoe%2B%2B%2B07Hjh1Tenq63bJBgwaZlMq9Pf300xowYICefPJJpaWlacaMGfr000/13HPPmR0NAExDcQrAIcOHD9c333yjihUrytPT0zbOs99vX4sWLZQnTx4tWLBAJUqU0MaNG/Xaa6%2BpWbNmZkcDANNwKykADqlevbqio6MVEBBgdhQAQDZG5xSAQ/Lmzaty5cqZHSNbOX78uD788EOdPHlSGRkZdsvmzZtnUioAMBfFKQCHtGzZUrNnz1afPn3MjpJtPPfcc/L09FStWrXk4cGd/QBA4rA%2BgJsICwuTxWJRenq6YmNjlTdvXuXLl89unR9//NGkdO4tODhYGzZssHvqFgDc6%2BicArihyMjIGy7ngqjbFxQUpDNnzqhs2bJmRwEAl0HnFIBDunXr9p%2BFKOdH3p5du3bpmWeeUdOmTTN1o7k9F4B7FZ1TAA4JDQ21%2B/r8%2BfP67rvvFBERYVIi9zd58mQlJSVp165dduec0o0GcC%2Bjcwrgtu3atUvvvPOOPvnkE7OjuKXg4GD98MMPKly4sNlRAMBlcHkogNv24IMPaufOnWbHcFv%2B/v7y9vY2OwYAuBQO6wNwyKlTp%2By%2BTktL04oVK3gOvBN69%2B6tgQMHqnv37sqfP7/d4fwaNWqYmAwAzMNhfQAOCQoKsiuerFar8ufPr1GjRik8PNzEZO4rKCgoy3GLxaI9e/bc5TQA4BooTgE45OTJk3Zf58iRQ4UKFZKnp6dJiQAA2RHFKQDcZWfOnFGxYsUynSrxTyVKlLiLiQDAdVCcAsBdFhISoq1bt9pOlbj%2Ba/j6v3NYH8C9jOIUAO6y06dPq3jx4plOlfinkiVL3sVEAOA6KE4BwCQDBgzQ9OnTM4137dpV8%2BfPNyERAJiPW0kBwF104sQJffXVV5KkX375RVOmTLFbnpiYqH379pmQDABcA8UpANxFJUqU0IEDB5SQkKCrV68qJibGbrm3t7eGDx9uUjoAMB%2BH9QHAJMOGDdOoUaNuuM6WLVtUrVq1u5QIAMxHcQoALuz6lf0AcK/wMDsAAOC/0T8AcK%2BhOAUAF/bPR8YCwL2A4hQAAAAug%2BIUAAAALoPiFAAAAC6D4hQAAAAug%2BIUAFxY2bJlzY4AAHcV9zkFgLvs%2BuNLb6Rt27Z3PAcAuCKKUwC4y8LCwm643GKx6Mcff7xLaQDAtVCcAgAAwGXkNDsAANzLjh8/rtjYWNuToNLS0rR//3717NnT3GAAYBI6pwBgko8%2B%2BkgTJ060PQXKarXKYrHogQce0JIlS0xOBwDmoHMKACZZuHChJk2aJC8vL/3000967rnnNHLkSBUvXtzsaABgGm4lBQAmuXjxopo2baqgoCDt3LlTfn5%2Beu2117Ry5UqzowGAaShOAcAk/v7%2BSkxMVNGiRXXixAlZrVYVLFhQf//9t9nRAMA0HNYHAJPUqFFDUVFRev/991WpUiVNmDBB3t7eKlq0qNnRAMA0dE4BwCRDhw7Vfffdp/T0dL322mv68ccftWjRIr322mtmRwMA03C1PgCYZMeOHapSpUqm8XXr1qlBgwYmJAIA89E5BQCTPPXUU5nGEhMTNXjwYBPSAIBr4JxTALiLjh49qhYtWujq1auyWq164IEHMq0TEhJiQjIAcA0c1geAu2zPnj26ePGi%2BvXrpxkzZthuvi9J3t7eCgwMlK%2Bvr8kpAcAcFKcAYJLjx4%2BrdOnSkqT4%2BHjlz59fOXNyQAvAvY3iFABMkpaWpnfffVdffPGFUlJS5OXlpdatW%2Bv111%2BXl5eX2fEAwBRcEAUAJpk2bZpiYmL0/vvv65tvvtH777%2BvHTt26P333zc7GgCYhs4pAJikSZMmmjNnju3QviQdO3ZMXbp00fr1601MBgDmoXMKACb5%2B%2B%2B/Vbx4cbux4sWLKyUlxaREAGA%2BilMAMEnFihX1%2Beef2419/vnnCgwMNCkRAJiPw/oAYJLff/9dvXr1UlBQkEqXLq1jx47p4MGDmjVrFvc6BXDPonMKACb57rvv9PXXX6tevXrKnTu3wsPD9c0332TqpgLAvYQb6gHAXRQbG6sNGzZIkr744gtVrlxZZcqUUZkyZSRJP/30k3744QczIwKAqTisDwB3UWpqqjp37qyEhASdPn060wVR3t7e6tChg3r37m1SQgAwF8UpAJikd%2B/emjVrltkxAMClUJwCAADAZXBBFAAAAFwGxSkAAABcBsUpAAAAXAbFKQAAAFwGxSkAAABcBsUpAAAAXAbFKQAAAFwGxSkAAABcxv8BlVGric9AlNMAAAAASUVORK5CYII%3D" class="center-img">
</div>
    <div class="row headerrow highlight">
        <h1>Sample</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-12" style="overflow:scroll; width: 100%%; overflow-y: hidden;">
        <table border="1" class="dataframe sample">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>tv_make</th>
      <th>tv_size</th>
      <th>uhd_capable</th>
      <th>tv_provider</th>
      <th>total_time_watched</th>
      <th>watched</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-16</td>
      <td>Sony</td>
      <td>70</td>
      <td>0</td>
      <td>Comcast</td>
      <td>10.75</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-18</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.75</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-18</td>
      <td>Sony</td>
      <td>50</td>
      <td>1</td>
      <td>Dish Network</td>
      <td>20.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-19</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>Comcast</td>
      <td>1.50</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-17</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>Comcast</td>
      <td>17.50</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
    </div>
</div>
</div>



### 3. Investigation of the AB Test.

####  Part 1. Reproduce the Negative Result Found in the AB Test.


```python
merged = pd.merge(test_data, viewer_data, on="viewer_id")
merged.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>viewer_id</th>
      <th>date</th>
      <th>tv_make</th>
      <th>tv_size</th>
      <th>uhd_capable</th>
      <th>tv_provider</th>
      <th>total_time_watched</th>
      <th>watched</th>
      <th>test</th>
      <th>gender</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24726768</td>
      <td>2018-01-16</td>
      <td>Sony</td>
      <td>70</td>
      <td>0</td>
      <td>Comcast</td>
      <td>10.75</td>
      <td>0</td>
      <td>1</td>
      <td>Male</td>
      <td>52</td>
      <td>Boston</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25001464</td>
      <td>2018-01-18</td>
      <td>Sony</td>
      <td>32</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.75</td>
      <td>0</td>
      <td>0</td>
      <td>Male</td>
      <td>38</td>
      <td>New York</td>
    </tr>
  </tbody>
</table>
</div>




```python
#  two-sample z-test of proportions:
n1 = merged[merged.test == 1].shape[0] # number of commercial viewers in experiment group
n2 = merged[merged.test == 0].shape[0] # number of commercial viewers in control group
p1 = merged[merged.test == 1].watched.sum() / n1 # proportion of show watchers in experiment group
p2 = merged[merged.test == 0].watched.sum() / n2 # proportion of show watchers in control group

p_pool = merged.watched.sum() / (n1 + n1) # pooled propportion of show watchers
print(n1, n2, p1, p2, p_pool)
```

    204327 213143 0.04577955923593064 0.06309379149209686 0.05579781428788168



```python
z = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
p_value = stats.norm.cdf(z) * 2
print("z statistic: {0:.6f}".format(z))
print("p value: {0:.6E}".format(p_value))
```

    z statistic: -24.363963
    p value: 4.123635E-131


Above is the reprodduction of the AB test. Since z = -24.36 < 0 and p_value << 0.05, the AB test result shows the experiment group (the audience who viewed the new commercials involving local Mayors) has a statistically significantly smaller proportion of commercial viewers that susequently watched the show.

#### Part 2. Explain what might be happening

Now we do AB test for each city, except Los Angeles (this is because all viewers in the Los Angeles can only be categorized into the control group). And the z-satistic of the proportion test as well as the corresponding p value will be presented for each city.


```python
def two_sample_z_test(df):
    n1 = df[df.test == 1].shape[0] 
    n2 = df[df.test == 0].shape[0] 
    p1 = df[df.test == 1].watched.sum() / n1 
    p2 = df[df.test == 0].watched.sum() / n2 
    p_pool = df.watched.sum() / (n1 + n1) 

    z = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    p_value = stats.norm.cdf(z) * 2 if z < 0 else (1-stats.norm.cdf(z)) * 2
    s = pd.Series([ p_pool, z, p_value], index=["p_pool","z_statistic", "p_value"])
    return s
```


```python
res = merged[merged.city!="Los Angeles"].groupby("city").apply(two_sample_z_test)
```


```python
res[["z_statistic", "p_value"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z_statistic</th>
      <th>p_value</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Atlanta</th>
      <td>0.237036</td>
      <td>0.812629</td>
    </tr>
    <tr>
      <th>Boston</th>
      <td>0.299507</td>
      <td>0.764553</td>
    </tr>
    <tr>
      <th>Chicago</th>
      <td>0.930993</td>
      <td>0.351857</td>
    </tr>
    <tr>
      <th>Dallas</th>
      <td>-1.178892</td>
      <td>0.238441</td>
    </tr>
    <tr>
      <th>Detroit</th>
      <td>1.949756</td>
      <td>0.051205</td>
    </tr>
    <tr>
      <th>Houston</th>
      <td>0.441955</td>
      <td>0.658522</td>
    </tr>
    <tr>
      <th>Miami</th>
      <td>0.782176</td>
      <td>0.434111</td>
    </tr>
    <tr>
      <th>Minneapolis</th>
      <td>-0.208516</td>
      <td>0.834826</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>-1.837328</td>
      <td>0.066162</td>
    </tr>
    <tr>
      <th>Philadelphia</th>
      <td>0.002847</td>
      <td>0.997728</td>
    </tr>
    <tr>
      <th>Phoenix</th>
      <td>-0.678374</td>
      <td>0.497535</td>
    </tr>
    <tr>
      <th>San Francisco</th>
      <td>-0.111468</td>
      <td>0.911245</td>
    </tr>
    <tr>
      <th>Seattle</th>
      <td>-0.971047</td>
      <td>0.331525</td>
    </tr>
    <tr>
      <th>Tampa</th>
      <td>0.343183</td>
      <td>0.731461</td>
    </tr>
  </tbody>
</table>
</div>



Interestingly, the individual AB tests show no significant difference in watching rate (proportion of commercial vieweres who will end up watching the tv show) in any of the cities. 


```python
test_group = merged[merged.test == 1]
control_group = merged[merged.test == 0]
test_group.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>viewer_id</th>
      <th>date</th>
      <th>tv_make</th>
      <th>tv_size</th>
      <th>uhd_capable</th>
      <th>tv_provider</th>
      <th>total_time_watched</th>
      <th>watched</th>
      <th>test</th>
      <th>gender</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24726768</td>
      <td>2018-01-16</td>
      <td>Sony</td>
      <td>70</td>
      <td>0</td>
      <td>Comcast</td>
      <td>10.75</td>
      <td>0</td>
      <td>1</td>
      <td>Male</td>
      <td>52</td>
      <td>Boston</td>
    </tr>
  </tbody>
</table>
</div>




```python
# In below, I use "views" and "number of observations" interchangeably.
test_cnt = test_group.groupby('city').size()
df1 = test_cnt / test_group.shape[0]
control_cnt = control_group.groupby('city').size()
df2 = control_cnt / control_group.shape[0]
nobs = pd.concat([test_cnt, control_cnt, df1, df2], axis=1)
nobs.columns = ["test_city_views", "control_city_views", "prop_to_test_views", "prop_to_control_views"]
nobs = nobs.fillna(0).round(4)
nobs.test_city_views = nobs.test_city_views.astype(int)
nobs["diff"] = nobs.prop_to_test_views - nobs.prop_to_control_views
p_pool = pd.DataFrame({"p_pool":merged.groupby("city").apply(lambda df: df.watched.sum() / df.shape[0])})
nobs = nobs.join(p_pool)
```


```python
nobs.sort_values("p_pool")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_city_views</th>
      <th>control_city_views</th>
      <th>prop_to_test_views</th>
      <th>prop_to_control_views</th>
      <th>diff</th>
      <th>p_pool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Philadelphia</th>
      <td>24495</td>
      <td>4261</td>
      <td>0.1199</td>
      <td>0.0200</td>
      <td>0.0999</td>
      <td>0.023473</td>
    </tr>
    <tr>
      <th>Seattle</th>
      <td>15850</td>
      <td>2846</td>
      <td>0.0776</td>
      <td>0.0134</td>
      <td>0.0642</td>
      <td>0.026048</td>
    </tr>
    <tr>
      <th>Dallas</th>
      <td>13807</td>
      <td>12874</td>
      <td>0.0676</td>
      <td>0.0604</td>
      <td>0.0072</td>
      <td>0.049286</td>
    </tr>
    <tr>
      <th>San Francisco</th>
      <td>12609</td>
      <td>11691</td>
      <td>0.0617</td>
      <td>0.0549</td>
      <td>0.0068</td>
      <td>0.049794</td>
    </tr>
    <tr>
      <th>Detroit</th>
      <td>9192</td>
      <td>8739</td>
      <td>0.0450</td>
      <td>0.0410</td>
      <td>0.0040</td>
      <td>0.050137</td>
    </tr>
    <tr>
      <th>Atlanta</th>
      <td>12599</td>
      <td>11728</td>
      <td>0.0617</td>
      <td>0.0550</td>
      <td>0.0067</td>
      <td>0.050561</td>
    </tr>
    <tr>
      <th>Chicago</th>
      <td>16991</td>
      <td>16054</td>
      <td>0.0832</td>
      <td>0.0753</td>
      <td>0.0079</td>
      <td>0.050598</td>
    </tr>
    <tr>
      <th>Miami</th>
      <td>8984</td>
      <td>8301</td>
      <td>0.0440</td>
      <td>0.0389</td>
      <td>0.0051</td>
      <td>0.050969</td>
    </tr>
    <tr>
      <th>Tampa</th>
      <td>9631</td>
      <td>8974</td>
      <td>0.0471</td>
      <td>0.0421</td>
      <td>0.0050</td>
      <td>0.051492</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>35812</td>
      <td>34082</td>
      <td>0.1753</td>
      <td>0.1599</td>
      <td>0.0154</td>
      <td>0.051492</td>
    </tr>
    <tr>
      <th>Boston</th>
      <td>12660</td>
      <td>11758</td>
      <td>0.0620</td>
      <td>0.0552</td>
      <td>0.0068</td>
      <td>0.051888</td>
    </tr>
    <tr>
      <th>Houston</th>
      <td>12589</td>
      <td>11802</td>
      <td>0.0616</td>
      <td>0.0554</td>
      <td>0.0062</td>
      <td>0.051986</td>
    </tr>
    <tr>
      <th>Phoenix</th>
      <td>10015</td>
      <td>9317</td>
      <td>0.0490</td>
      <td>0.0437</td>
      <td>0.0053</td>
      <td>0.052555</td>
    </tr>
    <tr>
      <th>Minneapolis</th>
      <td>9093</td>
      <td>8203</td>
      <td>0.0445</td>
      <td>0.0385</td>
      <td>0.0060</td>
      <td>0.052787</td>
    </tr>
    <tr>
      <th>Los Angeles</th>
      <td>0</td>
      <td>52513</td>
      <td>0.0000</td>
      <td>0.2464</td>
      <td>-0.2464</td>
      <td>0.103060</td>
    </tr>
  </tbody>
</table>
</div>




```python
nobs.test_city_views - nobs.control_city_views
```




    Atlanta            871
    Boston             902
    Chicago            937
    Dallas             933
    Detroit            453
    Houston            787
    Miami              683
    Minneapolis        890
    New York          1730
    Philadelphia     20234
    Phoenix            698
    San Francisco      918
    Seattle          13004
    Tampa              657
    Los Angeles     -52513
    dtype: int64



The above shows that the number of observations in test and control groups are highly unbalanced given each city (except Los Angeles, all other cities are having more observations in the control group than the observations in the test group). It seems this is a consequence of the erroneous data collection process aming to balance the total number of observations in the test and control groups!

From another perspective, we can see that the distribution of views (by city) are drastically differerent between the test and control groups (see column "prop_to_test_views" and "prop_to_control_views"). In the test group, views are spread in the non-Los-Angeles cities where pooled watching rate are low (ranging from 0.023 to 0.053) and no view is present in Los Angeles; whereas in the control group, 24.64% views are from Los Angeles where the pooled watching rate is as high as 0.103, and less views are collected from lower watching rate cities. This is the exact setup of Simpson's Paradox! Here, it is the difference in distribution of city_views between the test and control groups, combined with the fact that different cities have different overall watching rates (p_pool, Los Angeles vs others) that drives the misleading AB Test results.

(In order to fix the above issue, we should first make sure the views or the viewers are balanced on all levels of all charateristics)

At this stage, I suggest only trust the city-wise AB test results, which means, we found no enough evidence for any of the investigated city to conclude that the new commercial can help to change the watching rate.


```python
sizes = pd.DataFrame({"num_obs": merged.groupby(["test", "city"]).size()}).reset_index()

g = sns.catplot(
    data=size1, kind="bar",
    x="city", y="num_obs", hue="test",
    ci="sd", palette="dark", alpha=.6, height=3.2 , aspect=1.5)

for ax in g.axes.ravel():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    
g.fig.suptitle("Number of Observations in Different Cities (Test Group v.s. Control Group)")
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABs4AAAQMCAYAAAA2xWwrAAAgAElEQVR4XuzdC/w9U6H//6UQicidqCidSClOKqmUhIRUSrqrU6ioUCkdl1RuUa6dlK4i3UgK0Y2jEJJKBym3Xy4hdFPxf7ym//o8lvnO3nv2fGbPnj3zWo/HeZx8P3su67lmZs9e71lrFrr//vvvDxYFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFeiaw0EILLZRWeSGDs54dAVZXAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFAgEzA480BQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQwODMY0ABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUECBfws44swjQQEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAGDM48BBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBf4t4IgzjwQFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFDM48BhRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRT4t4AjzjwSFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFDA48xhQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQ4N8CjjjzSFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFDA4MxjQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQIF/CzjizCNBAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAYMzjwEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEF/i3giDOPBAUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUMzjwGFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFPi3gCPOPBIUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUMDjzGFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFDg3wKOOPNIUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUMDgzGNAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAgX8LOOLMI0EBBRTokcBrXvOacOGFF2Y13n333cPOO+9cqvb7779/+NKXvpR99pxzzgmPfOQjSy3Xpg/Fuj/0oQ8Nl156aZt2bar78o9//CMcf/zx4Vvf+la46aabwoMf/OCw/PLLh4985CPhKU95SqV9u+CCC7Lj5OKLLw633HJLuOuuu8Jiiy0WVltttbDeeuuFF73oRWGDDTYYue7nPe954cYbbwyPe9zjwumnnz7y835gfgIcC9ddd11Yc801H7CiG264ITz/+c/P/m3HHXcMH/zgB+e3oRlb+sgjjwxHHXVUttdf/epXw7rrrttoDb7+9a+H973vfQO3+aAHPSgsuuiiYemllw6rr7562GSTTcJWW20VVlhhhaH7+fjHPz77+3Of+9zwyU9+coHP3nrrrYG6//jHPw633XZbWGKJJbJ1nnLKKeEhD3lI+N3vfpf9ne+UO+64Izz84Q/Pjp3Pf/7zjfq0aWN//vOfw+23355d6+ooV199dXZtxvjaa68N99xzT9bWK664YnjSk56UXUs33nhjftAVbu6nP/1peO1rX5v9ba+99go77bTTAp/ry3l/2GGHhf/5n/8JH/jABwL3A+l5XbWt3va2t4W3v/3tVRevZblB7Vd15X//+9/D9773vfDDH/4w/PrXvw7/7//9v/DXv/41POxhDwvLLrtsdtxxzL3gBS/IjkWLAnmB+V632ibKNf2f//znyO/USe73e9/73vCNb3wj2wT32I94xCPmvbnLL788nHnmmdlvot///vfhT3/6U1hkkUXCMsssEx796EeHjTbaKGy++eZh1VVXnfe2urSC//7v/w4nnXRSOOigg8K2227bpapZFwUUUECBAgGDMw8LBRRQoEcCaXBGh8c3v/nNBTrJizgMzrp7kLzzne8MZ5xxxgIVPPfcc8f+sXzFFVeEAw88MFxyySUjwQjl9ttvvxA774sWMDgbyVjbB37yk58EzvMttthigY5gg7N2B2dFB8Hiiy8eOLdf97rXDTxGhgVnf/nLX8I222yTBalpoQONa8PNN98ctt5663DnnXc+4O9Pe9rTwhe+8IXajstZWtF3v/vd7IGD3XbbLWy33Xbz2nV8P/ShD4Wzzz473H///UPXxbWU7T7mMY9Z4HOjgrO+nPc8xMH9D0annXZaWHjhhTsRnA1rv3EPQI6zL3/5y5kLQcGossoqq4Q999wzbLnllqM+6t97IlDXdastXPfdd1/4yle+Ej72sY9l58WGG244tV2rMzgjEOd+r8y9OkHaK17xiuxc5wE4S8iuj5tttln23XzqqafO5MOktqMCCiigQHkBg7PyVn5SAQUUmHmBNDijMnS4nXjiiYERC8OKwdnMN31hBa655pq5Ti86xPfYY48sLOOp02c/+9ljVZrwjR/2PK1OYZQYHWpPfOITs5EwjJQgWGP0xP/93/9ln+EHOU9sMmqiqBicjdUElT/MiAJGHVGKRlAYnLUnOKMDa4cddnhAW//rX/8KBF2MGCUo4RxjFArlrW99axagFZVhwRkd6Pvuu2+2GGHYf/3Xf4Ull1wy6yjie+PQQw8Nn/rUp7K/v/CFL8z2iYcxGNH7hCc8ofKxOKsLEswwGpNCiDWf4OznP/95eMtb3pKN4qMwio+Qcp111smupYxq+9WvfpWNfuQaTuHfP/vZzy5gPyw468t5f++992Z+jNg77rjjshGZFEZUMpKyqHD8n3zyydmfGFEWR9zmP7vccstlI7SnUUa13zj7xPf2u971rmykGYURjM94xjOy7wVGnjAC5e677w6//e1vw1lnnTU3cwGffc973hPe+MY3jrM5P9tBgTqvW23hIRRhpC6FkdRdCM54YJJRt/EegdHRjB596lOfOjeK7Q9/+EM4//zzw3e+853s3oLC3/nOZ+SpJWSjlxnF/JznPCf73xYFFFBAge4KGJx1t22tmQIKKLCAQD444wNMAfb6179+qJbBWTcPJqZoecc73pFV7oADDgjbb799pYr+6Ec/yjrW6VRnpMs+++yTdRwXTR/GZwhrP/zhD2dT3xDaHnvssXPBTboDBmeVmmPshdJgrA1Tj41dgY4vkE7VWKZ9rrrqqix4YZpTykc/+tHwkpe8ZCwlrgdf/OIXs2W4TtB5nhYCue9///tZ+M00ggRmfS5pQDWf4IzpL7kO8/AC108eZiCUKHq4hesnAeYJJ5yQ0TMCiNC0bMdmX8772MFJ4Mv0WmVKOo3jfNqzzLaqfqbO9ktHnj/2sY/Nwl+mZBxU6FRnmXicEjRWndq5av1drj0CTV63mqx1+t3bheCM6Vf57mYkHd/dPOz2yle+MhuBW1SYap3PcL5T+CwzRVhCNn0tgSMPYBxzzDEDH67QSgEFFFBg9gUMzma/Da2BAgooUFogBmd0wtEpx0gFgg4624a9k8XgrDTxTH2QJ095WpxC5+szn/nMsfefH428T4kp25jGhWnahnW4xQ3wA5537HEM8m4kpjnLv7PB4Gzs5qi0QJ0dsJV2wIWGCowbnLEy3jFDeM1IEkbEMJJknGmWeKCC7VJ++ctfLtCxxnuzCIt41xbBed9LHcEZDxW86lWvmps+i/eo8N+jCiOiGAVE4QGGd7/73aMWyf7eh/OeUVJ8j/CezaOPPjpsuummpWz6FJxx/0dAS1lrrbWy99kutdRSI51+9rOfZaMs4yjUsqHkyBX7gZkSaPq61SROl4IzroVML8g0g7zLmAcKnvWsZ43k5B7i5S9/efjNb36T/W789re/XWqK/5Er7sAHeDcs03hy3WQK4EHvGu1AVa2CAgoo0GsBg7NeN7+VV0CBvgnE4IzRAXTIHX/88RnB05/+9PC5z31uIIfBWTePlDo6BZhq8TOf+UwGVLajN2oy6iwed7yLae+9934AtMFZM8ddHzrQm5GczFaqBGfsyRFHHJGN5qRwbg1731l+z9P3qdBhli/xuyS+82wyNZ+dtdYRnDGCj9EAFKYEY4RDmcL5S4coDyEwZeOPf/zjbNrMUaUP5/1RRx2VvZtohRVWCD/4wQ+yDuMypS/BGccMoyYYncqoEzp/mRq0bGF6RzrSKYS3j3rUo8ou6uc6ItD0datJtjrukeva3/m+4+zwww/PpqqlDJvCuWh/+U5505velP2JB9523333uqo10+thRB5T/zL6m+8MvoctCiiggALdEzA4616bWiMFFFBgoEAanPFSed77wRQrlGFT9Q0LztIOJt65su666xZuf9g64n7xrpxPfOITgSeZCVR4cTVPijNigh8nPE1PBxjluuuuC5/+9KezTkJ+vPD+nfXXXz+boqxoH9K6X3rppeGiiy7Klud//+1vfwsrrbRSNlc901Yy5dWwwhQdTE3EKA7e+cH7u+iw5H1evK+L/yuaWivtXGW0F/vNj1neU7PEEkuE//iP/wiHHHLIWO9M4f0trIsRW1deeWXmxVRdvGOM97LwTiRGFaalaMrO9O9lp6TBYaONNsreucO7XuhAKdNhG7fFcjzxyjsU2Gemg0lHxeSDM7wZ0cb7ffih+shHPjKrI4EA72AZVC6//PJsii7eQ8S7G2ibZZddNqy33nrZaLn4zptByzOajmnr6HjluKPejI5jeUb1cNwUlbTThePtnHPOCTyhSod1PF6oB8cwHbqM3MFxUImdwPy96FzDn85Ljmne3cN+4ooTowlf/epXZyOE0hLfc1W0zXgclH3HGefrV77ylcyZY5s6rbzyytm7chiZsMYaaxRWLXYIcfzzThGuSZz/5513Xrj55puz45cnerfddttsysFB72SkbRktwXKsg3d44Mw7tziWXvrSl451fMadHXaNi+cSxyHT9fAeP45Rpi+kDfDnesQT23RQVylVgzPagHcVMiKA95SxX0Vtz3uMOC4pw46HUfteFKI1ea2s6zzl+OJ9YWeffXa4/vrrMz8CAdqPa006FWJ6bhT5cM5z/pUpcfpLPsv2OW/KFr5HHvKQh4T//M//zKbMi1NvDXrHWZ3nPd9fXB8vuOCC7HzFi+sM4R/XHM7dQYXjg2sG13befcl3Kd/lePPdwLvzhl0Th/lwbeXaznmw0047zb2rqIzpuMEZ7xvj/OLaQwjFtYf7Fu5JmN6M9wMNKuwn1z3eE/rrX/86+w7nfoD7EI4BDPIzApRpvzL1xH3XXXfNPrrNNtuEgw8+uMxic5/hPpKRfFxnuD6n73sr8/3H/V56z8AxwFTOHLdczzmWWCfHNY6DRrOXDRW4P+R7nJJ/ICDeb3DPxP0q91S0KfvEPSJtwHX+DW94wwKj40ehcc8Sp8vlu/CDH/zg0HOC72vui9Jr87BtTPI8GlW3SVy30m1WPSbiOcz1hHsSZkdgZgXukzhfuUYSEm+xxRbZg4TpcZheN4vqH4+dNh7jRfvL9YhzlNFmPDjJvSYuZQvnIaPMue7gxXUtLfF6xEh1jll+T/I7DmOu5XvuuWf2gGYsXOe57vP9yOh4fgsw8wT3avwO5Fwpmj5y2Ds70/3h3N1tt92yf8pPtRuPC9r7F7/4Rfj973+fBYr8/sCHe0bqR33z9SzyevOb35x5jnoAtay1n1NAAQUUaJ+AwVn72sQ9UkABBSYmkA+P+DFJxxY/ivgRxZPD+Y51dqbJ4GzttdfORkqwT/lCJ/zJJ5+cBSc86RxfWp1+jnn7GWWx8cYbP2DxtO68J4iAqmgbBDe8E4gfh0WFEIbl6SAcVJ785CdnTx/mLdMffbwf5OMf/3j2roFYeI8Q7xMqWwjK+HEYw8+i5dgHApe0w6mu4Ox///d/s04kCp17++67b9ldn/sc71iLdWbqmDSESoMzfpTmO/7jSvjBTWixwQYbLLB91slUKkVtHT/MNunAK5rKjiklmcaKzsxBhc5Zjqd8R0TaqcL0abxIPC10juHGVGsUXthO2wwqHJMEtXT20MkaCz/2d9lllywwG1boMMEp7ZAv0wE7KjgjvKUj8Bvf+MbAzROi0UEbO2nTD6bBGfVg+lA6AosKZoQ8+YCWTnvWTQfMoLL66qtnoyOHTUtbtGzZ4AxXrh10hBcVOqYZoTnudD5VgzP24cUvfnHW8YsX4W16jMe2n1Rw1uS1sq7zlM48nqa/6aabCtuQ7yBCohiG1RWccQ4RbnLcE5rQVmVHRw075ycdnHE94fuFkUtFhRCSsILvqfxxT+c13x/XXnvtwCpwzeJ+YNDDCcPqno6EIYhPO26HXihDyL6/qRdl1DvOTjnllKyjmCnNBpWXvexl2Yjs/HWL6dPoeB127aYDme9WwvdYyly3R9WRv3Ot5cEbSt3v6Snz/Rffz8f3M9+PPMyU3hPl68D35fvf//7s/UxF3yH8G98F+Wmf42fLBmc8rEF4XVQI8pitgQc9xinxWsxDO/FhmaLluQ/n/pZCIL7lllsO3cykz6NhG5/UdYttzveYSIMzghHuD3i4oqhwPvGgEKEJpUpwNuger+ljvKh+BNxxxDkPgMRr2zjH77DPxusR13MeAuCeNBa+AwjI4gOJ7Av31ASZgwrvWSSQz79bte7gjLbhmkCQV1RoUx7YHFa4H+D6z/cbDyKUfVCmLnvXo4ACCigweQGDs8kbuwUFFFCgNQL54IwdS0MxAoA4lUe6000FZ4wauuOOO7LRX2984xuzEVz8uGKfeBKbQuciHbI8Xc9neBKZH++MwDn99NOzz9AxzsibdGRKrDs/bvhBTgcyo8sI2Ojo5gcPo8j433RY8r8JwNJy1VVXhe233z4L7OhYoROHEVe8D4TOC8IMnnSkEG7QoUYnaCzpjz72jaCFDkWesuQJVtbJU9tlCmEZ248/UGk7luXHKU/YY/Gd73wnWxXrJXCMP255wpI68GOWzgXKhz70ocybQsCQ7veg/Rmnc3HQOmg3OsIo+SlgYnAW24yAjOliCMgISHg3Cz/SKXSw8t/pj1Y6nwmGKXRy0XHAqCc6eXnKlR/NsdO2aOoaOuAYqRCnQONJcY43tsXIM4Ki+NJ0jks62tLO7rTjkPYmxKRTnuPzsssuC3QOELZwDNKhwwg22qmoMJKJEVOU9Md8/v0idIrQ0cY5REcuxwk/7KkvhX8/99xz5/aT84rjJXYO8LQ9x1V6HAwLzujk5BiO71hi1BHn1TrrrJO5MfKKTusYPBZN8xODM9qX4ICOYqw5t+ho5sllwrLYuZHvzGDd1BtDOkypC8cy1whGf3AexnZi5Afn9jilTHDGqJg//vGP2VPTHGccD7QNI1DoDOaJbwohLiNSxynzCc4I6GPIyvHKgwmxFAVn8TpLqE/wQIkd69QhdlhzzvLuMzqRP/WpT2Wf428c05Qmr5V1nqfUJ763kXaik5sRVRx/8QEFRkLRphS+e/g75yfBN4UgnJEpFL4HyozC5SEIglVKuv5xjpOizw7qaJzvec+20vOCenLO8l1GYf086BCvr0XnfXxPHtdMzhnOd84fziOuUXw3cA1hhB/fz8NGFRfVnfbg3Oe7noeE8mHLMNuy323puUmoigHXcbbFccG1hlENFEb484BFWtJ3CRKucR3jmONaRgcz125GO2HE91ucRrFM+5U5dnjnG6Mq2V+u1Xy31VXKfP/FQPTAAw+cm5qUduY7hNEefG9yv8cDD3xPDXKsc8RZvAZwD8T9BoEr9xtcB+M9Jvd8BFxxBoQyZtSBBycoBG/5h7viOuIILu4P+d7ie2xYmfR5NGzbk7pusc35HhPxHObY5rimDQmfuTZzTeEBPH5bxOMqjjRk23yWe7xB98jxOtfGY7yovQiheDiMMmx2kTLHcdFn4r0E5yv3PZw3PJDDdykP7sTRXzwgwLnNNY37esJkHgjj/on7zK997WvZPROFf8M3fQCxzuCMayrfN/yGYpQb99ec13xX8MBdvGcdNQU9D7PFBy3HnRK7qrfLKaCAAgo0K2Bw1qy3W1NAAQWmKlAUnPEDkR8vdDBTDj300Oy/09JUcMY26dinsyvtkCBM40cYP7Yo/LhhZMBjHvOYB+xnOnqJQCV9IjgdZcWPaDrz89P+0FnHiBd++NHxzo+4tBBM0WFFxxadgkXvAkmDoPz0UPmnWOlIobOwSuHJTkZ8Ueh848dovtBRTocShR+2mKRP/c/3/Q1pZxXTK5WZ1iS/j3R6x30n8OEJ61hicMZ/05nFNgj10sLUfByflDjVZ/x77JTkKWI6XvMjwv70pz9lU8Jw7LP++IOd5TnW6FTkhz9PvfI0ctFoTAIGntSnMOqKjtNYUt98x2dah/322y+rG4X9LBoRxagHgjk6Jgg0OE8ojLSJoRfHeOy8T9dP5zNh2M9//vPsnzFLR+eNetfRsOCMd+IwDQ+FUIoQJZ3Kjn/Hl33j/3P8cX6n06mmxxGdlVjkn+YnaGSqLs5Nws8YCrP+9DgvOg5ZhvOa85tCkDTOe3zKBGesl+OD61Jsm9gGbI8Ai0JnaXy3ZP5cGPTf8wnO4nHDugmKGbEXS1FwFv8233ecNXWtrPs8pf68ezGG1NGDTjSuT1wPOIYZMZJOSzffd5zRQcsxSuHhDDo36yjDOhrnc94TmhL0EJxz3eW+IR8QEtwTmNH5jxnXijhtI9cCru8UHibgc/nC8coITgojrmKgX9YlhkJF3+Wj1lEmOGPUOXUn7B907cOHztQ4GjcdVU3oyncm/58OfR5eyRceSIgjkhmZxkiNWEa136g6sl0eDmIfeeAmBuWjliv797LffzwYwVR5FO7p+K7NB1IEiTwoxXFHyb9PqM7gjPVzz0BoyZTXaUmPSa5xMQgrY8K1g6CQ72PuO+KxnS7LvS7fETykkAY5g9bfxHk0rG6Tum7VcUyk5zAP43Du5e+3CY35vRNH+nK9TAP2UffIbTzGi9qLh41i6Dvu6Nsyx3Y6AnbQ+9M47pkanaCJ+1ju9TfffPMFVp9OSZ5/mLPO4CxuuGhUGQ988H3D+Ui4xnU4jkbM7zD3l1z/eRgxTtldxszPKKCAAgrMjoDB2ey0lXuqgAIKzFugKDhjpQQGhDwUnvalozed6qbJ4GzQ031xHnn2kakSY2dSisITwUw9ROHpSjq1YkmDs7322muuvnnU9Anw9D1SdP7RcUPhSVg6DQcVPsfnCQHSH+Lpjz7m/Y+jdMZtWMK7ODJt0CjBuM60PoQavOcgllGdAqP2K323BU/DD3uXzaB18TRqDGrzIy3S4IyAquhHNuuNbUvnCO8viR3aPPVKB3d+asP8McNoP0aq0XkZO38JQPbZZ5/so8PeN5R2BtDpF0ccslzqS2ARp+zJW/BEfZyGiymamDomLXRs0m50vOXXw77R0cgoDTo+B01RxecIUSj56Z9GdcAOC84YJcNT57gR+hWFi2yT0QxxGsp8QJp2elL3OE1V3ilui054RnHEDi46xOI0mITJBNtFxlwfCCW5Lox6j2G6fNngjOsknZ35QscK73siqCVUI+wcp8wnOGOaO6aupeRHu00qOGvyWln3ecr7wXgfYlFJR0DkrwnzDc7SAHrYOTDOccNnJxWcxY7YUe/LYSQ213GuYWkQQIhPQFh0PYp1pCOSEJNRrFz38iPAh1kQtHDOUXgPZbz2lfUrE5xxHWXECtcjvsvzD3XEbTFSFgPO//Q7jncgxo58AjHucYoK1xWu6xybacf/qOv2qLryfcL+UHiQgfudQYV9HzR9aVyGBybShz7Kfv8RGBPAUIa9J5cRQNwD8J3LscC5H0vdwRmhVnwnWd6Eh2PiCEYe/BnnXVHxPhYrvqvyo8l48IMHaShlHkaa9Hk06hia1HWrjmMiPYcJbPJTZce6pdvivZbpeTzqHrmNx3hRm/H7Lj4YxkjJODK86LPczw2b2pxluNdPZ1dIgzPe91V0H5i+T5GZIOL9ddE+pL/X0ged6g7Oit79GveHB7zig2ijRp3xoA2jzgn8+c1hUUABBRToloDBWbfa09oooIACQwUGBWcslAYs+Y7tJoMz3neVn9ee/YvTLvG/8yNmYqX5wUJYQsl3fMS686QjP74YtVZU8u8hIxyipJ2mhDNMyzSopCEFUzXxNCIlXfe4Tyun20qDgvx7wfL7lIYyTIfF0++xjOoUGHU6MVKMjiMKHYeEgeMWprOK7/AYFJwR5tIZP+idP2k90hGTdLrydC2FTlpGNZQNTGJnCuEMhum0n/k6MlIgvn8t7TRI94v3azAiclDBAAs6JAgh05KOyuMJ96LpPOmYHraP6XtT8usY1QE7KDhLO37p0CTcHFaY+o4pI3lyl2nI4ujHtNOTEDB2eOfXRVgew2Y6LmOHZdohwwgO1pcfTTrucZl+vmxwRufvoPdb0HnPaAmuO0whOk6ZT3B28MEHz00rmA9MJxWcNXmtrPs8HfRQBu3FOR5HBTH1FSOaYplvcJaen4wmjiOFxzlOij47ieAsDYKZxi5eYwftK1MU8nBC+g5PnuTnes/UyHSyMlqX0TjjTKc4zGbUwwijXMsEZzHIJyziGjSs8N3DiFemTeb8p5444kfIx7WM6U+Z8qvoXZtF6x513R5VR0bMxQdpRo3KY7Q6DxwNK+m7Evlcme8/2p/Oa2Y+GBXesU5GV/PgAd8dHNuMBqHUGZxxjeZ+Y9AUq8xEEO+jyryDLDVLRx/nH+7ic4y8Y7QV4Q0hzqgy6fNo1PYncd2q65gocw5TP34rxIea8jNVjLpHbuMxXtRm8WE+/sbIs/xIynQZpnMe9M7K+Ln8ewTjvcSwkauET/GhlFHhHTMKMBKZkk5/WHdwxoNFg95nzWj2+CoArpNxSuoi33REH2F22Wv4qPPLvyuggAIKtEPA4Kwd7eBeKKCAAo0IDIU1YpwAACAASURBVAvOeKKYju34wua0Y7DJ4Iyn9oo6z9J9GBTSDOu8jHXPT/OWh7/77rvnprHDg1EaFJ6QHLfDm+XouI7vrkn3b9B0JmUOBKbG40lfCuscNIUIf6cTgneuMPVPfrTSqE6BUfuSBhnDnhQfth7eecAUfJR8+BJHnI2a3i59zwYda/yIpfAuN54ap1MuFoIpHPg/OuzoyCwqjOQb9YR90XJpqJv60uEeR5UVLZeGofnRezHUZnQHHXrD3kNDZyzv7GAKIt7JRFDF6Cx+zMf3bDH6giAnllEdsIOCszTQo+OSDv9hhQ6QOP1pGjCmnZ50LjPCpKgwmjS+b4vgLb7ziGOcdmb0Yix0xjM6g3fI8f+LRqGNOr7j38sEZ3TkMhI0nQo1XX+8ftB2HPPjlPkEZwQS8b15vJcrjjBh+5MKzpq8VtZ9ng4baZJ2mOc7vecbnKWjvgdN4zbOMRM/O4ngjGtLGhqW3S9GBMep9liG6RfT9w0yQpsgiWsz52x+Kuay2+FzaZg+arRA0XpHdbpzzSHo4YGFcQvBT5zOlc5YHvaIhRFIdNZy3eJczU9Zm25r1HV71H4xoo9RbJRRI2HnG5wN+v5jFBnvdaMMmmo4rUf6rqZ0RFadwRnHHtNoDyrp/Qb3QITtZQvTl9KuTP2an1qaaReZ6o3v8HHWO8nzaFS9JnHdquuYSM9hHnrgfq+ocC3nuKJwjY/v+uW/R90jl7nHq6s+ZY/xojryjrE4G8KwGRRYdj7B2bB3yMZQmNGWPPg06F6JfWCkMkE8JZ3Gtu7gbNDouGgYp5zmgag4KrbINz0H86MWR51D/l0BBRRQoP0CBmftbyP3UAEFFKhNYFhwxka44Y+dAEx3x9OxPH3bVHDGE7508heVYfsQP18mOKNTindXDCv8cCZkSEdAxdEy4zYGI+XiFHXp/hF+xdFx464zTveT74gctB7qQSC6zjrrZB0BsYzqFBi1X4cccsjc+5qGTaU4bD3p9Jr5MDEGZ6NG56VPzuffC0JIQfDEOwvyhU5KOml5lwGjHdJC2Mh7L8YtTIsX392T+g57spVt/OEPfwiEAHTEpg50tNGRR6A7qEOdjjae0GXqKgIyOkTzhdFosZO3ruAsfXqeY4GRJcMKIfQnP/nJ7CMEvzG4KdshNOxzHAOM2CiaJocOGqb24jhiitVxR7WUCc5GBWKjrr3D3OYTnKXTqeafNJ9UcNbktbLu83TYdSxth/zn5hucpWHUsGldx70eTSI4S0dzjbs/jKaJ70DkO5YHSwhACKLyhRHMjAZgZHMMyctuL51Crsy1Kb/eUcHZ7bffnoVbVUp67WN5pnvkeyO+wzVdJw8AELDwUEJ+tPR8gzO2w0MpPGjBNZHvjkGjuofVc9B1pMz3X3osDXrXXbptHgLgYQBK+l1b9juEaVCZMYDCKMi0VLnf4N6BTvNxClPU8V3N/QfTNcbzge9GviP5vuJevOhdp0XbmeR5NKpek7hu1XVMDPveTus17HOj7pHbeIwXtVmcVpa/DZq1YFRbDzvH4jWA+2geAisqcVaFMtPU8/7F+B7czTbbLHunIaXO4Izz7Fe/+tXQmRriSD1GkMV3BBfVjWlAY73zoxZHufp3BRRQQIH2Cxictb+N3EMFFFCgNoEynbd0XtAJT2Hedqa7qyM4S5/Iy09pVma/mgzOCJjoyEvfH0bnFSN46MAb9K6qooai4yu+d6rsj75RDR7fV0BnFyP0RhU6YnkHVn46plGdAqPWy+ggpqCilHlavGh96YiYtCOMz5btyCJ0isFX0X4QGDFCic4onjClwzFfONaZYi4+BRvDUzoE6GgoW3gyNU4hOKyjvWh9dI7SkZY+3cpTwjwtTCl6UphgjSkgY2cgnyMkY6onRmwwYoGngJnSKU61VVdwlk7TVKZzOg1a0xCnbKdnmc/RGYoZozroFMm/q4Nzm/M3TvFVpl1nOTiLoTnBHk95F72XJD/FGiapdb6DOZ7vvLeO0YFcB9LS5LWyyfN0ksEZfvE6TVvxXTFoqriiY5brGu3BNKdMVxpHpU4iOEtHCXPdjA+GlDmX8u/GYRmCG85Z3tFIsMY1LS2MqOacZSRE2cKoXd4bRsm/26/MOkYFZ+wzoROF/x9HOZdZN9fl/FRejPhnJD33RrRZ/uEHPs8+pe8orSM447uD6bEpjLJK36FWpi58pkxwNiiQvuyyy+beC1kmOEvfAUbQFEellPluYF/Td+UOCs7ife+g+qejYV772tdmD2yMUzjGGX1DSUe48h6uq666KpvxgJHr45ZJnEdl9qHu61Zdx0TTwVlbjvGiNuPaGqedH/a+t2HtXSY4K7qXiOtkRgkeYCsTnPEQQXynZToys+xvqPShrvz9bjwuuBfiHnFYYXp7fjtwT8/906CSPhRW9X3PZc41P6OAAgooMB0Bg7PpuLtVBRRQYCoCZQIqAhaeDOS9GxQ6U+jMiT/k86HXUUcdNfc0IC9THvRuoXQO+GkGZ0XvkEobg3rHdywxnR0/uiiMZOIHPT+26DjMv9S9TIOW/dE3al2p5aipGnkamemY8iPo2MZ8gzNGZNExf88994Tlllsu6/gcp7OX5Qm86DSko5fQKJ06MQZnaYBZZMP0X3HqwTJTHPGUND+GOQ7pcI7vc0inX4uBw3xe9j1ucJZOh8UT6XQcMAKUwI936hFQ5N9jloZRdGDRMUh756egTPelruCMaSN5IpdSZqrGtOOFkWHYxmW/8Y1vZP87/+6MtL3Ldo7GZQgLCRNoYzrmOU4po15Mnz/GZjU4G/b+QOo4qRFnTV4rmzxPJx2cpcc37w1j2sKy5Z3vfGc2QpySdsZPIjj77W9/O/demKqdsIPqRWh2ySWXZFPSUh+mr6Pwvc1IrWHTe6XrTB/q4IEIRpqOU0YFZ+mICK638d0942xj0Gd5aIdRN3wfYsB5TOEBHL5jY+hWR3CWjsxLp0Qbpx7zCc54GImOcUqZh2/SdsE8TjWZvqN32HcI00LHqXIHBWdMQxqn7ht1v8G9GNNDj1sYRcNU0nEa6nT6x1HTOpfZVl3nUZlt1X3dquuYaEtwVld9xr3/SduOIIr7Q6YtZ0pczpFxf8PMNziL5x4jLAmPh5X02pbOIsH9XHxQY9isHdw/M7KTMig442/pu3KL9ufFL35xNgX4qCn+0/e6DnvXbZnzyc8ooIACCrRPwOCsfW3iHimggAITEygTnLHxtDOFEQUESXGKv/yPAkYJMQ0d5fOf//xc6JSvxI477jj3xN40gzN+LPJjaVDAQyc74QMlnWaR/00wSMm/KyhfVzq86AjBjqkh6x5xlr4Pi/+dn2Yw3Z/06d18x9R8g7P4o5SRUJRxp5885phjAk/pUorCjBicjXq/AO/JidMlpaPWCEro5GWU4KBpjxj5FEcLpE+ax9Ff7Buhy7D37fB3pgqkvemciKM9xg3OCBJ54p/ODY5BnhBmOjA6aZli6l3vetcDDjUCP85NpnF89KMfHRgBxvSdRWVYR/CoDthB7zi77bbb5kYoMKVavA4MuoDFJ455epf3BcZO8LIdQoM+hw+dkHQWpu8nSfeDvzNVIyM5OBZ4B1LZMqvBWdqZUzQ906SCsyavlU2ep5MOztKQa1TnfXrscu3h3Ug8HEGoQijNFMuUSQRnbIfvNa5XjKhmtGs+0E/3j4dvuCYyCpbrI4VRwIRiXFsGTXlIZy+drbw7kMJ2eICgTGEkNtdzSpmRTPl1jgrO+HwMPxj5zYMYccq9ov3j3oHrFN8RhL3xOs1oaUZh4Fl07caJaz/3JZT0HZqjrttlnNgnvmeZypl68K7SYe9VK1rnfIIzvsMYYcV1mYeu4j3WoH2Po+357qDzP07hmV5zhnVcMyKGEWOUQcFZ0SjadH8I7HhvHoV31KWjAMuY85n4rrZ47HAfxwg67k8JjeOo9VHrm/R5NGr7+WtMHdetuo6JtgRnddWn7H3SoDYjQIr36kwTz/36OGW+wRnnTHzAgAcC1lxzzYGbT2czIBRnul5KOlqTh8p4UK6o8LuC3xeUYcHZsHffcR/O/TXfdzxMynSXg0qcqYXvQaZ0HOcBwnHawM8qoIACCkxHwOBsOu5uVQEFFJiKQNngjJ1L3wXBj3t+PFDynRJ05DCNImXQk92M/KBzgU6aonWU2a+6pmpk+8PeNxWnLqJjhikrY2CSTplHXegwKSpMDcePLAIbCk++0xFDqWvEGdOL8L4rCp1ehEWDynve857Ae8QovEuF0Vux1BGc0bYEIozU4zihk3TQS9jTfaTTix/vPF1PJxG+jFpLSwzO+LdBoxnx5kl53o3HKCs6jVlfOn3jsHek3XXXXVmnJSUdPUE4yvt3KMPeY0KHH6bUPz9qbtzgjG3FJ+d5wpVpMGPnRlFHA52ddMJS0uls8scC590222wzd0zmn2i/6aab5o6Los6IQcEZ2+G9ZnRA0uHHyDg60osKHctMe1O0r2U7hAZ9LnZg8w4gRmQMKhgQaI87inAWgzPqyXlB26+00krZdGz56eEmFZw1ea1s8jwddj4TBBP+U/IddQMPyII/MIVbfBq/zBSDdJzvsssuc8c9QSLnSSzDvnPmc96n9wd0KPKdV1TSaR3phOThGkoadBBiD3qwIQ1/OYZ5QKBM4aEJpqykMNqM9YxTygRn6T0J00LGB27y2yGoIVDguy59CCR96GjYCEM6u+PI93Q6xVHtV7a+6XtGH/vYx2bvgC37Tjm+x2Ondn6atrLff3zPxeleWYbpdIsKDz/wgAZBRP59rTwYgA2laEpj/p2RI4wgiWVQcMbf04Ay3RfuNxgBQ+c49xnUf9x3ZrI+vlM5Jlgf5w/3pNSP90Ny3pctkz6Pyu5H3detOo6JOoKz9NwoejCwyWO87H3SoDbjHpXzh/czEvAQLKW/B4a1Nff5zC4QpzbMj+ocdi8R15u+Q3vU6NL4m5Bl06kPmcqU+3QK93PxHj2/7zw0EaexHxacDftuSKeFHfYdx7a512KU8KjQvez55OcUUEABBdolYHDWrvZwbxRQQIGJCpQJqOIOEDzwIz5Obxb/PR+cMXqL0WQUnhjmx0bakUBnESN66EwdtI4y+1VncLb88stnL2enoz0tTBcXOx3z0wPSWUM4wTR/lEEdZYceeuhcqMYogPjUI8vUFZyxrvSHJe/Y4F0b+ZLvEGMkYfqOozqCM7bJ0/CMjsKIAIX3lvHDtWhaLTqKeKqd9qRTn8/wdGicrimtQxqcPe5xj8s6s/LvpkqnCs2PWiNc5Ic+T3/SGRjfmZBuIw1+0ydbCdQ4Bjj+2Uc6s/Kdw9SF44BRa5T8+07Kdqqk+5MeI3QOMg3loCfxGZFB5zDnGMc04VocaRLXiTFP+sYRo/w77RPPWf47HTlG5whha1qGBWfpOUMAyVPzTAWUFjp4aRtGmNBhgznvXYulbIfQoM+lT1KnbZjuA0ESxyRWo8Lm/Hk0a8EZnUt0PMZr1aAHBSYVnDV5rWzyPB12PtORvv3222eHDlNExRBtgYvyiH9gWj6mneXc5nt07733zoL7omspoT3nNtd1CqEM1/x0tMqw75z5nPdpEM6Iap7cJ3RJC1Pwsu9xqkFG2RAWULiPIPCjvOAFL8ime87XkWsv1/Drrrsu8J4zRuIMGlFbxEonMQ+w5N/tWaZNygRnjBTjHil+7xHixocw0usv11SCVUp6fUqn5+O7ie+o/EgFHljiu54AkuOBUXfxAZNR7VemnvEzHGdf+9rXsv/k4Qem+4yjA4vWwyhHHsRhBAnhLSV/z1T2+y+dfo1RKAQU+YdoOJYIhfk+pORD5fReh+OJ+4K0MHqEKRXT9xQNC84GBYhp2Mk9D1OkVi3xHi5+z7OeUTMI5LfVxHlUpn51X7fqOCbqCM7Sd2XlHzzDpcljvOx90rD2IvAi4Oe6wr0YD0BwHR40Qor7Je7xmMacd+jFwvU/DdfLBGdcJ7kmExCzbdbJuZovcTQm/86DYVxX0+spD+Yx2pn7TB5uzD+sxUMIvJs7lmHBGfvBOR3flRiXIWTnHpn7C0ZKc48/aGpL6sV9OPs0amRamXPJzyiggAIKtE/A4Kx9beIeKaCAAhMTKBNQpRs/+eSTs072tOSDM340MOKIzi0KP2oIEOj4oNOKTnI6O/jxET8zzakaGZXEDxw6+/gBud5662XTBDE1CD8QCUPopON9U4zUSAsjARg1E0ffMUUinZxMH8VT5fyI/uEPf5gtQsDDf9OZGUudwRmWBAH8sKMQBjCyin1hJBL14Uc/9eEHHyO24o/buD91BWesjx+wBC5Ml0dhWzwZ+oQnPCGzoOOLEIv9ih1WdIIecMABc+8nyx/4MTiLbYYlbcZUUjwBS504ligcX7RfOl0WnYx0DFBYB524dGwuu+yyWVjEU+50uMXgiXA3XT7tNKFTl05S/o9jB39C4vi+FPaN7afBVdlOlbTetBeBa3y3D3/LB13p5+M70Pg33gNE5yKjJDnGOe8IKXnHRlro7IsviuffqT/nAcc1o7HodKAevMSdc2FYcMb+0vESRwzgwAgEOgO5NtABRkdofGdiUWdj2Q6hQZ+jE5djjXOBdmIEGh00nAtMY0mowT5wDHLMcU2ivmVLm4IzRjtwHKeFdqNjmA4pOsZ4sju+t29Y5+6kgjP2rclrZVPn6bDzOR39Q8jPOcuxxvUv/77BUccdoyaZgip+z9CJz9P16667bnZ94tqHL0EH13oK37eMuMlf44d958znvGebTI/LNLkURtvS0chUswQ8TK/I/uBCyYcZhC1Mw8i5SaFu/Dcjyjh26YinAzReu/baa6/ANH3jlLh/7A/B1TjtUCY4Y18Yec7DMhS2w8gDvrfo1I11IMimFD1YtNtuu809VMR1m3s0wiOOHb5juFbFkRP4pvdjo9pvHCvMGYnMd1osfJ/QEcz+8L3ANYYwnu/V8847b+745PN0PLNvcXQ9/zbO9x/bJnyl8P3MfRad0VzPGU1+wgknZKPIKYwai+ZxX7n2870Z74c43mgLQmSORa7/HEvpfeiw4Iz18mAVYdvaa6+djdKhPmeddVa2SY5T7hHzo3jHMef8JbCMhYdfuH9MH27ib+k5zP11dOJvdZxH6QNK83k/U53XLeo232OijuAsHUmMPfdOmMdjs8ljvOx90qhjkAfdqEd8KJJ7Pe6XeJiJeyauPdxTETJzvMfvGNZLSMWDIfnAq0xwxvJMHc9DJTG4Y8YCfj9yznPPy/0q1xYK99mcY/FduLFe6fsMGanMPT7fudzT8xAJI5PT83xYcMY6qS/3VDzUwTWceyi+u7jeEawdf/zxc1OSF9lyfeG3F2W//fbLvscsCiiggALdEjA461Z7WhsFFFBgqMC4wRmd4nRg8MM9lqIf1vwYYto9OqmLCh0a/LCI89FPMzjjx2EcGVO0r/xwZNRMvgMyfpb3l/GjMwYBResgcOOpyfz7luoMztguT6zvuuuuWbAxqPDDkmlG6JjMlzqDM9bNU5p0Vo568TeffcpTnpJ1tNEpNajEDp04NUsc2ZX/PB18PKld9P4bnlY97LDD5oKEom3RQcbTxEVtzg9x9pMgalDhRztPreanGxunUyVdd/p+Bn7IM/3koKmz6OBgmqRhxwCd7XQ2UA86SNMpKeN24xSl6X4QoBHODgvO+DxhKaMemVJnUKFzgvOGzu/8yJKyHULDPscT0FxfYsdp0X7Quc9Ix3S6rjJfGW0KzsrsL5+hM4zRkARtg8okgzO22eS1sonzdNT5zEMUcURMNB82Bd+wtmTaJ87Zq6++emSTMwUiUxEWTXc46jtnPuc91xKmyqKO3CsMKgTZhxxyyAIhA0EI14NhdeRawYM4acAwEuT//0A6jeC4I3nKBmdsio5VvmMJsgYV3uPFKKj8dZx7JkaHxhFpg5anYxvD/LSAw9qvrFP6OYIb2rTMccdy3Esxoo7v6nwZdb6knyeMoH4EZIOOJY4FHspgBoOi6RGZ8pN3DcXAOb8/fE8SAvKwDmVQcEaAyT0FHfBFhXsp7hfyo+LG9Sa4YERNvLfIT7Ma1zcsOOMz8z2P6grO2Je6rlusa77HRB3BGSN/mQ0hhraxTeL0sk0e42Xvk8och9SH85yHxeJDNsOWI7zi/OE3ZNG7HMsGZ2yDezXuBQmjBxUevOKaysNb+cJy/C7l90ZR4fzlWstDbpRhwRnfP4TqRdcM7hf57VB0bUu3y/WfaxeBN6HkfK8LZdrPzyiggAIKNCtgcNast1tTQAEFpiowbnDGzvKEMR3N8cf9oCdSeSqRp/R46pSnzHm6mx9TzCHPU4X8exzlMu3gjKfk+eFLhx9PC/LDkScU+YHMD7L8VHP5RqPDg3Xw5DVPldP5xY8sRgYQEvLEYdGPy1GdmFUODqbiY+QVo0zoCGJfeHqTH48EJPx4HPSkfd3BWdx/nlRlZBmjsQhdGNGHD8YEl3QCxvfPDKtz7NCJT5jzhDZTQ9GpRyca3tSRp0WHvWeEgJH2ItDjqVbM6MCkE432Yoq1YU+O8yQrT/4TYDEKgCdRaV9GvlEXOsyLproZp1MldWDkEJ3NFJ6CJYQdVgiLOPc4r9g/OnDZP44BOuYIT3iaPb6HhFCFDlLaJBaOaToJWAcdE3yGDlEC8VHBWdruTIGKM9cDRjoyAmHjjTfOnv6n/YtK2Q6hUZ+L7cSUbowuiO0U94H6D3oH2zDftgdnPBVNW9JhQ4hLmzNSJD9tZ76Okw7O2F6T18pJn6ejzmemsqIzksCQc5LRmhyzMfgf9/pOxzEjOTknGXXEqGaupVyreDiDhw/4biU4G1RGfefUcd7H6yujS+mQpROS0QKM6iR4Z2T2oMK1mOs6IxvoCGVUKNdyOmqpF/cPjNSqUvBj+kD2iWtgfBdrmXWNE5zF+yS+I2h77n+4X6L96QCmjTgfOU+LCvvJQyGMnGRUNtdfHjTgfOZ7ktGGg6ZNHNZ+ZepZ9BmCKzq3Oe4YEch3Ctvhek67MtqKtqFdCZgGlVHnS9FyHEuMeuO45aEQzHiwhe3xHcJ37rDCdycPy/AdwPnIiDOCLq797C/TYY4KzriG8hAIxyWzLnC/wTnH/QbHM/dUVd5rVrTfvMM0TrXK6Jqi+o0KzljvfM6jOoMz9qWO61ZqVfWYqCM4Yz+YOYNg5JJLLsnuKbi/ZsQjoXGTx/io+58q5zv3doRnnO8c51x7aD/uj7lX4n79Gc94RnZPMeyYHyc4i/cFnOf8LsQXV675XE+43sXRX4PqxMNa3NPzO4PfYFyzCNm4H+c3HL/p4nTgw4IzRrhRLx584/uL6zYPoDCClpFxZe4X+b3Hbx2uLzygYVFAAQUU6J6AwVn32tQaKaCAAgoooIACCiiggAK9FmA0AaPxCLF48GHQu3x6jWTlM4EYIBGcDRrdLpUCCsy2QNlAtUwtebA0vreTUD7/jssy6/AzCiiggALtFzA4a38buYcKKKCAAgoooIACCiiggAJjCDDVGh2bjIBlNCCjGSwKFAkYnHlcKNB9gTqDM0YiMlUjgRnBmUUBBRRQoJsCBmfdbFdrpYACCiiggAIKKKCAAgr0WoBpbA866KBs2kSmVrMoYHDmMaBAPwXqCs6YNpmpgHnfNe9mHDSdbj+VrbUCCijQLQGDs261p7VRQAEFFFBAAQUUUEABBRQIIfA+nJe85CXZu3COOeaY7L2WFgXyAo4485hQoPsCdQVnn/zkJ8PHPvaxbIpX3pFmUUABBRToroDBWXfb1popoIACCiiggAIKKKCAAr0WuPzyy8MOO+wQ1lhjjfDNb34zPPjBD+61h5VfUMDgzKNCge4L1BGcMcrshS98YYbF+xCXX3757sNZQwUUUKDHAgZnPW58q66AAgoooIACCiiggAIKdF3giCOOyEYG7LPPPuHVr35116tr/cYUMDgbE8yPKzCDAnUEZ/vtt1848cQTw2GHHRa22mqrGVRwlxVQQAEFxhEwOBtHy88qoEBnBW699e7O1s2KKaCAAgoooIACCiiggAIKKKCAAgoosPzyS4qggAIlBAzOSiD5EQUU6L6AwVn329gaKqCAAgoooIACCiiggAIKKKCAAn0WMDjrc+tb93EEDM7G0fKzCijQWQGDs842rRVTQAEFFFBAAQUUUEABBRRQQAEFFAghGJx5GChQTsDgrJyTn1JAgY4LGJx1vIGtngIKKKCAAgoooIACCiiggAIKKNBzAYOznh8AVr+0gMFZaSo/qIACXRYwOOty61o3BRRQQAEFFFBAAQUUUEABBRRQQAGDM48BBcoJGJyVc/JTCijQcQGDs443sNVTQAEFFFBAAQUUUEABBRRQQAEFei5gcNbzA8DqlxYwOCtN5QcVUKDLAgZnXW5d66aAAgoooIACCiiggAIKKKCAAgooYHDmMaBAOQGDs3JOfkoBBTouYHDW8Qa2egoooIACCiiggAIKKKCAAgoooEDPBQzOen4AWP3SAgZnpan8oAIKdFnA4KzLrWvdFFBAAQUUUEABBRRQQAEFFFBAAQUMzjwGFCgnYHBWzslPKaBAxwUMzjrewFZPAQUUUEABBRRQQAEFFFBAAQUU6LmAwVnPDwCrX1rA4Kw0lR9UQIEuCxicdbl1rZsCCiiggAIKKKCAAgoooIACCiiggMGZx4AC5QQMzso5+SkFFOi4gMFZxxvY6imggAIKKKCAAgoooIACCiiggAI9FzA46/kBYPVLCxiclabygwoo0GUBg7Mut651U0ABBRRQQAEFFFBAAQUUUEABBRQwOPMYUKCcgMFZOSc/pYACHRcwOOt4A1s9BRRQQAEFFFBAAQUUUEABBRRQoOcCBmc9PwCsfmkBg7PSVH5QAQW6LGBw1uXWtW4KKKCAAgoooIACCiiggAIKKKCAAgZnHgMKlBMwOCvn5KcUUKDjAgZnHW9gq6eAAgoooIACCiigwPmxxAAAIABJREFUgAIKKKCAAgr0XMDgrOcHgNUvLWBwVprKDyqgQJcFDM663LrWTQEFFFBAAQUUUEABBRRQQAEFFFDA4MxjQIFyAgZn5Zz8lAIKdFzA4KzjDWz1FFBAAQUUUEABBRRQQAEFFFBAgZ4LGJz1/ACw+qUFDM5KU/lBBRTosoDBWZdb17opoIACCiiggAIKKKCAAgoooIACChiceQwoUE7A4Kyck59SQIGOCxicdbyBrZ4CCiiggAIKKKCAAgoooIACCijQcwGDs54fAFa/tIDBWWkqP6iAAl0WMDjrcutaNwUUUEABBRRQQAEFFFBAAQUUUEABgzOPAQXKCRiclXPyUwoo0HEBg7OON7DVU0ABBRRQQAEFFFBAAQUUUEABBXouYHDW8wPA6pcWMDgrTeUHFVCgywIGZ11uXeumgAIKKKCAAgoooIACCiiggAIKKGBw5jGgQDkBg7NyTn5KAQU6LmBw1vEGtnoKKKCAAgoooIACCiiggAIKKKBAzwUMznp+AFj90gIGZ6Wp/KACCnRZwOCsy61r3RRQQAEFFFBAAQUUUEABBRRQQAEFDM48BhQoJ2BwVs7JTymgQMcFDM463sBWTwEFFFBAAQUUUEABBRRQQAEFFOi5gMFZzw8Aq19awOCsNJUfVECBLgsYnHW5da2bAgoooIACCiiggAIKKKCAAgoooIDBmceAAuUEDM7KOfkpBRTouIDBWccb2OopoIACCiiggAIKKKCAAgoooIACPRcwOOv5AWD1SwsYnJWm8oMKKNBlAYOzLreudVNAAQUUUEABBRRQQAEFFFBAAQUUMDjzGFCgnIDBWTknP6WAAh0XMDjreANbPQUUUEABBRRQQAEFFFBAAQUUUKDnAgZnPT8ArH5pAYOz0lR+UAEFuixgcNbl1rVuCiiggAIKKKCAAgoooIACCiiggAIGZx4DCpQTMDgr5+SnFFCg4wIGZx1vYKungAIKKKCAAgoooIACCiiggAIK9FzA4KznB4DVLy1gcFaayg8qoECXBQzOuty61k0BBRRQQAEFFFBAAQUUUEABBRRQwODMY0CBcgIGZ+Wc/JQCCnRcwOCs4w1s9RRQQAEFFFBAAQUUUEABBRRQQIGeCxic9fwAsPqlBQzOSlP5QQUU6LKAwVmXW9e6KaCAAgoooIACCiiggAIKKKCAAgoYnHkMKFBOwOCsnJOfUkCBjgsYnHW8ga2eAgoooIACCiiggAIKKKCAAgoo0HMBg7OeHwBWv7SAwVlpKj+ogAJdFjA463LrWjcFFFBAAQUUUEABBRRQQAEFFFBAAYMzjwEFygkYnJVz8lMKKNBxAYOzjjew1VNAAQUU6JTAPgefOvX6HLDXNlPfB3dAAQUUUEABBRRQQIFxBAzOxtHys30WMDjrc+tbdwUUmBMwOPNgUEABBRRQYHYEDM5mp63cUwUUUEABBRRQQIH2CBictact3JN2Cxictbt93DsFFGhIwOCsIWg3o4ACCiigQA0CBmc1ILoKBRRQQAEFFFBAgd4JGJz1rsmtcEUBg7OKcC6mgALdEjA461Z7WhsFFFBAgW4LGJx1u32tnQIKKKCAAgoooMBkBAzOJuPqWrsnYHDWvTa1RgooUEHA4KwCmosooIACCigwJQGDsynBu1kFFFBAAQUUUECBmRYwOJvp5nPnGxQwOGsQ200poEB7BQzO2ts27pkCCiiggAJ5AYMzjwkFFFBAAQUUUEABBcYXMDgb38wl+ilgcNbPdrfWCiiQEzA485BQQAEFFFBgdgQMzmanrdxTBRRQQAEFFFBAgfYIGJy1py3ck3YLGJy1u33cOwUUaEjA4KwhaDejgAIKKKBADQIGZzUgugoFFFBAAQUUUECB3gkYnPWuya1wRQGDs4pwLqaAAt0SMDjrVntaGwUUUECBbgsYnHW7fa2dAgoooIACCiigwGQEDM4m4+pauydgcNa9NrVGCihQQcDgrAKaiyiggAIKKDAlAYOzKcG7WQUUUEABBRRQQIGZFjA4m+nmc+cbFDA4axDbTSmgQHsFDM7a2zbumQIKKKCAAnkBgzOPCQUUUEABBRRQQAEFxhcwOBvfzCX6KWBw1s92t9YKKJATMDjzkFBAAQUUUGB2BAzOZqet3FMFFFBAAQUUUECB9ggYnLWnLdyTdgsYnLW7fdw7BRRoSMDgrCFoN6OAAgoooEANAgZnNSC6CgUUUEABBRRQQIHeCRic9a7JrXBFAYOzinAupoAC3RIwOOtWe1obBRRQQIFuCxicdbt9rZ0CCiiggAIKKKDAZAQMzibj6lq7J2Bw1r02tUYKKFBBwOCsApqLKKCAAgooMCUBg7MpwbtZBRRQQAEFFFBAgZkWMDib6eZz5xsUMDhrENtNKaBAewUMztrbNu6ZAgoooIACeQGDM48JBRRQQAEFFFBAAQXGFzA4G9/MJfopYHDWz3a31gookBMwOPOQUEABBRRQYHYEDM5mp63cUwUUUEABBRRQQIH2CBictact3JN2Cxictbt93DsFFGhIwOCsIWg3o4ACCiigQA0CBmc1ILoKBRRQQAEFFFBAgd4JGJz1rsmtcEUBg7OKcC6mgALdEjA461Z7WhsFFFBAgW4LGJx1u32tnQIKKKCAAgoooMBkBAzOJuPqWrsnYHDWvTa1RgooUEHA4KwCmosooIACCigwJQGDsynBu1kFFFBAAQUUUECBmRYwOJvp5nPnGxQwOGsQ200poEB7BQzO2ts27pkCCiiggAJ5AYMzjwkFFFBAAQUUUEABBcYXMDgb38wl+ilgcNbPdrfWCiiQEzA485BQQAEFFFBgdgQMzmanrdxTBRRQQAEFFFBAgfYIGJy1py3ck3YLGJy1u33cOwUUaEjA4KwhaDejgAIKKKBADQIGZzUgugoFFFBAAQUUUECB3gkYnPWuya1wRQGDs4pwLqaAAt0SMDjrVntaGwUUUECBbgsYnHW7fa2dAgoooIACCiigwGQEDM4m4+pauydgcNa9NrVGCihQQcDgrAKaiyiggAIKKDAlAYOzKcG7WQUUUEABBRRQQIGZFjA4m+nmc+cbFDA4axDbTSmgQHsFDM7a2zbumQIKKKCAAnkBgzOPCQUUUEABBRRQQAEFxhcwOBvfzCX6KWBw1s92t9YKKJATMDjzkFBAAQUUUGB2BAzOZqet3FMFFFBAAQUUUECB9ggYnLWnLdyTdgsYnLW7fdy7ngo873nPCzfeeOPQ2l900UVhqaWWmvvM7bffHo499thw7rnnhptvvjn723rrrRd22mmnsP766w9c1w033BCOPvrocP755wfWsfTSS4cNN9wwvOUtbwlrrbXWwOWuvPLKcNxxx4ULL7ww3HXXXWH55ZcPG2+8cdh5553DyiuvPHA59vv4448Pl112WfjLX/4SVlpppbDppptm22Pb0yoGZ9OSd7sKKKCAAgqML2BwNr6ZSyiggAIKKKCAAgooYHDmMaBAOQGDs3JOfkqBxgTuvvvusMEGG4QHPehBWfA1qBA+LbHEEtmfb7nllrDDDjsEQrDFF188rLHGGll4dtttt2Xr2X///cPLX/7yBVZ19dVXhx133DHceeedYckllwyPetSjsnXw34suumg46qijwnOe85wFliMsI5C79957wzLLLBNWWWWVcO2112ZBGIHd5z73ubD22msvsNzpp58e9thjj3D//feHFVdcMSy33HLhqquuytZDgHbSSScNDd0m2QgGZ5PUdd0KKKCAAgrUK2BwVq+na1NAAQUUUEABBRToh4DBWT/a2VrOX8DgbP6GrkGBWgUYkfXqV786PPrRjw5nnnlmqXUTfl188cVho402Cocffnh4+MMfHu67775sZNdhhx0WFllkkXDqqaeGNddcc259//jHP8IWW2wRrr/++rD11luHAw44ICy22GJZiHXQQQeFL37xi1kIdtZZZ2XhWCx33HFH2GyzzbJRZm9+85vD7rvvHhZeeOFwzz33hL333jvb59VWWy2cccYZWfgWyzXXXBO22WabwHb32WefLLBbaKGFsnBvt912y/afkXEnnnhiqTrX/SGDs7pFXZ8CCiiggAKTEzA4m5yta1ZAAQUUUEABBRToroDBWXfb1prVK2BwVq+na1Ng3gJf+MIXwoc+9KEsnDryyCNHru+CCy4Ir3/967PRZ+ecc84DQi4W3nPPPcNpp52WhVYHH3zw3Pq++tWvhve///1h1VVXDd/97ncfEHIxIozwjjBrl112yYKtWD7+8Y+HY445JjzlKU/JRoilhdBtyy23zMK4Aw88MLzsZS+b+/Nee+2VhXdbbbVVFualhRFuTNfIaLvPfvaz4RnPeMbIetf9AYOzukVdnwIKKKCAApMTMDibnK1rVkABBRRQQAEFFOiugMFZd9vWmtUrYHBWr6drU2DeAoRZhFpvf/vbw9ve9raR63vPe94TvvnNb4Ztt902GymWLz/72c/Cq171qixY+8lPfjIXkBGM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ZqPR0lFivIeM95Edeuih4cUvfvECyx177LHZFJKMfmMUHIURcYRzf/nLX7J3nPGus3yJ9SL8IwRsuhicNS3u9hRQQAEFFKguYHBW3c4lFVBAAQUUUEABBforYHDW37a35uMJGJyN5+WnFZiowL777psFSyussEI444wzwpJLPnD4NNMtEpgRQB199NHZlIiEX4RgfPZ73/veAu8G22OPPbKRaYz4+uhHPzq3/4RthG6rr756NuVjfoQbYRuh29ve9rbw9re/fW65ww8/PBx33HHZiDVGv6WFgGzLLbcMjFbLj4B797vfHU4//fRshNwhhxzygOXuvPPO8PznPz/cc889A0fATRQ+hGBwNmlh16+AAgoooEB9AgZn9Vm6JgUUUEABBRRQQIH+CBic9aetren8BAzO5ufn0grUKsBILkZxEUA961nPysInQjTKlVdemY0O++1vf5uN5mKKRd4Ddv/992cjynj/GGHWxz/+8bDccsuF++67L3z605/ORoYxNeKpp54a1lxzzbn9/fvf/x622mqrcN111wVGkX34wx8OD3vYw7JtH3zwwdloMcK4s88+OyyzzDJzy/3xj38Mm2++ebjrrrvCa17zmrDXXntloRuhF1MtnnnmmWG11VYL3/nOdx4wJePVV18dtt1228DUkYRob3rTm7JpHFnfO97xjnDxxReH9ddfP/AOt2kUg7NpqLtNBRRQQAEFqgkYnFVzcykFFFBAAQUUUECBfgsYnPW7/a19eQGDs/JWflKBRgQInpgikWCLwIt3fvF+MgIzyhOf+MQsEFt66aXn9ocRXkzvePPNN2ch1uMe97hwyy23hFtvvTUL1wjCGOmVL5dffnk2Wo13nT30oQ8Na6yxRrjhhhsCI8DY9vHHHx+e/vSnL7Dc97///WwUGiEY+8G7zq699trw5z//OSy11FJZ+MU+5Auj6fbbb78s7OOdaYSCBGrUlSkhTzrppLmgsBHsZCMGZ02Luz0FFFBAAQWqCxicVbdzSQUUUEABBRRQQIH+Chic9bftrfl4AgZn43n5aQUaESAkO+GEE7JpGAnAFltssfDYxz42GyH2yle+Miy88MIL7Mftt9+eTaHI+8j+8Ic/hMUXXzw8+clPzkZ2FYVfcQU33nhjOOaYY8J5552Xjf5i1Bkj2nbeeefwhCc8YWB9f/Ob32Tbu/DCC7OgjVFp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F6+xY8dmr4KMJ+Liybgll1wy3H777d3a3XfffWGPPfbIXkn52GOPZWOUdQnOypI2DgECBAgQyC8gOMtvqAcCBAgQIECAAIHmCQjOmldzK25NQHDWmptWBNouEE+PxbBqn332yQKneHUXnL3++uvTvgd2ww03hBVWWKHbue20007hz3/+c4jfGdt3333DH//4x2nfH4unweJrHLu71ltvvRC/Pxa/a7bVVltl30KL3ztbcMEFQzxZ1t01adKksM4662R/Fb+jFsO3iy++OPu+2pe+9KUQT5Z1dz355JNhu+22y/4qfrdt4YUXLsR5RvP8ZOcrrrhaIWPphAABAgQIEGi/wNGnXtP+QWYywojv79jxOZgAAQIECBAgQIAAgd4ILLjg3L35ud8SaKyA4KyxpbfwlAVi0BO/afb5z38+O6nVt2/fbLrdBWcTJkwIX/va17K/v+2228KAAQO6XVp8TWIMyOK3zOI3zR544IEwZMiQ7LePP/74DE93xdcyxtczxm+nxW+oTT05tsQSS4Q777yz27Hi6bcYkMUrnpZbd911w/nnnx/OPffcLFC74oorum333HPPhS222CL7uzvuuCP069evkDKtuOKKM+0nhnYuAgQIECBAoBoCBx93dccnevbJ3+r4HEyAAAECBAgQIECAAAECBIoXEJwVb6pHArkE4mmteOrqpZdeygKm+B2yqVd3wVlPvjkW23/7298ODz/8cDj44IPD0KFDp31zLP5dPIkWX4/Y3fWVr3wlvPLKK+GMM87I5hVPtR122GHZ987id8+6u+I31VZdddXsr+IaYlg2atSocNZZZ2WvnIzfWevuit9pi99ri9fUb6rlwvz/GwvOilDUBwECBAgQSEdAcJZOLcyEAAECBAgQIECAAAECdRMQnNWtotZTeYFjjjkme73hfvvtFw4//PBPrae74OyTr0XsyasajzrqqLD33nuHJ554IvuGWrx68qrGH//4x9lpsN/85jdh2LBhPX5VY3wtYzx99tOf/jSccsopPX5VYzx1F18HWcTlVY1FKOqDAAECBAikI+BVjenUwkwIECBAgAABAgSqI+BVjdWplZl2VkBw1ll/oxP4lEB89eG//uu/huWXXz57JeIcc8wx0+Ds448/DmussUZ4//33s2+iDRw4sFvVQYMGhRdffHHaKxc/eVJt7NixoX///tO1i69cXH311cNHH3007ZWLMWSLr32MJ9Qee+yxbl/x+Pzzz4fNN98862/qKxennlSL48Txurvuueee7BWVse8//elPoU+fPqXtkFdffbu0sQxEgAABAgQI5BM47vTr8nVQQOuTj/jf/wOSiwABAgQIECBAgEBVBBZbbL6qTNU8CXRUQHDWUX6DE/i0QDwNds01Pf/YfQzJYlgWX6EYv9F16qmnhp122mk61vjqxPjKxxiEXX311dNe/xi/Pfb2229np8HWW2+96dp9MgC76667wmKLLRbeeuutaeFcPH221FJLTdduagA255xzZq+HnGWWWbLvqO24445h9tlnz/4s/rPr9Ytf/CIcffTR2bfdbr755lK3h+CsVG6DESBAgACBXAKCs1x8GhMgQIAAAQIECDRUQHDW0MJbdq8FBGe9JtOAQPsE4nfA4qmzGV3xFFY8WbbsssuGhRdeOKyyyirhuOOOy/7z3//932HXXXcNJ5988nTNH3zwwbDbbruFvn37hvvvvz/7Z7z22WefcPfdd4dDDjkk7L///tO1u/7667PXRXY9JRa/Qxa/R3bmmWeGbbfddrp2F1xwQTj77LOzgC0Ge/GK847h3OTJk7NvnMVvnXW9jj322PCzn/0sC/9iCFjmJTgrU9tYBAgQIEAgn4DgLJ+f1gQIECBAgAABAs0UEJw1s+5W3XsBwVnvzbQg0DGB7r5xFicTw68Ygs0333zh17/+9XTfBjvssMNCfFViPPF12mmnTZt/DNti6DZgwIDwy1/+crpXQ8awLYZuBx54YDjooIOmtTvrrLNCDPniibXLL7/8Ux4xINt6661DPK3W9QTcoYceGm688cbshNwZZ5zxqXZvvvlm2GyzzUL8ZtuMTsC1E15w1k5dfRMgQIAAgWIFBGfFeuqNAAECBAgQIECgGQKCs2bU2SrzCwjO8hvqgUBpAjMKzqZMmZKdKIvfH4th1jnnnBMWXXTREL9/dvHFF2cnw+KrEa+77rrs+2lTr/feey9885vfDH//+99DPEU2YsSIMO+882anw04//fTstFgM42677baw0EILTWv32muvha222ipMnDgx7LHHHuGII47IQrcYesVXLd5yyy1h6aWXDjfddNOnXsn41FNPhR122CHEV0fGEG2//fbLXuMY+xs+fHh44IEHwjrrrBOuuOKK0kynDiQ4K53cgAQIECBA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BWfF1lRvBAgQIJC2gOAs7fqYXToCgrN0amEmiQisuuqq4aOPPgp33313WHjhhbNZjRs3Luy4447ZaxrPPPPMsM0220yb7fPPPx++/vWvh7nmmis8/PDDiazCNHorIDjrrZjfEyBAgACBzgmk8LBdcNa5+hu5PgIp3MuCs/rsJyshQIAAgZkLCM5mbuQXBKKA4Mw+INBFYNNNNw0TJkwIl19+eVh77bWzv73gggvCOeecE2adddZw1113hYUWWmhaq2uvvTYcddRRYbnllsu+deaqpoDgrJp1M2sCBAgQaKZACg/bBWfN3HtWXaxACvey4KzYmuqNAAECBNIWEJylXR+zS0dAcJZOLcwkEYEjjjgiXH/99WGTTTYJ559/foivYtx5553Dq6++GuJrHMeMGTNtpvG02V577ZUFbfE7ZyeeeGIiqzCN3goIznor5vcECBAgQKBzAik8bBecda7+Rq6PQAr3suCsPvvJSggQIEBg5gKCs5kb+QWBKCA4sw8IdBF49NFHw7e//e3sT+PrF+NrG997773sNY3nnntu9lrGeB166KHh9ttvD5MnTw6zzz57iCfPll9++UI8n3322fBf//Vf4Q9/+EMW2C2wwAJhjTXWCEOGDAkbbbRRt2PEgC+ejBs7dmx4+eWXw/zzzx/WXHPNsO+++4Z11llnhvN64YUXsoAwvpoy9rHggguG9dZbL3zve98LK6ywwgzbPfHEE2HUqFHhvvvuCxMnTgyLLbZY+MpXvhKGDh0a+vXrN8N2999/f7jooovCI488ktktscQSYfPNN8/Gi2N36hKcdUreuAQIECBAoPcCKTxsF5z1vm5aEOgqkMK9LDizLwkQIECgSQKCsyZV21rzCAjO8uhpW1uBq666Kpxyyinhgw8+mLbGPfbYIxxzzDHT/vdWW20VYsA1xxxzZL/ddtttC/H43e9+Fw488MDw7rvvZsHdMssskwVar7zyStb/fvvtFw4//PBPjRX/7jvf+U6IIVhsE18bGcOzf/zjH2GWWWYJJ510Uthll12mm99TTz2VhXFvvvlmmG+++bKxYh/xf8d1nXfeednJu65XDMtiIPf+++9nr61ccsklwzPPPJMFYTGw+8lPfhJWWWWV6drdeOON4bDDDgtTpkwJiy++eFh00UXDX//616yfGKBF988K3QoBnkEngrN26uqbAAECBAgU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UvGkVgysDj/88LDffvt129ehhx4abrzxxjB48ODsxNlxxx0X/vu//zvsuuuuIX5vresVXx+52267hb59+4b7778/+2e89tlnn+xU3SGHHBJiWNj1uv7667N5dD0lFr9DFr9HFgO8bbfddrp2F1xwQTj77LOzgC2egotXPBEXw7nJkydn3ziL3zrreh177LHhZz/7WRb+xRCw7EtwVra48QgQIECAQOsCKTxsF5y1Xj8tCUwVSOFeFpzZjwQIECDQJAHBWZOqba15BARnefS0rbxAfPVgd9ctt9wSfvKTn4R4aioGY/FbYvEk1Nxzz519/ywGZb///e/DJZdckp3COuKII7Jvj80111y5TI4//vjsVNiyyy6bhWNdT2a98MILYZtttsnmcP7552evRIzhVwzBYtAXT791/TbYYYcdFuKrEuOJr/hdtKlXDNti6DZgwIDwy1/+MswxxxyfmnsM22LoFk/THXTQQdP+7qyzzgqjRo3KTqzFVzh+8ooB2dZbb52Ff11PwE0N/OIJuTPOOONT7d58882w2WabhUmTJs3wBFwu2B40Fpz1AMlPCBAgQIBAIgIpPGwXnCWyGUyj0gIp3MuCs0pvIZMnQIAAgV4KCM56CebnjRUQnDW29BYeBVZaaaXP/JbWlClTevytrfhNrvhKwjzX3//+9+wUVwzGtthii3DiiSdOe2Vj7DueDounveLprRjsxTHjHGPIFUPAGGadc845YdFFFw0ff/xxdgoungyLAdx1110Xll9++WnTe++998I3v/nNEMeMp8ji6bX4nbYYfp1++unZabEYxsXvqS200ELT2r322mthq622yr4Dt8cee2ShYQzdYugVX7UYQ8ell1463HTTTZ8K/p566qmwww47hPjqyBiixRN18TWOsb/hw4dn34dbZ511whVXXJGHsOW2grOW6TQkQIAAAQKlC6TwsF1wVnrZDVhDgRTuZcFZDTeWJREgQIDADAUEZzYHgZ4JCM565uRXNRWIwVlRVwyxxo0bl7u722+/PQvI/vnPf2aBVDzpFkOuGJjFa6211spOfH3yZFk84TVkyJDw8ssvZ23i6yVfeeWVEF8lGecVg7B40qvr9dhjj2Wn1eK3zuJpuuWWWy7EU23xBFgM2y666KLs221drzjHeAothmBxHvFbZ88880x45513wvzzz5+FX9294jK+pjGG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PRAgQKAsAcFZWdIzHkdw9tk1OOKII8J1110XVl555XDBBReEfv36ZQ2uvfbacMwxx4QPP/wwjB49Omy44YZtLabgrK28OidAoCoCgrOqVMo8CRAgQICA4MweIFAXAcFZCGvsfXzlyyk4q3wJLYAAgQYJCM46X2zB2Yxr8Oyzz4att946TJkyJdx4441h+eWX/9SPzz777CxMW3fddcPll1/e1mIKztrKq3MCBKoiIDirSqXMkwABAgQICM7sAQJ1ERCcCc6K2Mt1CB+LcNAHAQIEeiIgOOuJUnt/Izibse+5554bzj///PDlL385XHbZZdP98OWXXw5f/epXQ58+fcKdd94ZFl988bYVS3DWNlodEyBQJQHBWZWqZa4ECBAg0HSBFB62e1Vj03eh9RchkMK97FWN+SvpxFl+Qz0QIECgLAHBWVnSMx5HcDZjm3322SfcfffdYf/99w+HHHJItz/cdNNNw/jx48OZZ54Ztt1227YVVHDWNlodEyBQJQHBWZWqZa4ECBAg0HSBFB62C86avgutvwiBFO5lwVn+SgrO8hvqgQABAmUJCM7KkhactSK92WabhRdeeCGMGDEiDB48uNsu9thjj3DfffeFgw46KBx44IGtDNOjNoKzHjH5EQECdRcQnNW9wtZHgAABAnUSSOFhu+CsTjvKWjolkMK9LDjLX33BWX5DPRAgQKAsAcFZWdKCs1ak11prrTB58uQwatSoMGjQoG67iIHZrbfeGoYMGRKOP75934oVnLVSQW0IEKidgOCsdiW1IAIECBCosUAKD9sFZzXeYJZWmkAK97LgLH+5BWf5DfVAgACBsgQEZ2VJC85akV555ZXDxx9/HC699NKwwQYbdNvFYYcdFm644YbsRFo8mdauS3DWLln9EiBQKQHBWaXKZbIECBAg0HCBFB62C84avgktvxCBFO5lwVn+UgrO8hvqgQABAmUJCM7KkhactSK96qqrhg8++CCMHj06bLjhht12ceihh4Ybb7wx7LLLLuEHP/hBK8P0qI3grEdMfkSAQN0FBGd1r7D1ESBAgECdBFJ42C44q9OOspZOCaRwLwvO8ldfcJbfUA8ECBAoS0BwVpa04KwV6YEDB4a33nqrR69qjN86O/bYY1sZpkdtBGc9YvIjAgTqLiA4q3uFrY8AAQIE6iSQwsN2wVmddpS1dEoghXtZcJa/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Z5111h/dW7z/bbbZJsS31R78BtzBBx8czj///BDfkDv++OP/6HN33nln2HrrrYffbJvvDbgxJxnjbEy6xIYABCAAAQjkJVDDYTvGWd45Jdo0CdSQyxhnuvYwznSGRIAABCBQigDGWSnS8/eDcbZyc4BxtnLc+BQEuiFw8sknh5NOOmkYz1FHHRV22mknc2xLliwZ3iiLvz8WzawPfehD4YlPfGKIv392+umnD2+GRRPuvPPOC+utt97yePfee+/wu2nxzbb4FtkxxxwTHvOYxwxvh33gAx8Y3haLZtyFF14YHve4xy3/3C9+8Yvwmte8Jtx1111ht912C4ceeuhgukXTK37V4le/+tXwtKc9LVxwwQV/9JWMP/rRj8J2220X4ldHRhPtbW972/A1jjHeAQccEK666qqw6aabhrPPPtscc+4GGGe5iRIPAhCAAAQgMB6BGg7bMc7Gm18iT4dADbmMcabrDeNMZ0gECEAAAqUIYJyVIo1xlps0xlluosSDQEMEomH1kpe8ZPgdsYc97GHhec973oJ3v9VWWy3/isX4hlf8esdbb711MLGe9axnhfiVj7fffvvwu2jRCItvej3479prrx3eVou/dbbqqquGddddN9x0000hvgEWzbaPfexjYYsttljhc5dccsnwFlo0wVZfffXht85+/OMfh9/85jfhsY997GB+xXt48F/8msb3v//9IZp98TfTnvSkJ4VoqEUTL34l5Gc+85nh35X+wzgrTZz+IAABCEAAAitPoIbDdoyzlZ8/PgmBZQRqyGWMM12PGGc6QyJAAAIQKEUA46wU6fn74Y2zlZsDjLOV48anINAFgeuuuy7suOOOyWN58Fcv3nHHHcNXKF588cXh5z//eXjUox4VNt544+HNrrnMr2Ud3XzzzeGUU04Jl1566fD2V3zrLP4+2d577x023HDDee/n+uuvH/q74oorBqMtvpUWjb999913eONsvr/4Zlk05K655prhLbU111wzLF68OOyzzz7hCU94QvL4czbEOMtJk1gQgAAEIACBcQnUcNiOcTbuHBN9GgRqyGWMM11rGGc6QyJAAAIQKEUA46wUaYyz3KQxznITJR4EIACBBAIYZwmQaAIBCEAAAhCohEANh+0YZ5WIgdtomkANuYxxpksI40xnSAQIQAACpQhgnJUiTT+tE1gUv8LtAX+LlsTv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QipHEeDNEOPMzpOUFhT1KZT6bkMu6wfFNTDkjbO+85TRlSFQQy6zLutz7X3YrtYpOgEiRALe+Uwu6zr0zuUcNbNOof8IGGf9zzEjzEMA4ywPR6JAYDQC//mf/xn233//cPfddw99zGecRfPrqquuCi9+8YvDCSecEP7kT/4k3H///cObXf/0T/8UVllllXDeeeeF9dZbb/m9/v73vw9/8Rd/EW688cbwute9Lhx55JHhkY985GBi/eM//mP41Kc+NZhgX/va1wZzbNnfL3/5y/CqV71qeMvsb/7mb8KBBx4YHvawh4Vf//rX4bDDDgtf/epXw9Oe9rTwH//xH4P5tuzvf//3f8PrX//6EPs94ogjBsNu0aJFg7n3d3/3d8P9xzfjzj777NF4zhcY48xGrhakUy+kchQB3gzjGFovSGGoGxb204IWFgF0qOuwBoYYZ5bSuQ4Bm0ANudz63samPH4L78N2tU4Zn9A0evDOZ3JZ15l3LueomXUK/UfAOOt/jhlhHgIYZ3k4EgUC2Qnce++94aMf/ejwVYh/+MMflsefyziL5lp8I+3Rj350+PrXv/5HJlf84Nvf/vbwxS9+cTCtPvCBDyyP9W//9m/h3e9+d1h77bXDV77ylT8yueIbYW9+85sHM2ufffYZjK1lfx/60IfCKaecEp7//OcPb4g98C+abttss81gxh199NHhDW94w/LLhx566GDebbvttoOZ98C/+IZb/LrGaBB+4hOfCC960YuyM10oIMaZjVstSKdeSOUoArwZxjG0XpDCUDcs7KcFLSwC6FDXYQ0MMc4spXMdAjaBGnK59b2NTXn8Ft6H7WqdMj6hafTgnc/ksq4z71zOUTPrFPqPgHHW/xwzwjwEMM7ycCQKBLIS+MlPfhL22GOP8LOf/Sw89KEPDQcccEA455xzws033zznG2fveMc7whe+8IWw3XbbDW+KPfjv6quvDrvssstgrF1++eXLDbJojMXfG9t3332HPh78F822aLo94xnPGN4iW/a3ePHicMstt4Rjjjkm7Ljjjit8Lv7O2oknnji8/fbxj398uP7b3/42bLbZZiEagvFrHLfYYosVPnf44YeHc889N+y8887D228l/zDObNpqQTr1QipHEeDNMI6h9YIUhrphYT8taGERQIe6DmtgiHFmKZ3rELAJ1JDLre9tbMrjt/A+bFfrlPEJTaMH73wml3WdeedyjppZp9B/BIyz/ueYEeYhgHGWhyNRIJCVwLe//e2w++67h0022WT4SsONNtoovOIVr5jXOHv1q18dbrjhhnDUUUfN+Ttm8asRY6z77rsvfPrTnw4veMELhq9xjP8uGllnnHFG2HLLLVcYQ/yts6233nr499/61reG3ySLv5v2spe9bPh38Ssc42+iPfhv2f3Hr338r//6r8H8i/+Mv8EWf28t/mbbA7/CcdnnP//5z4d3vetdKxh1WeHOEwzjzKasFqRTL6RyFAHeDOMYWi9IYagbFvbTghYWAXSo67AGhhhnltK5DgGbQA253PrexqY8fgvvw3a1Thmf0DR68M5nclnXmXcu56iZdQr9R8A463+OGWEeAhhneTgSBQJZCcQ3zqIRttVWWy2PO59xFg2w5z73uYMpNt+bXDHIss8fd9xxYfvttx/eGItvjsW/iy66aPhNsgf/xa+IjLHjP88888zhjbH4hlp8Uy2aYdEAi7+d9uC/+GZc7C/+XXzxxcNXQcY34uKbcU95ylPCJZdcMievK664Iuy2227D76Vde+21Qx+l/jDObNJqQTr1QipHEeDNMI6h9YIUhrphYT8taGERQIe6DmtgiHFmKZ3rELAJ1JDLre9tbMrjt/A+bFfrlPEJTaMH73wml3WdeedyjppZp9B/BIyz/ueYEeYhgHGWhyNRIDA6gfmMszvuuGP574F96UtfCuuvv/6c97LDDjuE733veyH+ztiee+4Zvvvd7y7//bH4Nlj8Gse5/jbffPMQf38s/q7Za17zmuG30OLvna2++uohvlk219+vf/3rsOmmmw6X4u+oRfPt9NNPH35f7TnPeU6Ib5bN9Xf99deH173udcOl+Lttj3/847Nwne8+Hxh8gw2eu2Bfhx3771nuZWWDvGGNa1f2o9k+94K3vVeKBcMQWmcYBeCtRRhKaTh8WGWo3wERvJ+H5HIeDR7zru3zBCIKBCZMgOdhH+vyf33s/a4qZm/jin9559753HqdUsMseucytUoZFay++qplOqIXCDROAOOs8Qnk9qdDYD7jLP4O2stf/vIBxIUXXhjWWWedOaHEr0mMBln8LbP4m2ZXXXVV2HXXXYe23//+9+d9uyt+LWP8esb422nxN9SWvTn25Cc/OXzzm9+cs6/49ls0yOLfWWedFV74wheGk08+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05lrRto4gYWMszi0+IbXrrvuGm699dbBxHrWs54VbrvttnD77beHRYsWDUZYfNPrwX/XXnvt8LZa/K2zVVddNazGz2yHAAAgAElEQVS77rrhpptuCvENsGi2fexjHwtbbLHFCp+75JJLhrfQogm2+uqrD7919uMf/zj85je/CY997GMH8yvew4P/4tc0vv/97w/R7Iu/mfakJz0pREMtmnjxKyE/85nPDP+u9B/GmU1cLUinXkilbGCtWfBmGO+v9YIUhrphYemU6zYBdKjrsAaGGGe21mkBAYtADbnc+t7GYlziuvdBsVqnlGBk9dEDQ+98JpctldnXvXWYo2a2RzluixYYYpyNqwGi90MA46yfuWQknROwjLM4/DvuuGP4CsX4e2Q///nPw6Me9aiw8cYbD292zWV+LUN28803h1NOOSVceumlw9tf8a2z+Ptke++9d9hwww3nJXv99dcP/V1xxRWD0RbfSnvJS14S9t133+GNs/n+4ptl0ZC75pprhrfU1lxzzbB48eKwzz77hCc84QkuM4lxZmNXC9KpF1I5igBvhhhndp6ktKCoT6HUdxtyGeOsb4UzOgikE+B5qD8P02mP19L7oFitU8Yjkx65B4be+cweO11v87X01mGOmlmnoEVogSHGmTbHfHo6BDDOpjPXjBQCEFiAAMaZLQ+1IJ16IZWjCPBmiHFm50lKC4r6FEp9tyGX9YPiGhjyxlnfecroyhCoIZdZl/W59j4oVusUnYAeoQeG3vlMLrevwxw1s05Bi+CdyykMMc60OebT0yGAcTaduWakEIAAxpmkAbUgnXohlbKBtSbImyHGmTVDadcp6tM49dyKXMY461nfjA0CsxDgeag/D2fhPVZb74NitU6JXLy12MP+EIbvGSvFisX1zuUcNXMxWPN01AJDjDNvldB/KwQwzlqZKe4TAhAYlQBvnNl41YJ06oVUjiLAmyHGmZ0nKS16OBhJGSdt5idALusHxTUw5I0zshwCOoEacpl1WZ9H74NitU7BONPXZRjmYahnoxbBO5dz1MwaAf3TLTDEONPnmQjTIIBxNo15ZpQQgIBBAOPMlohakHofjHgfiuQoArwZYpzZeZLSwluLai6njHHsNt4FqcqQXNYPl2pgiHE2dqYTfwoEashl1mVdaa2vy5g++roMwzwM9WzUInjnco6aWSOgf7oFhhhn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PCIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAuVCQIGzcsGuLxUBERABERABERABERABERABERABERABERABERABERABERABEcg2AgqcZdsV0fGIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiUCwEFzsoFu75UBERABERABERABERABERABERABERABERABERABERABERABEQg2wgocJZtV0THIwIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiUC4EFDgrF+z6UhEQAREQAREQAREQAREQAREQAREQAREQAREQAREQAREQAREQgWwjoMBZtl0RHY8IiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiIAIiEC5EFDgrFyw60tFQAREQAREQAREQAREQAREQAREQAREQAREQAREQAREQAREQASyjYACZ9l2RXQ8IiACIiACIiACIiACIiACIiACIiACIiACIiACIiACIiACIiAC5UJAgbNywa4vFQEREAEREAEREAEREAEREAEREAEREAEREAEREAEREAEREAERyDYCCpxl2xXR8YiACIiACIiACIiACIiACIiACIiACIiACIiACIiACIiACIiACJQLAQXOygW7vlQEREAEREAEREAEREAEREAEREAEREAEREAEREAEREAEREAERCDbCChwlm1XRMcjAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiJQLgQUOCsX7PpSERABERABERABERABERABERABERABERABERABERABERABERCBbCOgwFm2XREdjwiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIQLkQUOCsXLDrS0VABERABERABERABERABERABERABERABERABERABERABERABLKNgAJn2XZFdDwiIAIiIAIiIAIiIAIiIAIiIAIiIALlQuCjjz5yDRs2LJfv1peKgAiIgAiIgAiIgAhkBwEFzrLjOugoREAERCCMwI8//uh+//13988//4T9/u+//3a//fabW716tXvllVfcLbfcInIiIAIiIAIiIALlQID3Me/l7bffvhy+XV8pAiKQLIE777zT9e7dO+HmmzdvdnfddZebPHmyW7x4ccLttYEIiED5Efh//+//uRdeeMHWxF9//bX79ddf3Xbbbedq1arlWrdu7Tp06OAqVqxYfgeobxYBERABEch5Agqc5fwl1AmIgAjkE4GZM2fagn39+vVJndZ///vfpLbTRiIgAiKQiwTmz5/vKleu7P71r38lPPy33nrLffPNN+6cc85JuK02EIFkCXjH3FZbbeWOPvpo+9imTZvcdddd51566SX7/yZNmrhBgwa5vffeO9ndajsREIEyJMA75IorrrCfWPbxxx+7/v37mwMe0xy7DC+Qviougb/++ksBoGKE1qxZ4y677DL36aefOt7Txe3//u//XIMGDdzYsWNd9erVNcJEQAREQAREIC0CCpylhU0fEgEREIHME3jn9GFjeQAAIABJREFUnXfcBRdckNSOd9xxR8ukGzZsWFLbB3kjKvc2btxoVQHRFlaezR577BFkTFHPXZmcpTckCI7jdKeqtFWrVpYdK4skgLOToMS0adMS4jn99NMtcEawTSYCmSDw559/ugsvvNDG1DHHHGOJLdgNN9zgSHQJtV122cU9//zzrkqVKpn4au1DBFIiQKXUvHnz7J3CM5OEA1kRgQMOOMDmgVdffbW7+OKLw9DAbsyYMe7BBx+0bQiSX3755a5Hjx5CWEICzMFfffVVd/zxx5dwT/n18XvvvTdiHMY6w6+++spdc8017sknn8wvCCU4mz/++MN16tTJff75527rrbe2pJYDDzzQqs1+/vlnt2jRIqtCY7v69eu7GTNmuC233LIE35j/H33zzTfd3Llz3dKlS40hcxzWz1OmTHFnn32222mnnfIfQppnSKXjsmXLTBWouK+BoDfPQa8WNGnSpDS/RR8TAREoLwIKnJUXeX2vCIiACBQjcOWVV5ojvWnTpu6qq65ylSpVcjiCTz31VFu8M+F67LHH3OzZs20R8Pjjj7sttthCHGMQYAEwatSopDKGyUokY1FWRECZnJkZDUuWLHEjR450++yzj7v22mttp2S1d+vWzSRlMO5jnCLnn39+Zr40R/fCgnPDhg1hR3/kkUeaM2T06NExz4pF6rfffusuueQSW7B+8MEHOUqgdA8bNjhEkkkk4D0kc2769OlWSbbNNtvYe5gx9tNPP7mWLVsWOuH5b2Tg3n77bdumV69egUdHwLtChQoWSPRVePvvv39KXPRejo6Ld/P48eMdyT4+AITDjnfKd999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PbhixQp3zz33OKosZMkRSKWvXrw9LlmyJLkvDNBW9G299957rSLqo48+sjP3AYrivQkffPBBCwrJCpJbMBgRMEP2kqAZPeNkqRHgvuzYsaPNs3v37m2BMebayLwhgUmlLnNunpvMu0kGpGpFVkSA6pTTTjstTM42Hh9VM4fTYV3HfIaKnvr162toJUkA1ZXu3btboBajT5x/BqIswr1MWwju2y233NLWflTnMv641+l1FlQr6TovqNx03iKQDwQUOMuHq6hzEAERyEkCBMiYsLKg9H2NmKhWrFjRFvNMXo8++mjrs0DllG9YnpMnW4oHzeITGSgm9vSloDIPeRgqz+BJI16fcU0gDaYsAHAe43ziOoTKHpXioebMrrt06eLmz59vGe0DBgxwu+22W84ce3kdKAuqfffd13ilY4MGDbIqU5xOQTX1iSu9K09yAc8/GphHkxvkmYgsHo5Omp/LCgggBwWXF198UUhSIEDlaCasRo0amdhN3u0DuUH6nPXp06fw3OhlSKCMgDgVAlQNBL1CIPTCz5o1y7399tvGjd57PshI7z2Ci1TnMff2SgV5N2gyfEK+t17x3SJVRiUV7xl+SGq79NJLM/ztub875nwEfli3oCTCezdRghp9C2UFBAj8wI3kn1GjRmneksTAIDCG0grzQO5Tqva4P728IO8Qfng2sq4ePHiwjck33njDgmZ8nudoUHvKKXCWxCDTJiKQpwQUOMvTC6vTEgERyG4CZMo9++yz1nzXOzHJxuzQoYP97LTTToUnwERNgbPY1/OOO+5wkyZNsmoAAmBebofqsx49eljmHAZzMmRDbfbs2bYY4LPjx4/P7kFThkd36KGHmrzlm2++aT31ZIkJZEKajG8JcuCspItSVe1FH6e8Z0jEwFmCk5jnIEFKJH/pU7No0SLrE4LU21577WX/LTngApYtW7a09zHOIpkIZDsBkoZwKDNmU+0nl+3nlsnjI0kFaW+CaPTp+u233woDaTVr1rQAGj9UpcliE2COSAVktHkLHAne4oCXRRJAXnX16tVWLcqcW5YaAcYdCRootRDoqV69uiX5EYiMZmyDLGaQjWA3wbCGDRtaklRoPy58EYcffrgjGZV5IsmloSxpdUBrAyrOQhM2gsSTNQqV3vBJ1yQvmi45fU4EypeAAmfly1/fLgIiEFAC3slONjsVUPT2wWEZ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TWYbjcciAryHN23a5J544gmTm5eJgAiIgAiIQGkRUOCstMhqvyIgAiIgAiKQAwSmTJnihgwZklZTcqRRZs+e7S688ELXt2/fHDhbHaIIBJuAnHOpXX/6IRFEmzNnjjmHvbMdBzsSUMhB1a9fP7WdamsRSJOAlxakAhJJQcagLDUCa9eutT5SL7/8cmFgnD3ssssurk2bNlYZVbNmzdR2GrCtFy9e7M4880yreuRZGM8qVqzoDj74YDd16tSAUSo63WTeu8lsE1iAUU68UaNGrnr16lYtKhMBERABERCB0iSgwFlp0tW+RUAEREAERCDLCfzyyy/uqKOOMvmiyy+/3F1xxRVJHfGECRMcklE4RQieydGUFDZtJALlSkDOufTw00vq2WefdU899VRhP0cfRKNH5Omnn26VaFWrVk3vC/QpEUiCAL33ttxyS/fmm2+6rbbaKolPaJN4BKhY4d7edtttXeXKlQUrSQL9+vWz52G9evVc9+7dXaVKldxVV13ljj32WOshhzwjCQfvv/++a968uXvooYeS3HN+bsZ7l/fEgAEDYp5gly5dEm6DbLCsgECLFi3snn3xxReFRAREQAREQARKlYACZ6WKVzsXAREQAREQgewngJQT8kU4ghs2bGgZ182aNSvs5+PPgCAb/T/I1v7www/t16kE27KfhI5QBPKbgAJnJb++X3zxhclDzZo1y/3www+2Q56dBDSQciShQCYCpUGAwNnee+9tQQmZCJQXAZ5za9asMYlvxiNGtR5VezwbMSrRevbsaf3N7rjjDlM1CKrx3o0mDZoKD/UrDKeFTCg9Cq+//np33nnnpYJS24qACIiACIhASgQUOEsJlzYWAREQAREQgfwkQDDszjvvLOxTQTPtGjVqWO8A32QbR0moNE/Xrl0L+63kJxWdlQjkFwEFzjJ3PXkuvvbaa+Yofuutt+w5iXMzyL18MkdXe4pGgGqe5cuXWwJLSR3xIiwC6RIgwWq33XYzuUtvl1xyiaPv2QcffGBKBNi3337rjj76aKsOYo4ZVOO9W1LTuyWc4BtvvOHGjx9vSXysVUgqqFatWtyej8kqapT0WunzIiACIiAC+UVAgbP8up46GxEQAREQARFIm8CCBQvcwIED3dKlS+Pu46CDDnJXXnmlO/zww9P+Ln1QBESg7AkocFY6zNevX28yjk8//bR7/vnnS+dLtNfAE3j88cdN7o2fc845J/A8YgHYf//9S8xGFT6xEdJfCplGxqO32267zU2bNs2q0OrWrVv4e+QbUSsguSCoRgAxE0aASFZAwFfx+R57ySQSKKlFo0cEREAERCAdAgqcpUNNnxEBERABERCBPCZAAG3evHkWQPvpp5/cNtts43bddVdXp04d64emxXseX3ydWk4TIHATz/r37+9q167tqA6IZ6ecckpOc9DBi0C+Erj22mstOHvuuee6tm3bWpVFvH5ne+yxR76iiHleqvAp3UuOVOM///xjFbfepkyZ4oYOHepGjRplvc68dezY0X322WeFvSFL98i096AQoCdcqjZ16tRUP6LtRUAEREAERACVh/8LxfB//8+nbQiOCIiACIiACIiACIiACIhAzhBQL5WcuVQ6UBFImYCv8qbCMZkKi6BWTb333nsps33xxRfd9OnT7XO4Qw499FBHMEgWSaBXr14OXvfee69r1aqVbUBFWffu3V3nzp3dTTfdZL/7/fff7e8EdpFxlImACIiACIiACIhArhFQ4CzXrpiOVwREQAREQAREQAREQASiEMhEpQW7XbJkifiKgAhkGYF07m/dy/Ev4o8//ugGDRrkXnjhBQuYUWHfp08fq+iTRSdApRlVy7A6++yz3dVXX+3+/PNP17p1a/frr7+6wYMHuwMOOMACa0g3NmnSxGQcZSIgAiIgAiIgAiKQawQUOMu1K6bjFQEREAEREAEREAEREAEREAERCBSBdHolSVo59hCZM2eOu/nmm933339vQbOmTZu6IUOGuL322itQ4yqdkyW4iGRoxYoV3SeffGIVkMg0Tpw4MaIacvTo0a59+/bpfI0+IwJW+Ykk7THHHCMaIiACIiACIlDmBBQ4K3Pk+kIREAEREAEREAEREAEREAEREAEREIGyJkDvVqrMZs+eXVhl1rt3b5dO36SyPvZs+r4nn3zSJBhHjhxph/XXX3+5AQMGOHptEojcYostXNeuXV3fvn2z6bB1LDlGgErbxo0bu4cffjjqka9atcptvfXWbuedd86xM9PhioAIiIAI5AIBBc5y4SrpGEVABERABERABERABERABERABERABNIm8Morr1gPLl9lhozg0KFDVWWWNtHID65du9YRzKhZs6bbaaedMrhn7SqIBBIFzvi75ECDODJ0ziIgAiJQNgQUOCsbzvoWERABERABERABERABERCBPCGwefNmN2/ePPfPP/+Y065y5cp5cmY6jWwn8Nlnn7mFCxe6n3/+2ap8qO6JZVdccUW2n06ZHN/GjRut9xY9t+BVqVIl6811/vnnl8n360tEQATSI5BM4CxeRVp636pPiYAIiIAIiEABAQXONBJEQAREQAREQAREQAREQAREIAqBNWvWuPHjx7s99tjDXXzxxbbFsmXLXLdu3dx3331n/0/QDKe8erBoCJUmgb///tv179/fgj/J2n//+99kN83b7ebOnesGDhzo1q9fb0EznOz0MqtVq1bennOmTozKsUwYz0+ZCKRDQIGzdKjpMyIgAiIgApkioMBZpkhqPyIgAiIgAiIgAiIgAiIgAnlD4IcffnCnnnqqQ3qsTZs2FkDDqFJ59913yUB02223nfvll19cxYoV3TPPPOPq1q2bN+evE8kuAtOmTXO33nqrHVSVKlUs8ENvn3g2derU7DqJMjwaKvLg9eyzzxZWmfXq1cvuX+5dWWIC+++/f+KNEmwB608//bTE+9EOgklAgbNgXnedtQiIgAhkCwEFzrLlSug4REAEREAEREAEREAEREAEsobAqFGj3MSJE93ee+9tlT6tW7d2K1ascO3atXMVKlRw06dPd40aNXJ+u06dOlnlmUwESoPAaaed5qggo/LxqquusjEoi07g9ddfdwMGDHDr1q2zoNnBBx9svcxq164tZCkQIGiRCfv/7d0LkJZV/Qfwn3IV1C5SkEKGdhElRBRKQEVQURBGtCaZklTy0gil4I3Ky5hB2RDmEGGTgBJpaIiJGBfJW5NykYvKNcFLhIDgBJpIAv85z7/dCUGWd/fd3fd993NmmJZ3z3Mun/OEM/vdc87y5cvz0Yw26qCA4KwOLropEyBAoIAEBGcFtBiGQoAAAQIECBAgQIBAYQik3WarVq2KmTNnZkc1pnLfffdlx7z9750q//nPf6JLly7ZLqBZs2YVxuCNouQEUkh78MEHxzPPPGPH1D5Wd9iwYTF16tQsMEvh4je/+c1sl1muQaPjBSPWrl2bl/8fHXHEEXlpRyN1T0BwVvfW3IwJECBQSAKCs0JaDWMhQIAAAQIECBAgQKAgBE466aRo0aLFbndKXXHFFfH000/H9773vfjud79bPs60G+iVV16JxYsXF8TYDaL0BDp27BitWrWKKVOm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DkBvBAgQIFByAoKzkltSEyJAgAABAgQIECBAgAABAgQIECBQ/AJvv/12TJw4MZYsWRIjR46Mj33sY+WT2rhxY1x99dXxwgsvlH+Wvv/9738/+vfvX/yTNwMCBAgQqDUBwVmt0euYAAECBAgQIECAAAECBAgQIECAAIG9CSxdujS+853vRArPUpk2bVocffTR2dfvv/9+9O3bN15//fXYtWtXNGjQIJo0aRL/+te/4oADDoghQ4bEZZddBpYAAQIECFRKQHBWKTYPESBAgAABAgQIECBAgAABAgQIECBQHQIpGDv77LNj3bp1cdhhh0WPHj2ynWTp61RGjx6d/UkhWa9eveLHP/5xFpw9/fTTWWiWnn/00Ufjc5/7XHUMT5sECBAgUOICgrMSX2DTI0CAAAECBAgQIECAAAECBAgQIFBMApMmTcrCsOOPPz7GjBlTHpilOaQdZl27do1NmzZFs2bNYs6cOdGwYcPy6U2ePDluvvnmbMfZ0KFDi2naxkqAAAECBSIgOCuQhTAMAgQIECBAgAABAgQIECBAgAABAgQiC72effbZ3Y5nLHNZuHBhdodZ2m12ySWXxPXXX78b2bZt26JTp05x1FFHxdSpU3ESIECAAIGcBQRnOZN5gAABAgQIECBAgAABAgQIECBAgACB6hLo1q1bHHjggdlusg+XX//61/HLX/4yC87GjRsXJ5988h51+vTpExs2bIjnn3++uoaoXQIECBAoYQHBWQkvrqkRIECAAAECBAgQIECAAAECBAgQKDaBdu3aRZs2beIPf/jDHkMfOHBg/PWvf40GDRrE/Pnzo1GjRnvU+frXvx7Lly+PF198sdimbrwECBAgUAACgrMCWARDIECAAAECBAgQIECAAAECBAgQIEDg/wU6duwYrVq1iilTpuxGsmPHjux77733XrRv3z7uv//+vZJ17949tm/fnh33qBAgQIAAgVwFBGe5iqlPgAABAgQIECBAgAABAgQIECBAgEC1CZx77rmxfv36mDt3bnYkY1lJfx8wYED22VVXXRWDBg3aYwxr166NHj16xHHHHRd//OMfq22MGiZAgACB0hUQnJXu2poZAQIECBAgQIAAAQIECBAgQIAAgaITuP3222PSpElx1113xZlnnlk+/htvvDGmTp2aBWcPP/xwHHPMMXvMbfjw4TFx4sS45JJL4vrrry+6uRswAQIECNS+gOCs9tfACAgQIECAAAECBAgQIECAAAECBAgQ+K/AihUrol+/ftG4ceMYNmxYdizjnDlz4s4778xqfNQxjSlU++EPfxi7du2KBx98MNt1phAgQIAAgVwFBGe5iqlPgAABAgQIECBAgAABAgQIECBAgEC1CqTdZmPGjNntqMYUiDVt2jQmT54cRx99dHn/d999d8yePTteeumlLDRLoduIESOqdXwaJ0CAAIHSFRCcle7amhkBAgQIECBAgAABAgQIECBAgACBohVId5SNHTs23njjjWwOHTp0iJtuuinatGmz25x69uwZr732WvZZ+nrkyJFRv379op23gRMgQIBA7QoIzmrXX+8ECBAgQIAAAQIECBAgQIAAAQIECOxDYOvWrVkQdtBBB+211s033xzbt2+Pvn37RufOnVkSIECAAIEqCQjOqsTnYQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgVIREJyVykqaBwECBAgQIECAAAECBAgQIECAAAECBAgQIECAQJUEBGdV4vMwAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIBAqQgIzkplJc2DAAECBAgQIECAAAECBAgQIECAAAECBAgQIECgSgKCsyrxeZgAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQKBUBARnpbKS5kGAAAECBAgQIECAAAECBAgQIECAAAECBAgQIFAlAcFZlfg8TIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgUCoCgrNSWUnzIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQqJKA4KxKfB4mQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAoFQHBWamspHkQIECAAAECBAgQIECgCAV27NgR9erVK8KRGzIBAgQIECBAgAABAqUoIDgrxVU1JwIECBAgQIAAAQIECBSAwJQpU2LYsGHZSJ566qlo0aJF+ah27twZDzzwQKxevTp+9KMfFcBoDYEAAQIECBAgQIAAAQIRgjNvAQECBAgQIECAAAECBAhUi8C+grPrrrsu/vSnP0W/fv3ipz/9abX0r1ECBAgQIECAAAECBAjkKrBHcJZrA+oTIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQKEWBA0pxUuZEgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAIFeB/wOGOxnER4aiMgAAAABJRU5ErkJggg==" width="967.7778034151343">





    [Text(0, 0, 'Atlanta'),
     Text(1, 0, 'Boston'),
     Text(2, 0, 'Chicago'),
     Text(3, 0, 'Dallas'),
     Text(4, 0, 'Detroit'),
     Text(5, 0, 'Houston'),
     Text(6, 0, 'Los Angeles'),
     Text(7, 0, 'Miami'),
     Text(8, 0, 'Minneapolis'),
     Text(9, 0, 'New York'),
     Text(10, 0, 'Philadelphia'),
     Text(11, 0, 'Phoenix'),
     Text(12, 0, 'San Francisco'),
     Text(13, 0, 'Seattle'),
     Text(14, 0, 'Tampa')]






    Text(0.5, 0.98, 'Number of Observations in Different Cities (Test Group v.s. Control Group)')



Given the visualized distribution of views across different cities, the differences of views in Los Angeles (52513 obs in the control group and 0 obs in the test group), Seattle (13004 more obs in the test group) and Philadelphia (20234 more obs in the test group) are quite offending. 

To see if there's any factor other than city will influence the watching rate (the value of watched), we can run a logistic regression model with the "watched" column as the label column and check the coefficient estimation (as well as their test results).


```python
formula = """
watched ~ C(tv_make)+C(tv_size)+C(uhd_capable)+C(tv_provider) + total_time_watched + 
C(test) + C(gender) + age + C(city)
"""
model_fit = smf.glm(formula, merged, family = sm.families.Binomial()).fit()
print(model_fit.summary())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                watched   No. Observations:               364826
    Model:                            GLM   Df Residuals:                   364792
    Model Family:                Binomial   Df Model:                           33
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -76142.
    Date:                Sun, 27 Sep 2020   Deviance:                   1.5228e+05
    Time:                        22:25:25   Pearson chi2:                 3.65e+05
    No. Iterations:                     7                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================================
                                              coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------------------------
    Intercept                              -2.9096      0.053    -54.705      0.000      -3.014      -2.805
    C(tv_make)[T.Philips]                  -0.0027      0.032     -0.084      0.933      -0.066       0.061
    C(tv_make)[T.Sony]                     -0.0196      0.025     -0.796      0.426      -0.068       0.029
    C(tv_make)[T.Toshiba]                  -0.0376      0.030     -1.261      0.207      -0.096       0.021
    C(tv_size)[T.40]                       -0.0299      0.030     -1.011      0.312      -0.088       0.028
    C(tv_size)[T.43]                        0.0082      0.029      0.279      0.780      -0.049       0.066
    C(tv_size)[T.50]                        0.0074      0.029      0.252      0.801      -0.050       0.065
    C(tv_size)[T.55]                        0.0054      0.029      0.184      0.854      -0.052       0.063
    C(tv_size)[T.60]                        0.0431      0.029      1.481      0.138      -0.014       0.100
    C(tv_size)[T.65]                        0.0301      0.029      1.031      0.302      -0.027       0.087
    C(tv_size)[T.70]                        0.0288      0.029      0.987      0.324      -0.028       0.086
    C(uhd_capable)[T.1]                    -0.0122      0.018     -0.669      0.503      -0.048       0.024
    C(tv_provider)[T.Cox]                   0.0038      0.021      0.181      0.856      -0.038       0.045
    C(tv_provider)[T.DirecTV]               0.0050      0.027      0.187      0.851      -0.047       0.057
    C(tv_provider)[T.Dish Network]         -0.0295      0.027     -1.090      0.276      -0.083       0.024
    C(tv_provider)[T.Time Warner Cable]     0.0057      0.019      0.305      0.760      -0.031       0.043
    C(test)[T.1]                            0.0028      0.017      0.167      0.868      -0.030       0.036
    C(gender)[T.Male]                       0.0081      0.015      0.552      0.581      -0.021       0.037
    C(city)[T.Boston]                       0.0278      0.044      0.634      0.526      -0.058       0.114
    C(city)[T.Chicago]                     -0.0138      0.041     -0.336      0.737      -0.095       0.067
    C(city)[T.Dallas]                      -0.0246      0.043     -0.567      0.570      -0.110       0.060
    C(city)[T.Detroit]                     -0.0414      0.048     -0.857      0.392      -0.136       0.053
    C(city)[T.Houston]                      0.0220      0.044      0.501      0.616      -0.064       0.108
    C(city)[T.Los Angeles]                  0.7609      0.036     21.210      0.000       0.691       0.831
    C(city)[T.Miami]                        0.0027      0.048      0.056      0.956      -0.092       0.097
    C(city)[T.Minneapolis]                  0.0427      0.048      0.893      0.372      -0.051       0.137
    C(city)[T.New York]                     0.0199      0.036      0.551      0.582      -0.051       0.091
    C(city)[T.Philadelphia]                -0.7725      0.052    -14.907      0.000      -0.874      -0.671
    C(city)[T.Phoenix]                      0.0326      0.046      0.702      0.483      -0.058       0.124
    C(city)[T.San Francisco]               -0.0267      0.044     -0.601      0.548      -0.114       0.060
    C(city)[T.Seattle]                     -0.7021      0.058    -12.004      0.000      -0.817      -0.587
    C(city)[T.Tampa]                        0.0136      0.047      0.287      0.774      -0.079       0.106
    total_time_watched                      0.0001      0.001      0.125      0.900      -0.002       0.002
    age                                    -0.0003      0.001     -0.502      0.616      -0.001       0.001
    =======================================================================================================


Note: the fitted model above is a simplified one where only main effects are investigated (no interaction terms considered). The test summary shows no factor other than the Intercept, C(city)[T.Los Angeles], C(city)[T.Philadelphia] and C(city)[T.Seattle] have statistically significant coefficient values (-2.9096, 0.7609, -0.7725, -0.7021). This means:

(1) being a viewer from Los Angeles drives watching rate up; 

(2) being a viewer from Philadelphia & Seattle are drives watching rate down; 

(3) there's no strong enough evidence to show that adding the new commercials featuring local Mayors (see the p value of C(test)[T.1]) can help to change the watching rate. 

#### Part 3. Design an algorithm that returns FALSE if the problem happens again in the future.

Since the experience tells only city is a factor that can influence watching rate, the algorithm required should check if the number of views (sample sizes) in the test and control groups are similar for each city. (If yes, we can at least do the AB test on city level.) There are three test to be done:
(1) One-sample z-test of proportion to see, given sampled data from all cities, if the test group and control group have similar sizes.

(2) Chi-square test of proportion, to test if distribution of views (among cities are different) are different between test group and control groups. 

And False will be returned if any of the test show significant difference between the test and control groups.


```python
class test_data_validity:
    def __init__(self, df):
        self.data = df
        
    def test_of_equal_sizes(self):
        n1 = self.data[self.data.test == 1].shape[0]
        n2 = self.data[self.data.test == 0].shape[0]
        n = n1 + n2
        p = 0.5
        z = (n1 - n * p) / np.sqrt(n * p * (1 - p))
        p = stats.norm.cdf(z) * 2 if z < 0 else (1 - stats.norm.cdf(z)) * 2
        return z, p, False if p < 0.05 else True

    def test_of_same_proportions(self):
        sizes1 = self.data[self.data.test==1].groupby("city").size()
        sizes2 = self.data[self.data.test==0].groupby("city").size()
        obs = pd.concat([sizes1, sizes2], axis=1).dropna().values
        chi2, p, dof, ex = stats.chi2_contingency(obs)
        return z, p, False if p < 0.05 else True
    
    def data_sanity_check_result(self):
        _, _, res1 = self.test_of_equal_sizes()
        _, _, res2 = self.test_of_same_proportions()
        
        if res1 and res2:
            return True
        else:
            return False
```


```python
t = test_data_validity(merged)
```


```python
t.test_of_equal_sizes()
```




    (-13.64454146789537, 2.175777393001849e-42, False)




```python
t.test_of_same_proportions()
```




    (-24.36396265268454, 0.0, False)




```python
t.data_sanity_check_result()
```




    False



Data sanity check code usage example:


```python
obj = test_data_validity(merged)
obj.data_sanity_check_result()
```




    False


