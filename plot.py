#!/usr/bin/python3

###############################################################################
# Imports

# Data visualization
import matplotlib
matplotlib.use('agg')  # noqa
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import logging

###############################################################################


class Plot():
    """
        Class containing the different plots that can be performed. Most of the
        following methods take pandas dataframes as input.
    """

    def __init__(self):
        """
            Change matplotlib's default style. Other attributes are set in
            matplmatplotlibrc file
        """

        # Set pyplot style
        plt.style.use('seaborn-dark-palette')

        # Define a list of different linestyles in case a plot has multiple ax
        self.linestyles = ['--', '-.', ':', '-']

        self.values_per_day = 288
        self.values_per_week = 2016

        self.figs_dir = 'figs/'

        # Setup logging parameters
        logging.basicConfig(filename='logs/logs.log', level=logging.INFO,
                            format='%(asctime)s %(processName)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def get_location_name(self, name):
        """ Return the entire name of a location from its code. """
        names = {
            'SA': 'South America',
            'NA': 'North America',
            '??': 'Unknown',
            'AF': 'Africa',
            'OC': 'Oceania',
            'EU': 'Europe',
            'AS': 'Asia'
        }

        return names[name] if name in names else name

    def plot_all(self, df):
        """ Plot every graph for all time series in df. """

        # Plot number of IP, acf, and pacf of a DataFrame over time
        self.plot_number_of_unique_ip(
            df,
            'ip_all.pdf',
            title='Number of unique IP addresses per time series'
        )

        for ts_code in df.columns:
            self.plot_autocorrelation(
                df[ts_code],
                'acf_{}.pdf'.format(ts_code),
                title='Autocorrelation of the number of unique IP addresses'
            )
            self.plot_partial_autocorrelation(
                df[ts_code],
                'pacf_{}.pdf'.format(ts_code),
                title='Partial autocorrelation of the number of unique IP'
            )

        self.plot_distplot(
            df,
            'distplot.pdf',
            title='Distribution of the number of unique IP addresses'
        )

    def plot_number_of_unique_ip(self, df, filename, **kwargs):
        """ Plot the number of unique IP addresses over time """

        fig, ax = plt.subplots()

        # Plot a line plot to visualize the data
        ax = df.plot(style='-', **kwargs)

        # Different style for each plot
        for index, line in enumerate(ax.lines):
            plt.setp(line, ls=self.linestyles[index % len(self.linestyles)])

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        labels = list(
            map(self.get_location_name, [l.split('_')[0] for l in labels])
        )
        ax.legend(handles=handles, labels=labels, loc='best',
                  shadow=True)

        # Axes
        ax.set_xlabel('Time')
        ax.set_ylabel('# of IP')

        # Save figure
        plt.savefig(self.figs_dir + filename)
        plt.close()

    def plot_autocorrelation(self, df, filename, n=288, **kwargs):
        """ Plot autocorrelation of the first n points. """

        fig, ax = plt.subplots()

        ax.set_xlabel('Number of lags')
        ax.set_ylabel('Correlation')

        # Plot an autocorrelation
        plot_acf(df.values, lags=n, zero=False, **kwargs)

        # Layout
        fig.autofmt_xdate()
        plt.tight_layout()

        # Save the result
        plt.savefig(self.figs_dir + filename, bbox_inches='tight')
        plt.close()

    def plot_partial_autocorrelation(self, df, filename, n=40, **kwargs):
        """ Plot partial autocorrelation of the first n points. """

        fig, ax = plt.subplots()

        ax.set_xlabel('Number of lags')
        ax.set_ylabel('Correlation')

        plot_pacf(df.values, lags=n, zero=False, **kwargs)

        # Layout
        fig.autofmt_xdate()
        plt.tight_layout()

        plt.savefig(self.figs_dir + filename, bbox_inches='tight')
        plt.close()

    def plot_distplot(self, all_time_series, filename, **kwargs):
        """ Plots distplot of every time series in df. """

        df = all_time_series.copy()

        # Remove very small time series
        df = df[df.columns[df.median() > 15]]

        # Reorder columns based on their mean (if .index is not specified
        # the code will only return one value for each column)
        df = df.ix[:, df.mean().sort_values(ascending=False).index]

        # Figure settings
        fig, ax = plt.subplots()

        ax.set_xlabel('Time series')
        ax.set_ylabel("Number of unique IP addresses")

        # Boxplot
        df.boxplot(return_type='dict')

        plt.savefig(
            self.figs_dir + filename,
            bbox_inches='tight',
            width=25
        )
        plt.close()

    def cdf(self, data, filename, logx=False, **kwargs):
        """ Create a cdf of data and save it to filename using title """

        data_size = len(data)

        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)

        x = bin_edges[0:-1]
        y = cdf

        f = interp1d(x, y)

        xnew = np.linspace(0, max(x), num=1000, endpoint=True)

        # Plot the cdf
        plt.plot(x, y, 'o', xnew, f(xnew), '-')

        # Plot x axis as log
        if logx:
            plt.gca().set_xscale('log')

        plt.legend(['data', 'linear'], loc='best')
        plt.title("Interpolation")
        plt.ylim((0, 1))
        plt.ylabel("CDF")
        plt.grid(True)

        plt.savefig(
            self.figs_dir + filename,
            bbox_inches='tight'
        )
        plt.gcf().clear()

    def plot_predictions(self, x, real, x2, predicted, std, potential_outages,
                         outages, title, filename, training_end, validation_end
                         ):
        """ Plots predictions at all the different steps. """

        fig = plt.figure(figsize=(10, 3))

        plt.xlabel('Time')
        plt.ylabel('Number of unique source IP', fontsize=10)
        plt.grid(True)

        real_values, = plt.plot(x, real, linestyle='-', color='C0',
                                label='Original time series', zorder=10)

        plt.legend(
            loc='upper left'
        )

        for key, items in outages.items():
            for event in items:
                plt.axvline(event[0], color='black', linestyle='-', lw=3, zorder=4)
                plt.axvline(event[1], color='black', linestyle='-', lw=3, zorder=4)

        ymin = 0 if np.min(real) > 0 else min(np.min(real), np.min(predicted))
        plt.gca().set_ylim([ymin, np.max(predicted) * 2])
        plt.gca().set_xlim([real.index[0], real.index[-1]])
        plt.tight_layout()
        plt.savefig('real-' + filename + '.pdf')

        plt.axvline(training_end, color='black', ls='--', lw=2)
        plt.axvline(validation_end, color='black', ls='--', lw=2)

        plt.gca().text(
            training_end - pd.DateOffset(days=3.5),
            max(predicted) * 2.05,
            'Training',
            horizontalalignment='center'
        )
        plt.gca().text(
            training_end + pd.DateOffset(days=3.5),
            max(predicted) * 2.05,
            'Validation',
            horizontalalignment='center'
        )
        plt.gca().text(
            validation_end + pd.DateOffset(days=(real.index[-1] - validation_end).days / 2),
            max(predicted) * 2.05,
            'Test',
            horizontalalignment='center'
        )

        fig.autofmt_xdate()

        plt.legend(
            loc='upper left'
        )

        ymin = 0 if np.min(real) > 0 else min(np.min(real), np.min(predicted))
        plt.gca().set_ylim([ymin, np.max(predicted) * 2])
        plt.gca().set_xlim([real.index[0], real.index[-1]])
        plt.tight_layout()

        plt.savefig('sets-' + filename + '.pdf')

        predicted_values = plt.errorbar(x2[:2016], predicted[:2016],
                                        yerr=std[:2016], fmt='-', ms=7,
                                        color='C2', ecolor='0.2',
                                        linewidth=2,
                                        label='Predicted time series', zorder=5)

        plt.legend(
            loc='upper left'
        )

        ymin = 0 if np.min(real) > 0 else min(np.min(real), np.min(predicted))
        plt.gca().set_ylim([ymin, np.max(predicted) * 2])
        plt.gca().set_xlim([real.index[0], real.index[-1]])
        plt.tight_layout()
        plt.savefig('valid-' + filename + '.pdf')

        predicted_values[0].remove()
        for line in predicted_values[1]:
            line.remove()
        for line in predicted_values[2]:
            line.remove()

        predicted_values = plt.errorbar(x2, predicted, yerr=std, fmt='-', ms=7,
                                        color='C2', ecolor='0.2',
                                        linewidth=2, zorder=5)

        ymin = 0 if np.min(real) > 0 else min(np.min(real), np.min(predicted))
        plt.gca().set_ylim([ymin, np.max(predicted) * 2])
        plt.gca().set_xlim([real.index[0], real.index[-1]])
        plt.tight_layout()
        plt.savefig('preds-' + filename + '.pdf')

        for index in potential_outages.index:
            if potential_outages.loc[index] > 1.7:
                plt.axvline(index, color='C3')

        ymin = 0 if np.min(real) > 0 else min(np.min(real), np.min(predicted))
        plt.gca().set_ylim([ymin, np.max(predicted) * 2])
        plt.gca().set_xlim([real.index[0], real.index[-1]])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('outages-' + filename + '.pdf')

    def plot_roc_curve(self, global_results, filename, **kwargs):
        """ Plot a ROC curve using x as TP and y as FP """

        fig, ax = plt.subplots(figsize=(6, 6))

        # Axes
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        plt.xlim((0, 1))
        plt.ylim((0, 1))

        colors = ['b+', 'r^', 'gx']
        threshold = filename.split('-')[2].split('.')[0]
        labels = [
            '< ' + threshold + ' IPs',
            '> ' + threshold + ' IPs',
            'All time series']

        points = {}
        thresholds = {
            .7: ['2 sigma - 95%', 's'],
            1.0: ['3 sigma - 99.5%', 'h'],
            1.7: ['5 sigma - 99.99%', 'd']
        }
        for t in thresholds:
            points[t] = [[], []]

        for index, i in enumerate([0, 2, 1]):
            norm_x = []
            norm_y = []

            for threshold, values in global_results[i].items():
                tp = values[0]
                fp = values[1]
                tn = values[2]
                fn = values[3]

                tmp = fp + tn if (fp + tn > 0) else 1
                norm_x.append(fp / tmp)

                tmp = tp + fn if (tp + fn > 0) else 1
                norm_y.append(tp / tmp)

                if threshold in thresholds:
                    points[threshold][0].append(norm_x[-1])
                    points[threshold][1].append(norm_y[-1])

            norm_x.append(1)
            norm_y.append(1)

            # Plot entire line
            ax.plot(
                sorted(norm_x), sorted(norm_y), colors[i], linestyle='-',
                label=labels[i]
            )

        for key, value in points.items():
            logging.info('Values for {}: {}'.format(
                thresholds[key][0],
                [(round(tpr, 2), round(fpr, 2)) for tpr, fpr in zip(value[0],
                                                                    value[1])]
            ))
            ax.plot(value[0], value[1], thresholds[key][1], zorder=20,
                    c='black', label=thresholds[key][0], markersize=10)

        # Plot a line to be compared against
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        # Reorder the legend to be prettier
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[1], handles[0]] + handles[2:]
        labels = [labels[1], labels[0]] + labels[2:]

        plt.legend(handles, labels, loc='lower right')
        plt.tight_layout()

        # Save figure
        plt.savefig(self.figs_dir + filename)
        plt.close()
