#!/usr/bin/python3

###############################################################################
# Imports

# Cython
import pyximport; pyximport.install() # noqa

# Argument Parsing
import argparse

# Data manipulation
import pandas as pd
import numpy as np

# Plots
from plot import Plot

# ARIMA methods
from arima import Arima, queue_iterator

# Getting CAIDA IODA data
from ioda import Ioda

# Sending mails
from mail import send_mail

# Type hinting
from typing import Iterator

# Utility
import sys
import configparser
import json
import logging
import os
from math import floor, ceil
import pickle
import multiprocessing as mp

###############################################################################
# Class 'Chocolatine'


class Chocolatine():
    """
    Main program that connects everything.

    This program parses all arguments and sets every configuration file before
    analyzing the ground truth dataset on Internet outages. Each network event
    is imported from the 'gt-validation.conf' and is analysed with the
    parameters contained inside the corresponding section.
    """

    def __init__(self):
        """ Initializes logging options and parses arguments. """

        # Get and set arguments
        parser = self.parse_arguments()
        args = self.format_args(parser.parse_args())

        # Setup logging parameters
        logging.basicConfig(filename='logs/logs.log', level=logging.INFO,
                            format='%(asctime)s %(processName)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        if args.debug:
            logging.getLogger().addHandler(logging.StreamHandler())

        logging.info('Started: {}'.format(sys.argv))

        self.continents = args.continents
        self.countries = args.countries
        self.autonomous_systems = args.autonomous_systems
        self.duration = args.duration
        self.save = args.save
        self.plot = args.plot
        self.mail = args.mail
        self.gt_validation = args.gt_validation
        self.gt_comparison = args.gt_comparison
        self.validation = args.validation
        self.results_dir = 'results/'
        self.gt_dir = 'combined_ground_truth/'

    def parse_arguments(self):
        """ Parse options. """

        parser = argparse.ArgumentParser(
            description="perform ARIMA on CAIDA UCSD network telescope"
        )

        # Generate ROC curves
        parser.add_argument(
            '-V',
            '--validation',
            action='store_true',
            help='Use the ground truth to generate ROC curves.'
        )

        # List continents that one wishes to fetch
        parser.add_argument(
            '-C',
            '--continents',
            nargs='+',
            choices=['SA', 'AF', 'OC', 'EU', 'NA', 'AS', '*', 'AN', '??'],
            help='Input which continents are going to be selected from the '
            'list. Inputing * will fetch all of them except for ?? and '
            'Antartica.'
        )

        # Specify if all countries have to be analyzed
        parser.add_argument(
            '-c',
            '--countries',
            action='store_true',
            help='If set, will analyze all countries.'
        )

        # Specify if provider ASes have to be analyzed
        parser.add_argument(
            '-a',
            '--autonomous-systems',
            action='store_true',
            help='If set, will analyze a set of ASes that belong to different '
            'ISPs.'
        )

        # Select duration of analysis.
        parser.add_argument(
            '-d',
            '--duration',
            choices=['week', 'month', 'year'],
            default='month',
            help='Select the duration of the analysis.'
        )

        # Option stating that every single continent, country, and AS has to be
        # recorded.
        parser.add_argument(
            '-A',
            '--all',
            action='store_true',
            help='Select everything from CAIDA IODA'
        )

        # Plot number of unique source over time, acf, and pacf
        parser.add_argument(
            '-p',
            '--plot',
            action='store_true',
            help='Plot figures'
        )

        # Save every dataset in a pickle file
        parser.add_argument(
            '-s',
            '--save',
            action='store_true',
            help='Save datasets into pickle files'
        )

        # See messages written by login
        parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            help='See messages written by logging module.'
        )

        # Send a mail with a screenshot
        parser.add_argument(
            '-M',
            '--mail',
            action='store_true',
            help='Send mail when results have been generated'
        )

        # Analyze the ground truth for calibration
        parser.add_argument(
            '-g',
            '--gt-validation',
            action='store_true',
            help='Analyze the calibration ground truth and save results to '
            'pickle files'
        )

        # Analyze the ground truth for comparison
        parser.add_argument(
            '-G',
            '--gt-comparison',
            action='store_true',
            help='Analyze the comparison ground truth and save results to '
            'pickle files'
        )

        return parser

    def format_args(self, args):
        """ Modifies args object to format options correctly. """

        if args.all:
            args.continents = ['*']
            args.countries = True
            args.autonomous_systems = True

        if args.continents:
            args.continents = ['SA', 'AF', 'OC', 'EU', 'NA', 'AS'] if (
                args.all or '*' in args.continents
            ) else set(args.continents)

        args.duration = (
            'mon' if (args.duration == 'month') else args.duration[0]
        )

        return args

    def get_section_info(self, config: configparser.SectionProxy) -> Iterator:
        """
        Returns the info that's stored inside every section of
        'gt-validation.conf'.

        The info is stored in different forms depending of the type of data.
        json.loads is used to read lists and booleans. Most values are default
        values, and are loaded from the [DEFAULT] section.

        Parameters
        ----------
        config: configparser.SectionProxy
            Contains all of the data for one section. The parameters from the
            [DEFAULT] section that are not overridden are inherited from it.

        Returns
        -------
        List
            A list containing all of the information.
        """

        return map(
            json.loads,
            [config['signals'], config['test_time'], config['outages'],
             config['n'], config['difference_order'], config['arima_order'],
             config['save'], config['plot'], config['thresholds']
             ]
        )

    def get_comp_info(self, config: configparser.SectionProxy) -> Iterator:
        return map(
            json.loads,
            [config['signals'], config['test_time'], config['bgp'],
             config['ap'], config['dn']
             ]
        )

    def analyze_sections(self, config: configparser.ConfigParser,
                         validation=True) -> None:
        """
        For each section: get data, perform ARIMA, report outages.

        Each section is describing a ground truth event. The first step is to
        get the data, then to analyze the different signals. The outage reports
        (false positives and true positives alike) are then going to be stored
        inside a database.

        Parameters
        ----------
        config: configparser.ConfigParser
            Contains the different sections and their respective attributes.
        """

        def get_date_prefix(gt_file):
            if 'afnog' in gt_file:
                prefix = '2009-05-03 '
            elif 'cnci' in gt_file:
                prefix = '2009-08-17 '
            elif 'czech' in gt_file:
                prefix = '2009-02-16 '
            elif 'rd' in gt_file:
                prefix = '2010-08-27 '
            else:
                prefix = '2011-11-07 '

            return prefix

        # Get general info from config file
        (signals, test_time, outages, n, difference_order, arima_order, save,
         plot, thresholds) = (
            self.get_section_info(config[config.sections()[0]])
        )

        # Unique id to identify plots
        index = 0

        if validation:
            logging.info('Started to analyze the CAIDA files.')

            for gt_file in os.listdir('ground_truth_caida/'):
                logging.info('Analyzing file {}'.format(gt_file))

                # Load events into dataframe and keep outages
                gt = pd.read_csv('ground_truth_caida/' + gt_file)
                gt = gt[gt['GT'] == 1] # Remove if you want every single file

                # Date prefix
                prefix = get_date_prefix(gt_file)

                # Analyze each event
                for index, row in gt.iterrows():
                    signals = [str(row['ASN'])]
                    test_time = [
                        prefix + row['Time'].split('-')[0],
                        prefix + row['Time'].split('-')[1]
                    ]
                    outages = {signals[0]: test_time}

                    # Get the corresponding dataframes
                    df = Ioda().get_dataframe(signals, test_time)

                    arima = Arima(test_time, n, difference_order, arima_order,
                                save, plot, index, thresholds)

                    arima.static_analysis(df[signals[0]], outages)
                    index += 1

                    logging.info('Added an event for AS {}'.format(signals[0]))

            logging.info('Finished analyzing the CAIDA files.')

        logging.info('Starting to analyze sections')

        for section in config.sections():
            logging.info('Analyzing section "{}"'.format(section))

            (signals, test_time, outages, n, difference_order, arima_order,
             save, plot, thresholds) = self.get_section_info(config[section])

            # Get the corresponding dataframes
            df = Ioda().get_dataframe(signals, test_time)

            for signal in [s.replace('.', '_') for s in signals]:
                # Creating an instance with the correct parameters
                arima = Arima(test_time, n, difference_order, arima_order,
                              save, plot, index, thresholds)

                arima.static_analysis(df[signal], outages)
                index += 1

            logging.info('Finished analyzing section "{}"'.format(section))

        logging.info('Finished analyzing all sections.')

    def classify(self, pkl, time_bin, queue, threshold_min, threshold_max):
        df = pd.read_pickle(pkl)

        # Determine offset in minutes (e.g. outage starts at 10:20 instead of
        # at 10:00)
        minute_offset = 0

        # Dirty hack
        if str(df.index[0]) == '2009-08-10 18:00:00':
            # CNCI
            outages = {'x': [['2009-08-17 18:00:00', '2009-08-17 18:40:00']]}
        elif str(df.index[0]) == '2011-10-31 19:15:00':
            # Jun
            outages = {'x': [['2011-11-07 14:00:00', '2011-11-07 15:00:00']]}
        elif str(df.index[0]) == '2009-02-09 16:20:00':
            # Czech ISP
            outages = {'x': [['2009-02-16 16:20:00', '2009-02-16 17:20:00']]}
            minute_offset = 20
        elif str(df.index[0]) == '2009-04-26 12:00:00':
            # AFNOG
            outages = {'x': [['2009-05-03 12:00:00', '2009-05-03 13:00:00']]}
        elif str(df.index[0]) == '2010-08-23 07:35:00':
            # RIPE Duke
            outages = {'x': [['2010-08-27 08:30:00', '2010-08-27 09:30:00']]}
            minute_offset = 30
        elif 'AF_EG' in pkl or pkl.split('-')[1] in ['8452', '36992',
                                                     '24863', '24835']:
            outages = {
                'AF.EG': [['2011-01-27 21:00:00', '2011-02-02 12:00:00']]
            }
        elif 'SA_BR' in pkl:
            outages = {
                'SA.BR': [['2018-03-21 18:45:00', '2018-03-22 10:00:00']]
            }
        elif 'AS_SY' in pkl and str(df.index[0]) == '2018-05-19 00:00:00':
            outages = {
                    "AS.SY": [
                        ["2018-05-27 22:00:00", "2018-05-28 06:00:00"],
                        ["2018-05-28 23:00:00", "2018-05-29 06:00:00"],
                        ["2018-05-30 22:00:00", "2018-05-31 06:00:00"],
                        ["2018-06-02 23:00:00", "2018-06-03 06:00:00"],
                        ["2018-06-03 23:00:00", "2018-06-04 06:00:00"],
                        ["2018-06-05 23:00:00", "2018-06-06 06:00:00"],
                        ["2018-06-09 23:00:00", "2018-06-10 06:00:00"],
                        ["2018-06-11 23:00:00", "2018-06-12 06:00:00"]
                    ]
                }
        elif 'AS_SY' in pkl and str(df.index[0]) == '2017-05-22 00:00:00':
            outages = {
                    "AS.SY": [
                        ["2017-05-30", "2017-05-30 06:00:00"],
                        ["2017-06-01", "2017-06-01 06:00:00"],
                        ["2017-06-03 22:00:00", "2017-06-04 06:00:00"],
                        ["2017-06-04 22:00:00", "2017-06-05 06:00:00"],
                        ["2017-06-03 22:00:00", "2017-06-04 06:00:00"],
                        ["2017-06-05 15:00:00", "2017-06-05 20:00:00"],
                        ["2017-06-07 22:00:00", "2017-06-08 06:00:00"],
                        ["2017-06-10 22:00:00", "2017-06-11 06:00:00"],
                        ["2017-06-11 22:00:00", "2017-06-12 06:00:00"],
                        ["2017-06-12 22:00:00", "2017-06-13 06:00:00"]
                    ]
                }
        elif 'AS_AZ' in pkl:
            outages = {
                    "AS.AZ": [
                        ["2018-07-02 12:00:00", "2018-07-03 06:00:00"]
                    ]
                }
        elif 'AF_CD' in pkl:
            outages = {
                    "AF.CD": [
                        ["2017-12-23 15:00:00", "2017-12-26 09:00:00"],
                        ["2017-12-30 15:00:00", "2018-01-02 09:00:00"]
                    ]
                }
        elif 'AF_GM' in pkl:
            outages = {
                    "AF.GM": [
                        ["2016-11-30 17:00:00", "2016-12-04 22:00:00"]
                    ]
                }
        else:
            logging.error('ERROR: new type of event')
            raise ValueError('New type of event. Need to determine outage '
                             'period: {}'.format(pkl))

        df = df.resample(time_bin, base=minute_offset).max()

        res = {}

        for threshold in np.arange(threshold_min, threshold_max, 0.1):
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for signal, outage_dates in outages.items():
                for event in outage_dates:
                    start = pd.to_datetime(event[0])
                    end = pd.to_datetime(event[1])

                    # Finding true positives
                    true_positives += len(
                        df[start:end][df >= threshold]
                    )

                    # Finding false negatives
                    false_negatives += len(
                        df[start:end][df < threshold]
                    )

                    # Finding false positives
                    false_positives += len(
                        df[:start].iloc[:-1][df >= threshold]
                    )
                    false_positives += len(
                        df[end:].iloc[1:][df >= threshold]
                    )

                    # Finding true negatives
                    true_negatives += len(
                        df[:start].iloc[:-1][df < threshold]
                    )
                    true_negatives += len(
                        df[end:].iloc[1:][df < threshold]
                    )

                res['{:.1f}'.format(round(threshold, 1))] = [
                    true_positives,
                    false_positives,
                    true_negatives,
                    false_negatives
                ]

        queue.put((int(pkl.split('/')[1].split('-')[0]), res))

    def custom_dict_merge(self, dict1, dict2):
        only_dict1 = {
            key: dict1[key] for key in set(dict1.keys()) - set(dict2.keys())
        }
        both = {
            k1: [sum(x) for x in zip(v1, v2)]
            for (k1, v1) in dict1.items()
            for (k2, v2) in dict2.items() if k1 == k2
        }
        only_dict2 = {
            key: dict2[key] for key in set(dict2.keys()) - set(dict1.keys())
        }

        return {**only_dict1, **both, **only_dict2}

    def get_info(self, filenames):
        min_t = float('inf')
        max_t = float('-inf')

        for pkl in filenames:
            df = pd.read_pickle(self.gt_dir + pkl)

            min_t = df.min() if df.min() < min_t else min_t
            max_t = df.max() if df.max() > max_t else max_t

        # In pratice, there are all clustered to the top right of the ROC curve
        # and are not useful
        if min_t < -3:
            min_t = -3.0

        return floor(min_t * 10) / 10, ceil(max_t * 10) / 10

    def generate_roc_data(self, classes, time_bin, pickles):
        res = {}

        threshold_min, threshold_max = self.get_info(pickles)

        pool = mp.Pool()
        queue = mp.Manager().Queue()

        for pkl in pickles:
            # Parallelized
            pool.apply_async(
                self.classify,
                args=(self.gt_dir + pkl, time_bin, queue, threshold_min,
                      threshold_max)
            )
            """
            # Non-parallelized
            self.classify(self.gt_dir + pkl, time_bin, queue,
                          threshold_min, threshold_max)
            """

        pool.close()
        pool.join()

        logging.info('Finished classying. Now separating classes.')

        for c in classes:
            res[c] = [{}, {}]
        res['all'] = {}

        for med, events in queue_iterator(queue):
            for c in classes:
                d = 0 if med <= c else 1

                # Add to one of the two dictionaries in each class
                for key, value in events.items():
                    if key not in res[c][d]:
                        res[c][d][key] = [0, 0, 0, 0]

                    res[c][d][key] = [
                        sum(x) for x in zip(res[c][d][key], value)
                    ]

            # Add to the 'all' class
            for key, value in events.items():
                if key not in res['all']:
                    res['all'][key] = [0, 0, 0, 0]

                res['all'][key] = [
                    sum(x) for x in zip(res['all'][key], value)
                ]
        return res

    def generate_roc_curves(self):
        """
        Generates ROC curves to find two parameters: the minimal amount of
        unique IP addresses to make meaningful predictions and the 't'
        threshold.
        """

        def get_index(median, thresholds):
            try:
                res = next(index for index, value in enumerate(thresholds) if
                           value >= median)
            except(StopIteration):
                res = len(thresholds)

            return res

        def get_merged_dicts(res, first_range, second_range):
            dict1 = {}
            dict2 = {}

            dict1, dict2 = res[first_range[0]], res[second_range[0]]

            for i in first_range[1:]:
                dict1 = self.custom_dict_merge(dict1, res[i])

            for i in second_range[1:]:
                dict2 = self.custom_dict_merge(dict2, res[i])

            return dict1, dict2

        time_bins = [x + 'min' for x in ['5']]
        classes = [15, 20]

        res = {}
        if not os.path.isfile(self.results_dir + 'rocs.pkl'):
            logging.info('Creating rocs.pkl')

            pickles = os.listdir('combined_ground_truth')

            for time_bin in time_bins:
                logging.info('Analyzing for time bin ' + time_bin)
                res[time_bin] = self.generate_roc_data(
                    classes, time_bin, pickles
                )
                pickle.dump(
                    res[time_bin],
                    open(self.results_dir + time_bin + '.pkl', 'wb+'),
                    pickle.HIGHEST_PROTOCOL
                )

            logging.info('Finished all pickles. Saved results to '
                         'results/rocs.pkl')

            pickle.dump(
                res,
                open(self.results_dir + 'rocs.pkl', 'wb+'),
                pickle.HIGHEST_PROTOCOL
            )
        else:
            logging.info('rocs.pkl already exists.')
            res = pickle.load(open(self.results_dir + 'rocs.pkl', 'rb'))

        logging.info('Generating ROC curves')
        p = Plot()

        for time_bin in time_bins:
            logging.info('~ Creating roc curves for time bin = ' + time_bin)
            for c in classes:
                logging.info('~~ Creating roc curves for ip > ' + str(c))
                p.plot_roc_curve(
                    [{float(k): v for k, v in res[time_bin][c][0].items()},
                     {float(k): v for k, v in res[time_bin][c][1].items()},
                     {float(k): v for k, v in res[time_bin]['all'].items()},
                     ],
                    'roc-{}-{}.pdf'.format(time_bin, c)
                )

        logging.info('Finished drawing roc curves')

    def live_analysis(self) -> None:
        """
            Analyze the time series contained in self.continents and all
            countries if self.countries is True.
        """

        signals: list = []

        if self.continents:
            logging.info('Analyzing continents: {}'.format(self.continents))
            signals += self.continents

        if self.countries:
            logging.info('Analyzing all countries')
            signals += ['*.*']

        now = pd.Timestamp.utcnow()
        test_time = [now - pd.Timedelta(weeks=1), now]

        df = Ioda().get_dataframe(signals, test_time)

        index = 0

        # Get default values
        config = configparser.ConfigParser()
        config.read('gt-validation.conf')

        (_, _, _, n, difference_order, arima_order, save,
         plot, thresholds) = (
            self.get_section_info(config[config.sections()[0]])
        )

        arima = Arima(test_time, n, difference_order, arima_order,
                      save, plot, index, thresholds)

        arima.live_analysis(df)

    def get_arima_data(self, signal, index):
        for ts in os.listdir(self.results_dir):
            if signal[0].replace('.', '_') not in ts:
                continue

            # treating cases with two entries
            if str(index[-1].timestamp()) not in ts:
                continue

            df = pd.read_pickle(self.results_dir + ts)
            df = df[index][df > 1]

            return df

        logging.error('Did not find the file!! Exiting...')
        sys.exit(1)

    def generate_comparison(self):
        def get_df(index, data):
            df = pd.DataFrame(data=[False,] * len(index), index=index)

            if isinstance(data, list):
                for time_period in data:
                    i = pd.date_range(time_period[0], time_period[1],
                                      freq='5min')
                    try:
                        df.loc[i] = True
                    except(KeyError):
                        import IPython
                        IPython.embed(header='key error generate comp')
            else:
                df.loc[data.index] = True

            return df

        def combine(df):
            res1 = {
                'BGP': len(df.query('BGP & ~AP & ~DN')),
                'AP': len(df.query('AP & ~BGP & ~DN')),
                'DN': len(df.query('DN & ~BGP & ~AP')),
                'BGP + AP': len(df.query('BGP & AP & ~DN')),
                'BGP + DN': len(df.query('BGP & ~AP & DN')),
                'AP + DN': len(df.query('~BGP & AP & DN')),
                'BGP + AP + DN': len(df.query('BGP & AP & DN')),
            }
            res2 = {
                'BGP': len(df.query('BGP & ~AP & ~AR')),
                'AP': len(df.query('~BGP & AP & ~AR')),
                'AR': len(df.query('~BGP & ~AP & AR')),
                'BGP + AP': len(df.query('BGP & AP & ~AR')),
                'BGP + AR': len(df.query('BGP & ~AP & AR')),
                'AP + AR': len(df.query('~BGP & AP & AR')),
                'BGP + AP + AR': len(df.query('BGP & AP & AR')),
            }

            return res1, res2

        config = configparser.ConfigParser()
        config.read('gt-comparison.conf')

        results = {}
        results['DN'] = {
            'BGP': 0, 'AP': 0, 'DN': 0, 'BGP + AP': 0,'BGP + DN': 0,
            'AP + DN': 0, 'BGP + AP + DN': 0,
        }
        results['AR'] = {
            'BGP': 0, 'AP': 0,'AR': 0, 'BGP + AP': 0,
            'BGP + AR': 0, 'AP + AR': 0,
            'BGP + AP + AR': 0,
        }

        for section in config.sections():
            logging.info('Analyzing section "{}"'.format(section))

            (signals, test_time, bgp, ap, dn) = (
                self.get_comp_info(config[section])
            )
            index = pd.date_range(test_time[0], test_time[1], freq='5min')

            # Gather arima data for this period
            ar = self.get_arima_data(signals, index)

            df = pd.DataFrame(index=index)

            df['BGP'] = get_df(index, bgp)
            df['AP'] = get_df(index, ap)
            df['DN'] = get_df(index, dn)
            df['AR'] = get_df(index, ar)

            # Combine everything
            res1, res2 = combine(df)

            # Add globally
            for k in res1:
                results['DN'][k] += res1[k]
            for k in res2:
                results['AR'][k] += res2[k]

            # Add locally
            results[section] = [res1, res2]

            # Analysis of the AR only
            index = df.query('AR & ~BGP & ~AP').index

        with open('comp.pkl', 'wb') as fp:
            pickle.dump(results, fp)

    def main(self) -> None:
        """
        Perform analysis of the ground truth present in 'gt-validation.conf'.

        Load data from every network event described in 'gt-validation.conf' and
        try to detect outages in it. Outages that detected inside a period
        marked as an outage are considered true positives and the others are
        false positiives.
        """

        if self.validation:
            logging.info('Perform validation')

            if self.gt_validation:
                # Extract config from gt-validation.conf
                config = configparser.ConfigParser()
                config.read('gt-validation.conf')

                self.analyze_sections(config)

            if self.gt_comparison:
                # Extract config from gt-validation.conf
                config = configparser.ConfigParser()
                config.read('gt-comparison.conf')

                self.analyze_sections(config, validation=False)

            self.generate_comparison()
            self.generate_roc_curves()
        else:
            logging.info('Performing live analysis')

            self.live_analysis()

        if self.mail:
            send_mail('Chocolatine - end')

        logging.info('End')

###############################################################################


if __name__ == '__main__':
    chocolatine = Chocolatine()

    chocolatine.main()
