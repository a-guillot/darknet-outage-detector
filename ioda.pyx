#!/usr/bin/python3
# cython: language_level=3

###############################################################################
# Imports

# Data Formatting
import pandas as pd

# Webpage Requests
import requests

# JSON Parsing
import json

# Type hinting
from typing import Iterable, List, Tuple, Optional

# Threading web requests
from threading import Thread

# Logging messages
import logging

# Sleeping when page requests fail
from time import sleep

###############################################################################
# Classes


class ThreadWithReturnValue(Thread):
    """
    Variation of a thread that returns a value.

    Used to override join and return a value that has been computed by a
    Thread.
    """

    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


class Ioda():
    """
    Fetches data from CAIDA IODA and returns it inside a pandas dataframe.

    Different time series are fetched from the ioda.caida.org website and are
    fed into a pandas dataframe. The index is a DatetimeIndex, and each column
    contains one time series. The get_dataframe() function is the only one that
    should be called from outside.
    """

    def __init__(self):
        self.ioda_url = str(
            'https://ioda.caida.org/data/ts/json?from=START&until=END&expressi'
            'on=[{"type":"path","path":"darknet.ucsd-nt.non-erratic.overall.un'
            'iq_src_ip"},DATASETS]&annotate=true'
        )
        self.geoloc_url = str(
            '{"type":"path","path":"darknet.ucsd-nt.non-erratic.geo.netacuity.'
            'DATASET.uniq_src_ip"}'
        )
        self.routing_url = str(
            '{"type":"path","path":"darknet.ucsd-nt.non-erratic.routing.asn.DA'
            'TASET.uniq_src_ip"}'
        )

        self.pkl_dir = 'pickles/'

        # Setup logging parameters
        logging.basicConfig(filename='logs/logs.log', level=logging.INFO,
                            format='%(asctime)s %(processName)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def datasets_iterator(self, signals: List[str], limit=50) -> Iterable[str]:
        """
        Yields multiple signals containing 'limit' datasets each.

        Each yielded item will contain 'limit' signals to avoid having a huge
        url. The main url is 'ioda_url'.

        Parameters
        ----------
        signals: List[str]
            A list of strings that describe a signal, e.g. 'EU'.
        limit: int
            The number of signals per url, i.e. the number of signals that are
            going to be requested by every get.

        Returns
        -------
        Iterable[str]
            String iterator that will contain the dataset part of the url.
        """

        for i in range(0, len(signals), 50):
            datasets = ''

            for index in range(i, (i + 50)):
                if index >= len(signals):
                    break

                # Determine if signal[index] = continent, country, or AS
                try:
                    # see if converting to int raises ValueError
                    int(signals[index])

                    # AS
                    datasets += self.routing_url.replace(
                        'DATASET',
                        signals[index]
                    )
                except(ValueError):
                    # Continent or country
                    datasets += self.geoloc_url.replace(
                        'DATASET',
                        signals[index]
                    )

                datasets += ','

            # Remove the trailing comma
            datasets = datasets[:-1]

            yield datasets

    def record(self, url, duration: List[pd.Timestamp], is_country, keep=''
               ) -> pd.DataFrame:
        """
        Returns a pandas dataframe that contains data for all 'duration' with a
        granularity of 300 seconds.

        Uses 'duration_iterator' to do enough GETs to fetch the entirety of the
        data at the correct time granularity. Threads everything to go faster.

        Parameters
        ----------
        url: str
            Contains the formatted IODA url with two tokens that need to be
            replaced: START and END.
        duration: List[pd.Timestamp]
            A list that contains two timestamps (start, end).
        keep: str
            'keep' determines if the dataframe is saved inside a pickle file or
            not. Setting 'keep' to a non-empty string will result in the
            dataframe being saved under that name.

        Returns
        -------
        pd.DataFrame
            A dataframe containing as many columns as there are datasets in
            'url', for the entirety of 'duration'.
        """

        complete_df = pd.DataFrame()

        threads = []

        offset = 2 if is_country else 7

        # For every time period in the duration
        for start, end in self.duration_iterator(duration, offset):
            formatted_url = (
                url.replace('START', str(start)).replace('END', str(end))
            )

            thread = ThreadWithReturnValue(
                target=self.get_series_from_page,
                args=(formatted_url.replace('"', '%22'),)
            )

            thread.start()
            threads.append(thread)

            """  # non-parallelized code for debug purposes
            self.get_series_from_page(formatted_url.replace('"', '%22'))
            """

        for thread in threads:
            res = thread.join()
            complete_df = pd.concat([complete_df, res])

        complete_df.sort_index(inplace=True)
        complete_df = complete_df[:duration[1]]

        if keep:
            complete_df.to_pickle(self.pkl_dir + '{}.pkl'.format(keep))

        return complete_df

    def duration_iterator(self, duration: List[pd.Timestamp], offset
                          ) -> Iterable[Tuple[int, int]]:
        """
        Yields multiple (start, end) tuples for the entirety of 'duration'.

        Duration are a tuple of two timestamps. This function will yield as
        many time intervals as it is required to get the entirety of 'duration'
        with a time granularity of 300 seconds.

        Parameters
        ----------
        duration: List[pd.Timestamp]
            A tuple that contains two timestamps (start, end).

        Returns
        -------
        Iterable[Tuple[int, int]]
            Iterator that contains two dates: start and end
        """

        start = duration[0]
        end = duration[1]

        while start < end:
            upper = start + pd.DateOffset(days=offset)

            yield int(start.timestamp()), int(upper.timestamp())
            start += pd.DateOffset(days=offset)

    def get_series_from_page(self, url) -> pd.DataFrame:
        """
        Returns a dataframe with every dataset inside url.

        Creates and returns a dataframe with a DatetimeIndex and one column for
        each dataset.

        Parameters
        ----------
        url: str
            String that contains the fully formatted url.

        Returns
        -------
        pd.DataFrame
            A dataframe that contains every column referenced in 'url' for a
            duration of a week.
        """

        # Get page
        session = requests.session()

        while True:
            page = session.get(url)

            # Get the different series
            js = json.loads(page.text)

            if page.status_code == 500 or js['error'] is not None:
                logging.info('The fetched web page contained an error! '
                             'Retrying...')
                sleep(1)
            else:
                break

        js = js['data']['series']

        series = iter(js.keys())

        # Use the first series to create the datetime index
        series_name = next(series)
        attributes = js[series_name]

        start = attributes['from']
        step = attributes['step']

        df = pd.DataFrame(index=pd.DatetimeIndex(pd.Series(
            [start + step * i for i in range(len(attributes['values']))]
        ).apply(pd.to_datetime, unit='s')
        ))

        for series_name, attributes in iter(js.items()):
            if attributes['name']:
                if '#' in attributes['name']:
                    column_name = attributes['name'].split('#')[0][:-3]
                else:
                    column_name = '-'.join(attributes[
                        'name'].split('·')[:2])
            else:
                column_name = '_'.join(series_name.split('.')[5:-1])

            column_name = column_name.replace('·', '-')
            df[column_name] = attributes['values']

        return df

    def adjust_duration(self, duration: List[str]) -> List[pd.Timestamp]:
        """
        Modify 'duration' to contain enough data to properly train model.

        Modifying duration to have enough data to be able to difference without
        losing anything (a week), 1 week of training data, 1 week of validation
        data, the test that is delimited in 'duration', and an additional week
        to look for follow-ups to the outage(s).

        Parameters
        ----------
        duration: List[str]
            Contains two dates that start as start and end of the test set.

        Returns
        -------
        List[pd.Timestamp]
            Contains pandas datetime that have been extended with the
            information written above
        """

        # Convert both strings to pandas timestamps
        duration = list(map(pd.to_datetime, duration))

        duration[0] -= pd.DateOffset(weeks=12)

        return duration

    def get_dataframe(self, signals: List[str], duration:
                      Optional[List[str]] = None) -> pd.DataFrame:
        """
        Gets every signal in 'signals' and stores its data for 'duration' in a
        dataframe.

        Each time series in 'signals' is going to be extracted from CAIDA IODA
        and stored inside a dataframe that will be returned by this function.
        'duration' indicates when the data begins and when it finishes. The
        time granularity is 300 seconds, or 5 minutes.

        Parameters
        ----------
        signals: List[str]
            List of string containing the name of a time series stored on IODA.
            Continents are two letters long (e.g. 'EU'), countries are composed
            of the continent, a '.', and the first two letters of the country
            code (e.g. 'AS.JP' for Japan), and autonomous systems are just
            composed of their as number (e.g. 7922 for comcast).

        duration: Optional[List[str]]
            'duration' is a list of two dates (begin and end). Data is going
            to be added in order to include training data and a padding after
            the end of the given period to look for potential new outages.
            If the argument is not specified, it will be equal to [-1 week,
            now].

        Returns
        -------
        pd.DataFrame
            A dataframe that contains the time as an index and a column for
            each time series.
        """

        df = pd.DataFrame()

        duration = self.adjust_duration(duration)

        for datasets_url in self.datasets_iterator(signals):
            url = self.ioda_url.replace('DATASETS', datasets_url)

            df = pd.concat(
                [df, self.record(url, duration, '*.*' in url)],
                axis=1
            )

        return df
