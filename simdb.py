#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite3 database interface for storeing simulated light curves.
"""

import numpy as np
import os
import sqlite3
import sys

#==============================================================================
# CLASSES
#==============================================================================

class SQLiteConnection:
    """Wrapper class for SQLite connection that allows the use of the
    python with-statement."""

    def __init__(self, db_file):
        """Wrapper class for SQLite connection that allows the use of the
        python with-statement.

        Parameters
        ----------
        db_file : str
            File name of the database file.

        Returns
        -------
        None
        """

        self.db_file = db_file

    def __enter__(self):

        self.connection = sqlite3.connect(self.db_file)

        return self.connection

    def __exit__(self, type, value, traceback):
        self.connection.close()

#==============================================================================

class DBConnectorSQLite:
    """SQLite database connector."""

    #--------------------------------------------------------------------------
    def __init__(self, db_file):
        """SQLite database connector.

        Parameters
        ----------
        db_file : str
            File name of the database file.

        Returns
        -------
        None
        """

        self.db_file = db_file

    #--------------------------------------------------------------------------
    def _query(self, connection, query, commit=False):
        """Query database.

        Parameters
        ----------
        connection : sqlite3.Connection
            Database connection.
        query : str
            SQL query.
        commit : bool, optional
            If True, changes to the database are commited. The default is
            False.

        Returns
        -------
        result : sqlite3.Cursor
            Includes the query results.
        """

        cursor = connection.cursor()
        result = cursor.execute(query)

        if commit:
            connection.commit()

        return result

    #--------------------------------------------------------------------------
    def _get_sim_type_id(self, sim_type):
        """Get ID of simulation type.

        Parameters
        ----------
        sim_type : str
            Select the simulation type. Must be 'tk' or 'emp'.

        Raises
        ------
        ValueError
            Raised when the requested simulation type does not exist in the
            database, i.e. when `sim_type` is not 'tk' or 'emp'.

        Returns
        -------
        sim_type_id : int
            Database ID of the simulation type.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT sim_type_id
                FROM sim_types
                WHERE sim_type = '{0}';
                """.format(sim_type.lower())
            results = self._query(connection, query, commit=False).fetchone()

        if results is None:
            raise ValueError(
                    f"Simulation type '{sim_type}' does not exist in "
                    "database!")

        sim_type_id = results[0]

        return sim_type_id

    #--------------------------------------------------------------------------
    def _get_psd_type_id(self, psd_type):
        """Get ID of PSD type.

        Parameters
        ----------
        psd_type : str
            Select the PSD type. Must be 'powerlaw', 'brokenpowerlaw', or
            'kneemodel'.

        Raises
        ------
        ValueError
            Raised when the requested PSD type does not exist in the database,
            i.e. when `psd_type` is not 'powerlaw', 'brokenpowerlaw', or
            'kneemodel'.

        Returns
        -------
        psd_type_id : int
            Database ID of the PSD type.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT psd_type_id
                FROM psd_types
                WHERE psd_type = '{0}';
                """.format(psd_type.lower())
            results = self._query(connection, query, commit=False).fetchone()

            if results is None:
                raise ValueError(
                        f"PSD type '{psd_type}' does not exist in database!")

            psd_type_id = results[0]

        return psd_type_id

    #--------------------------------------------------------------------------
    def _add_sim_metadata(self, sim_meta):
        """Add simulated light curve meta data.

        Parameters
        ----------
        sim_meta : dict
            Dictionary of simulation metadata.

        Returns
        -------
        sim_id : int
            Returns the database simulation ID under which these metadata were
            saved.
        """

        sim_id = self.number_of_sim() + 1
        sim_type_id = self._get_sim_type_id(sim_meta['sim_type'])
        psd_type_id = self._get_psd_type_id(sim_meta['psd_type'])

        with SQLiteConnection(self.db_file) as connection:
            cursor = connection.cursor()
            query = """\
                INSERT INTO simulations(sim_type_id, psd_type_id)
                VALUES
                ({0}, {1})
                """.format(sim_type_id, psd_type_id)
            cursor.execute(query)
            connection.commit()

            # extract PDF check metadata:
            if 'pdf_reject' in sim_meta.keys():
                pdf_reject = int(sim_meta['pdf_reject'])
                pdf_pvalue = sim_meta['pdf_pvalue']
            else:
                pdf_reject = -1
                pdf_pvalue = "'NULL'"

            # add TK metadata:
            if sim_meta['sim_type'].lower() == 'tk':
                query = """\
                    INSERT INTO tk_meta(sim_id, pdf_pvalue, pdf_reject)
                    VALUES
                    ({0}, {1}, {2})
                    """.format(sim_id, pdf_pvalue, pdf_reject)
                cursor.execute(query)
                connection.commit()

            # add EMP metadata:
            elif sim_meta['sim_type'].lower() == 'emp':
                query = """\
                    INSERT INTO emp_meta(
                        sim_id, pdf_n_iter, pdf_converged, pdf_pvalue,
                        pdf_reject)
                    VALUES
                    ({0}, {1}, {2}, {3}, {4})
                    """.format(
                            sim_id, sim_meta['pdf_n_iter'],
                            sim_meta['pdf_converged'], pdf_pvalue, pdf_reject)
                cursor.execute(query)
                connection.commit()

        return sim_id

    #--------------------------------------------------------------------------
    def _add_sim_data(self, sim_id, data):
        """Add simulated light curve data.

        Parameters
        ----------
        sim_id : int
            Database simulation ID.
        data : structured np.ndarray
            Simulated light curve data. Expected structured array column names:
            'time', 'flux', 'flux_unc'.

        Returns
        -------
        None
        """

        n = len(data['time'])
        sim_id = [sim_id] * n
        data = zip(sim_id, data['time'], data['flux'], data['flux_unc'])
        # TODO: will crash when no flux_err exists

        with SQLiteConnection(self.db_file) as connection:
            cursor = connection.cursor()
            query = """\
                INSERT INTO lightcurves(sim_id, mjd, flux, flux_err)
                VALUES
                (?, ?, ?, ?)
                """
            cursor.executemany(query, data)
            connection.commit()

        return None

    #--------------------------------------------------------------------------
    def create_db(self, db_file=None):
        """Create the simulation database.

        Parameters
        ----------
        db_file : str
            Database filename. If none is provided the filename provided with
            the class instanciation is used. The default is None.

        Returns
        -------
        None

        Notes
        -----
        If a file with the provided name already exists, the user is asked,
        whether or not it should be overwritten.
        """

        if db_file is None:
            db_file = self.db_file

        # check if file exists:
        if os.path.isfile(db_file):
            answer = input(
                'Database file exists. Overwrite (y) or cancel (enter)?')

            if answer.lower() in ['y', 'yes', 'make it so!']:
                os.system(f'rm {db_file}')
                print('Overwriting current data base..')

            else:
                print('Keeping current data base.')
                return None

        # create database file:
        connection = sqlite3.connect(db_file)
        cursor = connection.cursor()

        # create lightcurves table:
        query = '''\
            CREATE TABLE IF NOT EXISTS lightcurves(
                data_id integer NOT NULL PRIMARY KEY,
                sim_id integer NOT NULL,
                mjd float,
                flux float,
                flux_err float,
                FOREIGN KEY (sim_id) REFERENCES simulations(sim_id)
                );
            '''
        cursor.execute(query)

        # create simulations table:
        query = '''\
            CREATE TABLE IF NOT EXISTS simulations(
                sim_id integer NOT NULL PRIMARY KEY,
                sim_type_id integer,
                psd_type_id integer,
                FOREIGN KEY (sim_type_id) REFERENCES sim_types(sim_type_id),
                FOREIGN KEY (psd_type_id) REFERENCES psd_types(psd_type_id)
                );
            '''
        cursor.execute(query)

        # create sim_types table:
        query = '''\
            CREATE TABLE IF NOT EXISTS sim_types(
                sim_type_id integer NOT NULL PRIMARY KEY,
                sim_type varchar(3),
                sim_type_ref varchar(20)
                );
            '''
        cursor.execute(query)

        # create psd_types table:
        query = '''\
            CREATE TABLE IF NOT EXISTS psd_types(
                psd_type_id integer NOT NULL PRIMARY KEY,
                psd_type varchar(20)
                );
            '''
        cursor.execute(query)

        # create tk_meta table:
        query = '''\
            CREATE TABLE IF NOT EXISTS tk_meta(
                meta_id integer NOT NULL PRIMARY KEY,
                sim_id integer,
                pdf_pvalue float,
                pdf_reject integer,
                FOREIGN KEY (sim_id) REFERENCES simulations(sim_id)
                );
            '''
        cursor.execute(query)

        # create emp_meta table:
        query = '''\
            CREATE TABLE IF NOT EXISTS emp_meta(
                meta_id integer NOT NULL PRIMARY KEY,
                sim_id integer,
                pdf_n_iter integer,
                pdf_converged boolean,
                pdf_pvalue float,
                pdf_reject integer,
                FOREIGN KEY (sim_id) REFERENCES simulations(sim_id)
                );
            '''
        cursor.execute(query)

        connection.commit()

        # insert sim_types to database:
        query = """INSERT INTO sim_types (sim_type, sim_type_ref)
            VALUES
            ('tk', 'Timmer+1995'),
            ('emp', 'Emmanoulopoulos+2013');
            """
        cursor.execute(query)

        # insert psd_types to database:
        query = """INSERT INTO psd_types (psd_type)
            VALUES
            ('powerlaw'),
            ('brokenpowerlaw'),
            ('kneemodel');
            """
        cursor.execute(query)

        connection.commit()
        print(f'Database created: {db_file}')

    #--------------------------------------------------------------------------
    def add_sim(self, lightcurves):
        """Add simulated light curves to database.

        Parameters
        ----------
        lightcurves : list of lcsim.ArtificialLightCurve instances
            The light curves to be added to the database. Light curve data and
            metadata is automatically extracted from the
            ArtificialLightCurve instances.

        Returns
        -------
        None
        """

        if type(lightcurves) is not list:
            lightcurves = [lightcurves]

        # iterate through light curves:
        for lightcurve in lightcurves:
            sim_id = self._add_sim_metadata(lightcurve.sim_meta)
            self._add_sim_data(sim_id, lightcurve.data())

        return None

    #--------------------------------------------------------------------------
    def number_of_sim(self):
        """Get number of stored simulations.

        Returns
        -------
        n_sim : int
            Number of stored simulations.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT MAX(sim_id)
                FROM simulations
                """
            results = self._query(connection, query, commit=False).fetchone()
            n_sim = results[0]

            if n_sim is None:
                n_sim = 0

        return n_sim

    #--------------------------------------------------------------------------
    def iter_sim(self, cache=100):
        """Iteratively return simulated light curve data.

        Parameters
        ----------
        cache : int, optional
            Simulations are loaded from the database in sets to speed up
            reading from the disk. This number sets how many simulations are
            read as a set. The default is 100.

        Yields
        ------
        numpy.ndarray
            Array with the simulated light curve. The three columns contain
            time, flux, and flux uncertainty.
        """

        # TODO: This is a simplistic interface to access the light curves.
        # Improvements: (1) yield different data structure, e.g. structured
        # array or ArtificialLightCurve instance. Include metadata in the
        # latter? (2) Allow quering specific types of stored simulations or
        # under specific conditions.

        # open database:
        with SQLiteConnection(self.db_file) as connection:

            # iterate through stored simulations in sets:
            n = self.number_of_sim() // cache
            for i in range(n):
                i0 = i * cache
                i1 = (i + 1) * cache
                query = '''\
                    SELECT sim_id, mjd, flux, flux_err
                    FROM lightcurves
                    WHERE sim_id>{0} AND sim_id<={1};
                    '''.format(i0, i1)
                result = self._query(connection, query).fetchall()
                result = np.array(result).transpose()

                # iterate through individual simulations:
                for j in range(i0, i1):
                    sel = np.nonzero(np.isclose(j+1, result[0]))[0]
                    j0 = sel[0]
                    j1 = sel[-1]
                    sim = result[1:,j0:j1]

                    yield sim



