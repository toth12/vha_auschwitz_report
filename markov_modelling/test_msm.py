#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import constants
from markov_modelling import markov_utils as mu
from tqdm.auto import tqdm
import json
import unittest


class TestDiscreteTrajectories(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the input data
        input_directory = '../' + constants.output_data_segment_keyword_matrix

        # Read the segment index term matrix
        data = np.load(input_directory + constants.output_segment_keyword_matrix_data_file.replace('.txt', '.npy'),
                       allow_pickle=True)

        # Read the column index (index terms) of the matrix above
        cls.features_df = pd.read_csv(input_directory +
                                  constants.output_segment_keyword_matrix_feature_index)

        # Create the row index  of the matrix above
        segment_df = pd.read_csv(input_directory +
                                 constants.output_segment_keyword_matrix_document_index)

        #cls.int_codes = segment_df['IntCode'].to_list()



        # Read the metadata partitions
        with open(input_directory + "metadata_partitions.json") as read_file:
            metadata_partitions = json.load(read_file)

        indices = metadata_partitions['complete']
        input_data_set = np.take(data, indices)

        # TODO: why is "complete" lacking so many interviews?
        cls.segment_index = pd.read_csv('../data/output/segment_keyword_matrix/document_index.csv').iloc[indices]


        # Estimate fuzzy trajectories
        cls.trajs = mu.estimate_fuzzy_trajectories(input_data_set, n_realizations=1)

    def test_lengths(self):
        self.assertEqual(len(self.trajs), len(self.segment_index))


    def test_state_assignment(self):
        rawdat = pd.read_csv('../data/input/all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv')


        # only keep keepwords that are in main data.
        cleaned_dat = rawdat[rawdat.KeywordID.isin(self.features_df.KeywordID.unique())]

        check_trajs = {}
        for intcode in tqdm(np.unique(cleaned_dat['IntCode'])):
            check_trajs[intcode] = []
            last_segnum = -1
            for segnum, kwid in zip(cleaned_dat[cleaned_dat.IntCode == intcode]['SegmentNumber'],
                                    cleaned_dat[cleaned_dat.IntCode == intcode]['KeywordID']):
                if segnum == last_segnum:
                    check_trajs[intcode][-1].append(kwid)
                else:
                    check_trajs[intcode].append([kwid])

                last_segnum = segnum

        relabled_trajs = [np.array([self.features_df[self.features_df['Unnamed: 0'] == d]['KeywordID'].to_numpy() for d in dtraj if d !=-1]).squeeze() for dtraj in self.trajs]

        for traj, intcode in zip(relabled_trajs, self.segment_index['IntCode']):
            if len(traj) == len(check_trajs[intcode]):
                for traj_step, reference in zip(traj, check_trajs[intcode]):
                    self.assertTrue(traj_step in reference, f'{traj_step} not in {reference}')
            elif len(traj) < len(check_trajs[intcode]):
                traj_match = False
                for shift in np.arange(len(check_trajs[intcode]) - len(traj) + 1):
                    step_match = []
                    for traj_step, reference in zip(traj, check_trajs[intcode][shift:]):
                        step_match.append(traj_step in reference)
                    traj_match = any([traj_match, all(step_match)])
                    if traj_match:
                        break

                self.assertTrue(traj_match, f'{traj_step} not in {reference} for IntCode {intcode}')
                # multiple trajectories in this int-code
            else:
                raise RuntimeError(f'Interview with IntCode {intcode} appears longer than in original dataset.')

