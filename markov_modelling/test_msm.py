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
        input_directory = constants.output_data_segment_keyword_matrix
        feature_map_file = constants.feature_map
        # Read the segment index term matrix
        data = np.load(input_directory + constants.output_segment_keyword_matrix_data_file,
                       allow_pickle=True)

        # Read the column index (index terms) of the matrix above
        cls.features_df = pd.read_csv(input_directory +
                                  constants.output_segment_keyword_matrix_feature_index)

        # Read the metadata partitions
        with open(input_directory + "metadata_partitions.json") as read_file:
            metadata_partitions = json.load(read_file)

        indices = metadata_partitions['complete']
        input_data_set = np.take(data, indices)

        cls.segment_index = pd.read_csv(input_directory+'document_index.csv').iloc[indices]
        

        # Estimate fuzzy trajectories
        cls.trajs = mu.estimate_fuzzy_trajectories(input_data_set, n_realizations=1)

        
        # load raw data for comparison
        rawdat = pd.read_csv(constants.input_data+'all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv')
        
        cls.feature_map = pd.read_csv(constants.input_data+feature_map_file)
        ct_id = {t:i for i, t in enumerate(cls.feature_map.CoverTerm.unique())}
        ct_id_list = [ct_id[term] for term in cls.feature_map.CoverTerm]

        cls.feature_map['CoverTermID'] = ct_id_list
        cls.features_df = pd.merge(cls.features_df, cls.feature_map[['CoverTermID', 'CoverTerm']],
                                   left_on='KeywordLabel', right_on='CoverTerm', how='left')

        assert(all(cls.features_df['index'] == cls.features_df['CoverTermID']))
        
        # only keep keepwords that are in main data.
        cls.cleaned_dat = rawdat[rawdat.KeywordLabel.isin(cls.feature_map.KeywordLabel.unique())]

        cls.cleaned_dat = pd.merge(cls.cleaned_dat, cls.feature_map, on='KeywordLabel', how='left')


    def test_lengths(self):
        self.assertEqual(len(self.trajs), len(self.segment_index))


    def test_state_assignment(self):
        check_trajs = {}
        for intcode in tqdm(np.unique(self.cleaned_dat['IntCode'])):
            check_trajs[intcode] = []
            last_segnum = -1
            for segnum, kwid in zip(self.cleaned_dat[self.cleaned_dat.IntCode == intcode]['SegmentNumber'],
                                    self.cleaned_dat[self.cleaned_dat.IntCode == intcode]['CoverTermID']):
                if segnum == last_segnum:
                    check_trajs[intcode][-1].append(kwid)
                else:
                    check_trajs[intcode].append([kwid])

                last_segnum = segnum


        _map = self.features_df.drop_duplicates()
        relabled_trajs = []
        for dtraj in self.trajs:
            _dtraj = pd.DataFrame(dtraj[dtraj >= 0], columns=['Unnamed: 0'])
            _cover_term_id_traj = pd.merge(_dtraj, _map,
                                           on='Unnamed: 0', how='left')['CoverTermID']
            relabled_trajs.append(_cover_term_id_traj.to_numpy())


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

    def test_has_emptysegments(self):
        self.assertTrue(-1 in np.unique(np.concatenate(self.trajs)), msg='Trajectories have no empty states.')

    def test_emptysegments(self):
        for traj, intcode in zip(self.trajs, self.segment_index['IntCode']):
            is_empty_traj = traj == -1

            segnums = self.cleaned_dat[self.cleaned_dat.IntCode == intcode]['SegmentNumber']
            segnums = segnums - segnums.min()
            is_empty_raw = np.ones(segnums.max()+1, dtype=bool)
            is_empty_raw[np.unique(segnums)] = False

            np.testing.assert_array_equal(is_empty_raw, is_empty_traj,
                                          err_msg=f'Empty state positions don`t match for IntCode {intcode}')

    def test_special_cases(self):
        #TODO: please complete test
        # one can use self.assert() as done here more than once to test the things you want to test,
        # I'm guessing the first line should be self.assertEqual(cover_term_numer, 79)

        # Example starts

        cover_term_number=self.trajs[105][1]
        # -> 79

        cover_term_label = self.features_df.iloc()[cover_term_number]['KeywordLabel']
        # -> perpetrators

        int_code = self.segment_index.iloc()[105]['IntCode']

        # -> 576

        # Test whether the interview code 576 in the raw data speaks about perpetrators

        # Get all original keywords belonging to the cover term perpetrators

        perpetrators = self.feature_map[self.feature_map.CoverTerm=="perpetrators"].KeywordLabel.to_list()

        # Get all original keywords in interview 576
        original_keywords = self.cleaned_dat[self.cleaned_dat['IntCode']==576].KeywordLabel.to_list()

        # Is there any overlapping between perpetrators and the original_keywords

        overlapping_elements = [element for element in original_keywords if element in perpetrators]

        self.assertGreater(len(overlapping_elements), 0)

