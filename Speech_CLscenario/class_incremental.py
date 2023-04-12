#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:42:31 2022

@author: umbertocappellazzo
"""

import warnings
from typing import Callable, List, Union

import numpy as np

from Speech_CLscenario.base_dataset import _ContinuumDataset
from Speech_CLscenario.base_scenario import _BaseScenario


class ClassIncremental(_BaseScenario):
    """Continual Loader, generating datasets for the consecutive tasks.
    Scenario: Each new tasks bring new classes only
    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario number of tasks.
    :param increment: Either number of classes per task (e.g. increment=2),
                    or a list specifying for every task the amount of new classes
                     (e.g. increment=[5,1,1,1,1]).
    :param initial_increment: A different task size applied only for the first task.
                              Deactivated if `increment` is a list.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param class_order: An optional custom class order, used for NC.
                        e.g. [0,1,2,3,4,5,6,7,8,9] or [5,2,4,1,8,6,7,9,0,3]
    :param cplitting_crit: The criterion by which to split the dataset. It can be:
                            -'scenario', 'action', or 'None' (general case for the 
                            datasets that only have a single type of classes (e.g., FSC
                            provides the classes for only the intents).

    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int = 0,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        class_order: Union[List[int], None]=None,
        # splitting_crit = 'scenario',
        splitting_crit = None,
    ) -> None:

        self.cl_dataset = cl_dataset
        self.increment = increment
        self.initial_increment = initial_increment
        self.class_order = class_order
        self.splitting_crit = splitting_crit

        self._nb_tasks = self._setup(nb_tasks)
        super().__init__(cl_dataset=cl_dataset, nb_tasks=self._nb_tasks, transformations=transformations,splitting_crit=self.splitting_crit)

    def _setup(self, nb_tasks: int) -> int:
        x, y, _, transcripts = self.cl_dataset.get_data()

        if self.splitting_crit is None:
            unique_classes = np.unique(y)
        else:
            if self.splitting_crit == 'scenario':
                # unique_classes = np.unique(y[:, 0])
                unique_classes = np.unique(y[:])
            else:
                unique_classes = np.unique(y[:, 1])
            
        
        if self.class_order is None:
            if self.cl_dataset.class_order is not None:
                self.class_order = self.cl_dataset.class_order
            elif self.splitting_crit is None:
                self.class_order = np.arange(np.max(y) + 1)
            elif self.splitting_crit == 'scenario':
                # self.class_order = np.arange(np.max(y[:, 0]) + 1)
                self.class_order = np.arange(np.max(y[:]) + 1)
            else: 
                self.class_order = np.arange(np.max(y[:, 1]) + 1)
                                              
        
        self.class_order = list(self.class_order)


        if len(np.unique(self.class_order)) != len(self.class_order):
            raise ValueError(f"Invalid class order, duplicates found: {self.class_order}.")

        if len(self.class_order) > len(unique_classes):
            missing_classes = set(self.class_order) - set(unique_classes)
            warnings.warn(f"There are some missing classes: {missing_classes}!")

        # Map the classes ids to their new values according to class order.
        # Aka if the user wants that the first 2 classes are 2, 7; then all
        # samples with classes 2 & 7 are resp. labeled as 0 & 1.
        self.class_order = np.array(self.class_order)
        if self.splitting_crit is None:
            new_y = self.class_order.argsort()[y.astype(np.int64)]
            
        else:
            if self.splitting_crit == 'scenario':
                new_y = np.copy(y)
                # new_y[:, 0] = self.class_order.argsort()[y[:, 0].astype(np.int64)]
                new_y[:] = self.class_order.argsort()[y[:, 0].astype(np.int64)]
            else:
                new_y = np.copy(y)
                new_y[:, 1] = self.class_order.argsort()[y[:, 1].astype(np.int64)]
                

        # Increments setup
        if nb_tasks <= 0:
            # The number of tasks is left unspecified, thus it will be determined
            # by the specified increments.
            self.increments = self._define_increments(
                self.increment, self.initial_increment, unique_classes
            )
        else:
            # A fixed number of tasks is required, thus the all increments will
            # be equal among tasks.
            if self.increment > 0:
                warnings.warn(
                    f"When both `nb_tasks` (given value = {nb_tasks}) and "
                    f"`increment` (given value = {self.increment} are both set, "
                    "we only consider the number of tasks. The `increment` "
                    "argument is ignored."
                )
            increment = len(unique_classes) / nb_tasks
            if not increment.is_integer():
                raise Exception(
                    f"Invalid number of tasks ({nb_tasks}) for {len(unique_classes)} classes."
                )
            self.increments = [int(increment) for _ in range(nb_tasks)]

        # compute task label
        task_ids = self._set_task_labels(new_y)

        # Dataset with task label
        self.dataset = (x, new_y, task_ids, transcripts)  # (data, class label, task label)
        
        return len(np.unique(task_ids))

    def _set_task_labels(self, y: np.ndarray) -> np.ndarray:
        """For each data point, defines a task associated with the data.
        :param y: label tensor
        :param increments: increments contains information about classes per tasks
        :return: tensor of task label
        """
        t = np.zeros(len(y))

        for task_index, _ in enumerate(self.increments):
            max_class = sum(self.increments[:task_index + 1])
            min_class = sum(self.increments[:task_index])  # 0 when task_index == 0.
            
            if self.splitting_crit is None:
                indexes = np.where(np.logical_and(y >= min_class, y < max_class))[0]
            else:
                if self.splitting_crit == 'scenario':
                    indexes = np.where(np.logical_and(y[:, 0] >= min_class, y[:, 0] < max_class))[0]
                else:
                    indexes = np.where(np.logical_and(y[:, 1] >= min_class, y[:, 1] < max_class))[0]
            
           
            t[indexes] = task_index

        return t

    def _define_increments(
        self, increment: Union[List[int], int], initial_increment: int, unique_classes: List[int]
    ) -> List[int]:

        if isinstance(increment, list):
            # Check if the total number of classes is compatible
            # with increment list and self.nb_classes
            if not sum(increment) == len(unique_classes):
                raise Exception(
                    f"The increment list is not compatible with the number of classes ({len(unique_classes)})"
                )

            increments = increment
        elif isinstance(increment, int) and increment > 0:
            increments = []
            if initial_increment:
                increments.append(initial_increment)

            nb_tasks = (len(unique_classes) - initial_increment) / increment
            if not nb_tasks.is_integer():
                raise Exception(
                    "The tasks won't have an equal number of classes"
                    f" with {len(self.class_order)} and increment {increment}"
                )
            increments.extend([increment for _ in range(int(nb_tasks))])
        else:
            raise TypeError(f"Invalid increment={increment}, it must be an int > 0.")

        return increments

    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.
        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self.class_order[targets]

    def _select_data_by_classes(self, min_class_id: int, max_class_id: int):
        """Selects a subset of the whole data for a given set of classes.
        :param min_class_id: The minimum class id.
        :param max_class_id: The maximum class id.
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        x_, y_, _, transcripts_  = self.dataset
        if self.splitting_crit is None:
            indexes = np.where(np.logical_and(y_ >= min_class_id, y_ < max_class_id))[0]
        else:
            if self.splitting_crit == 'scenario':
                indexes = np.where(np.logical_and(y[:, 0] >= min_class, y[:, 0] < max_class))[0]
            else:
                indexes = np.where(np.logical_and(y[:, 1] >= min_class, y[:, 1] < max_class))[0]
            

        selected_x = x_[indexes]
        selected_y = y_[indexes]
        selected_transcripts = transcripts_[indexes]

        return selected_x, selected_y, selected_transcripts