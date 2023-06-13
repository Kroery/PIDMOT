from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer, MotTrainer_sacaidm


train_factory = {
  'mot': MotTrainer_sacaidm,
}
