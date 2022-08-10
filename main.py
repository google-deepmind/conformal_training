# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main file to run training."""
from absl import flags

from ml_collections import config_flags

from absl import app
from conformal_training.train import train


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', 'config.py', 'Configuration.')


def main(argv):
  """Main method when called from command line."""
  del argv
  config = FLAGS.config
  train(config)


if __name__ == '__main__':
  app.run(main)
