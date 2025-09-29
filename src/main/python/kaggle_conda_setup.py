#import into kaggle notebook
import subprocess
import os
import sys
import json
from pathlib import Path

def run_command(cmd, capture=True, check=False):
  """Helper function to run shell commands and return output"""
  try:
    result = subprocess.run(cmd, shell=True, capture_output=capture,\
      text=True, check=check)
    if capture:
      return result.stdout.strip() if result.stdout else result.stderr.strip()
    return result.returncode == 0
        except Exception as e:
            return str(e)

def setup_venv():
  print(f"Current Python version: {sys.version}")
  print(f"Python executable: {sys.executable}")
  print(f"Python prefix: {sys.prefix}")
  print()

  !pip install -q condacolab
  import condacolab
  condacolab.install()

  print("Current conda environments:")
  print(run_command("conda env list"))
  print()

  print("Current Python symlinks:")
  print(run_command("ls -la /opt/conda/bin/python*"))
  print()

  #first list packages already installed.  there might be kaggle specific
  # ones needed for notebook magic commands, etc
  #installed in /opt/conda/envs/?
  !pip install -q condacolab
  import condacolab
  condacolab.install()
  !conda create --name my_tfx_env python=3.10 -y
  !ls /opt/conda/envs/
  !conda install --name my_tfx_env ipykernel -y
  !source activate my_tfx_env && python --version
  !python -m ipykernel install --prefix "/opt/conda/envs/" --name my_tfx_env --display-name my_tfx_env
  # then refresh the page.
  # does the kernel appear in options to select?