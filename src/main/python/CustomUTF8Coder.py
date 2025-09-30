
from apache_beam.coders.coders import Coder
import logging

class CustomUTF8Coder(Coder):
  """A custom coder used for utf-8 by default and then iso8859-1 by
  exception
  """
  def __init__(self):
    self._logger = logging.getLogger(__name__)

  def encode(self, value):
    try:
      return value.encode("utf-8")
    except Exception as ex2:
      self._logger.error(f"Error encoding to UTF-8 value='{value}': {e}")
      return None

  def decode(self, value):
    try:
      return value.decode("utf-8")
    except Exception as ex:
      try:
        return value.decode("iso-8859-1")
      except Exception as ex2:
        self._logger.error(f"Error decoding value='{value}': {e}")
        return None

  def is_deterministic(self):
    return True