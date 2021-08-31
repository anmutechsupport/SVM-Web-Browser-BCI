"""
Record from a Stream
This example shows how to record data from an existing Muse LSL stream
"""
from muselsl import record
from muselsl.stream import find_muse
"""
Starting a Stream
This example shows how to search for available Muses and
create a new stream
"""
from muselsl import *

if __name__ == "__main__":

    # Note: an existing Muse LSL stream is required
    record(60)

    # Note: Recording is synchronous, so code here will not execute until the stream has been closed
    print('Recording has ended')