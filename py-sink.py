##
## Copyright (c) 2017, 2021 ADLINK Technology Inc.
##
## This program and the accompanying materials are made available under the
## terms of the Eclipse Public License 2.0 which is available at
## http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
## which is available at https://www.apache.org/licenses/LICENSE-2.0.
##
## SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
##
## Contributors:
##   ADLINK zenoh team, <zenoh@adlink-labs.tech>
##

from zenoh_flow import Sink


class BasicSink(Sink):
    def initialize(self, _configuration):
        return {}
    def finalize(self, state):

        return None

    def run(self, _ctx, state, data):
        print(f'Data {data}')



def register():
    return BasicSink


def main():
    import time
    bs = BasicSink()
    state = bs.initialize({})
    while True:
        time.sleep(1)
        bs.run(None, state, b'Hello')

if __name__=='__main__':
    main()
