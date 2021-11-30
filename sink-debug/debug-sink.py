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
import json

class DebugSink(Sink):
    def initialize(self, _configuration):
        return None

    def finalize(self, _state):
        return None

    def run(self, _ctx, _state, data):
        people = data.decode('utf-8')
        people = json.loads(people)
        printl(f"Detected people: {people}")

def register():
    return DebugSink
