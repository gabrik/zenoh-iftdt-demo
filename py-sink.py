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

from zenoh_flow.interfaces import Sink
from zenoh_flow import Input
from zenoh_flow.types import Context
from typing import Dict, Any
import asyncio


class MySink(Sink):
    def finalize(self):
        return None

    def __init__(
        self,
        context: Context,
        configuration: Dict[str, Any],
        inputs: Dict[str, Input],
    ):
        self.input = inputs.get("Value", None)

    async def iteration(self) -> None:
        return None

def register():
    return MySink