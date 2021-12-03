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
import random
import uuid
import logging
import sys
import time
import os
import requests
from datetime import datetime, timedelta

# Constants
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.00Z"
LWA_TOKEN_URI = "https://api.amazon.com/auth/o2/token"
LWA_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
}

TOKEN = ""

# Update to appropriate URI for your region
ALEXA_URI = "https://api.eu.amazonalexa.com/v3/events"

def get_utc_timestamp(seconds=None):
    t = datetime.utcnow() + timedelta(hours=1)
    return t.strftime(UTC_FORMAT)

def get_utc_timestamp_from_string(string):
    return datetime.strptime(string, UTC_FORMAT)

def get_uuid():
    return str(uuid.uuid4())

### Authentication functions
def get_need_new_token():
    """Checks whether the access token is missing or needed to be refreshed"""
    need_new_token_response = {
        "need_new_token": False,
        "access_token": "",
        "refresh_token": ""
    }

    if len(TOKEN) != 0:
        token = TOKEN.split("***")
        token_received_datetime = get_utc_timestamp_from_string(token[0])
        token_json = json.loads(token[1])
        token_expires_in = token_json["expires_in"] - PREEMPTIVE_REFRESH_TTL_IN_SECONDS
        token_expires_datetime = token_received_datetime + timedelta(seconds=token_expires_in)
        current_datetime = datetime.utcnow()

        need_new_token_response["need_new_token"] = current_datetime > token_expires_datetime
        need_new_token_response["access_token"] = token_json["access_token"]
        need_new_token_response["refresh_token"] = token_json["refresh_token"]
    else:
        need_new_token_response["need_new_token"] = True

    return need_new_token_response

def get_access_token(client_id, client_secret, code):
    # Performs access token or token refresh request as needed and returns valid access token
    need_new_token_response = get_need_new_token()
    access_token = ""

    if need_new_token_response["need_new_token"]:
        if len(TOKEN) != 0:
            lwa_params = {
                "grant_type" : "refresh_token",
                "refresh_token": need_new_token_response["refresh_token"],
                "client_id": client_id,
                "client_secret": client_secret
            }
        else:
            lwa_params = {
                "grant_type" : "authorization_code",
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret
            }
        response = requests.post(LWA_TOKEN_URI, headers=LWA_HEADERS, data=lwa_params)

        if response.status_code != 200:
            return None

        # store token in file
        token = get_utc_timestamp() + "***" + response.text
        TOKEN = token

        access_token = json.loads(response.text)["access_token"]
    else:
        access_token = need_new_token_response["access_token"]

    return access_token

class AlexaSinkState:
    def __init__(self, configuration):
        if configuration['client_id'] is None:
            raise ValueError("Missing client ID for Alexa authentication")
        if configuration['client_secret'] is None:
            raise ValueError("Missing client secret for Alexa authentication")
        if configuration['code'] is None:
            raise ValueError("Missing CODE for Alexa authentication")

        self.client_id = configuration['client_id']
        self.client_secret = configuration['client_secret']
        self.code = configuration['code']

class AlexaSink(Sink):
    def initialize(self, configuration):
        return AlexaSinkState(configuration)

    def finalize(self, _state):
        return None

    def run(self, _ctx, state, data):
        light = json.loads(data.data)
        if light['Gabriele_Baldoni'] == 0.0:
            action = "NOT_DETECTED"
        else
            action = "DETECTED"

        token = get_access_token(state.client_id, state.client_secret, state.code)
        alexa_headers = {
            "Authorization": "Bearer {}".format(token),
            "Content-Type": "application/json;charset=UTF-8"
        }

        if token:
            message_id = get_uuid()
            time_of_sample = get_utc_timestamp()

            # ensure that this change or state report is appropriate for your user and skill
            alexa_psu = {
                "context": {
                    "properties": [{
                        "namespace": "Alexa.EndpointHealth",
                        "name": "connectivity",
                        "value": {
                            "value": "OK"
                        },
                        "timeOfSample": time_of_sample,
                        "uncertaintyInMilliseconds": 500
                    }, {
                        "namespace": "Alexa.ContactSensor",
                        "name": "detectionState",
                        "value": action,
                        "timeOfSample": time_of_sample,
                        "uncertaintyInMilliseconds": 500
                    }]
                },
                "event": {
                    "header": {
                        "namespace": "Alexa",
                        "name": "ChangeReport",
                        "payloadVersion": "3",
                        "messageId": message_id
                    },
                    "endpoint": {
                        "scope": {
                            "type": "BearerToken",
                            "token": token
                        },
                        "endpointId": "virtual-routine-01"
                    },
                    "payload": {
                        "change": {
                            "cause": {
                                "type": "PHYSICAL_INTERACTION"
                            },
                            "properties": [{
                                "namespace": "Alexa.ContactSensor",
                                "name": "detectionState",
                                "value": action,
                                "timeOfSample": time_of_sample,
                                "uncertaintyInMilliseconds": 500
                            }]
                        }
                    }
                }
            }

            response = requests.post(ALEXA_URI, headers=alexa_headers, data=json.dumps(alexa_psu), allow_redirects=True)

def register():
    return AlexaSink
