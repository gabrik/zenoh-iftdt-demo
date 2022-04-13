#
# Copyright (c) 2022 ZettaScale Technology
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
# which is available at https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
#
# Contributors:
#   ZettaScale Zenoh Team, <zenoh@zettascale.tech>
#

from zenoh_flow.interfaces import Sink

import json
import uuid
import requests
import os
import time
from datetime import datetime, timedelta
import threading

# Constants
ALEXA_URI = "https://api.eu.amazonalexa.com/v3/events"
LWA_TOKEN_URI = "https://api.amazon.com/auth/o2/token"
LWA_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
}
token_json = ""


def save_refresh_token(token):
    file_name = "/tmp/zf-iftdt.token"
    token_file = open(file_name, 'w')
    token_file.write(token)
    token_file.close()

def read_refresh_token():
    file_name = "/tmp/zf-iftdt.token"
    if os.path.isfile(file_name):
        token_file = open(file_name, 'r')
        token = token_file.readlines(token)
        token_file.close()
        return token[0]
    return ""

### Authentication functions
def get_token_params(client_id, client_secret, auth_code, refresh_token):
    global token_json

    token_params = {}

    if len(token_json) == 0:
        token_params["is_token_valid"] = False
        token_params["auth_code"] = auth_code
        token_params["refresh_token"] = refresh_token
        #print("Token does not exist yet")
    else:
        remaining_time = (token_json["expires_in"] - datetime.utcnow()).total_seconds()
        print(remaining_time)
        if remaining_time < 120:
            #print("Token has expired...it has to be renewed")
            token_params["is_token_valid"] = False
        else:
            #print("Token is still valid")
            token_params["is_token_valid"] = True

        token_params["access_token"] = token_json["access_token"]
        token_params["refresh_token"] = token_json["refresh_token"]

    return token_params

def get_access_token(client_id, client_secret, auth_code, refresh_token):
    global token_json

    token_params = get_token_params(client_id, client_secret, auth_code, refresh_token)

    if token_params["is_token_valid"]:
        return token_params
    else:
        if len(token_params["refresh_token"]) != 0:
            #print("Renewing token with the refresh token")
            lwa_params = {
                "grant_type" : "refresh_token",
                "refresh_token": token_params["refresh_token"],
                "client_id": client_id,
                "client_secret": client_secret
            }
        else:
            #print("Requesting token for the first time")
            lwa_params = {
                "grant_type" : "authorization_code",
                "code": auth_code,
                "client_id": client_id,
                "client_secret": client_secret
            }
        response = requests.post(LWA_TOKEN_URI, headers=LWA_HEADERS, data=lwa_params)
        if response.status_code != 200:
            return None

        token_json = json.loads(response.text)
        token_json["expires_in"] = datetime.utcnow() + timedelta(seconds=token_json["expires_in"])# + timedelta(hours=1)
        return token_json

def timeout(state):
    token = get_token(state)
    if token == None:
        return
    for item in state.person_mapping:
        action = "NOT_DETECTED"
        if action != item['last_state']:
            send_post(action, item['routine'], token)
            item['last_state'] = action

def get_token(state):
    tokens = get_access_token(state.client_id, state.client_secret, state.auth_code, state.refresh_token)
    #print(f'Tokens: {tokens}')
    token = tokens['access_token']
    rf_token = tokens['refresh_token']
    if token is None:
        print("Error while acquiring token")
        return None
    save_refresh_token(rf_token)
    state.refresh_token = rf_token
    return token

def send_post(action, routine, token):
    message_id = str(uuid.uuid4())
    time_of_sample = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.00Z")
    alexa_headers = {
        "Authorization": "Bearer {}".format(token),
        "Content-Type": "application/json;charset=UTF-8"
    }
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
                "endpointId": routine
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
    #print(f'Request {alexa_psu}')
    response = requests.post(ALEXA_URI, headers=alexa_headers, data=json.dumps(alexa_psu), allow_redirects=True)
    #print(f'Response {response.text}')



class AlexaSinkState:
    def __init__(self, configuration):
        if configuration['client_id'] is None:
            raise ValueError("Missing client ID for Alexa authentication")
        if configuration['client_secret'] is None:
            raise ValueError("Missing client secret for Alexa authentication")
        if configuration['auth_code'] is None and configuration['refresh_token'] is None:
            raise ValueError("Missing authentication code or refresh token for Alexa authentication")
        if configuration['person_mapping'] is None:
            raise ValueError("Missing person mapping to Alexa routine")

        self.client_id = configuration['client_id']
        self.client_secret = configuration['client_secret']
        self.auth_code = configuration['auth_code']
        self.person_mapping = configuration['person_mapping']

        if configuration['refresh_token'] is None:
            self.refresh_token = read_refresh_token()
        else:
            self.refresh_token = configuration['refresh_token']

        self.timer = None

        self.outfile = "/tmp/alexa-sink.csv"
        if configuration['outfile'] is not None:
            self.outfile = configuration['outfile']
        self.file = open(self.outfile, "w+")
        self.file.write("node,time_in,time_out,kind")
        self.file.flush()

class AlexaSink(Sink):
    def initialize(self, configuration):
        return AlexaSinkState(configuration)

    def finalize(self, state):
        state.file.close()
        return None

    def run(self, _ctx, state, data):
        intime = time.time_ns()

        if state.timer != None:
            state.timer.cancel()

        token = get_token(state)
        if token == None:
            return
        actions = json.loads(data.get_data())
        print(f'Actions {actions}')
        for item in state.person_mapping:
            action = "NOT_DETECTED"
            if actions.get(item['name']) is not None:
                if actions[item['name']] != 0.0:
                    print('Detected')
                    action = "DETECTED"

            if action != item['last_state']:
                print("Sending post")
                send_post(action, item['routine'], token)
                item['last_state'] = action

        outtime = time.time_ns()
        state.file.write(f'alexa-sink,{intime},{outtime},sink')
        state.file.flush()

        state.timer = threading.Timer(5, timeout, (state,))
        state.timer.start()

def register():
    return AlexaSink