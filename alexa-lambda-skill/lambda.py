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

import random
import uuid
import time
from datetime import datetime, timedelta

class AlexaResponse:
    def __init__(self, **kwargs):
        self.context_properties = []
        self.payload_endpoints = []
        self.payload_change = {}

        self.context = {}
        self.event = {
            'header': {
                'namespace': kwargs.get('namespace', 'Alexa'),
                'name': kwargs.get('name', 'Response'),
                'messageId': str(uuid.uuid4()),
                'payloadVersion': kwargs.get('payload_version', '3')
            },
            'endpoint': {
                "scope": {
                    "type": "BearerToken",
                    "token": kwargs.get('token', 'INVALID')
                },
                "endpointId": kwargs.get('endpoint_id', 'INVALID')
            },
            'payload': kwargs.get('payload', {})
        }

        if 'correlation_token' in kwargs:
            self.event['header']['correlation_token'] = kwargs.get('correlation_token', 'INVALID')

        if 'cookie' in kwargs:
            self.event['endpoint']['cookie'] = kwargs.get('cookie', '{}')

        if self.event['header']['name'] == 'AcceptGrant.Response' or self.event['header']['name'] == 'Discover.Response':
            self.event.pop('endpoint')

    def add_context_property(self, **kwargs):
        self.context_properties.append(self.create_context_property(**kwargs))

    def add_cookie(self, key, value):
        if "cookies" in self is None:
            self.cookies = {}
        self.cookies[key] = value

    def add_payload_endpoint(self, **kwargs):
        self.payload_endpoints.append(self.create_payload_endpoint(**kwargs))

    def add_payload_change(self, **kwargs):
        self.payload_change = self.create_payload_change(**kwargs)

    def create_context_property(self, **kwargs):
        return {
            'namespace': kwargs.get('namespace', 'Alexa.EndpointHealth'),
            'name': kwargs.get('name', 'connectivity'),
            'value': kwargs.get('value', {'value': 'OK'}),
            'timeOfSample': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.00Z"),
            'uncertaintyInMilliseconds': kwargs.get('uncertainty_in_milliseconds', 500)
        }

    def create_payload_endpoint(self, **kwargs):
        endpoint = {
            'capabilities': kwargs.get('capabilities', []),
            'description': kwargs.get('description', 'Zenoh Flow Endpoint Description'),
            'displayCategories': kwargs.get('display_categories', ['CONTACT_SENSOR']),
            'endpointId': kwargs.get('endpoint_id', 'virtual-routine-' + "%0.2d" % random.randint(0, 99)),
            'friendlyName': kwargs.get('friendly_name', 'Zenoh Flow Endpoint'),
            'manufacturerName': kwargs.get('manufacturer_name', 'Zenoh Flow Manufacturer')
        }

        if 'cookie' in kwargs:
            endpoint['cookie'] = kwargs.get('cookie', {})

        return endpoint

    def create_payload_change(self, **kwargs):
        change = {'cause': '', 'properties': ''}
        change['cause'] = {'type': 'PHYSICAL_INTERACTION'}
        change['properties'] = {
            'namespace': 'Alexa.ContactSensor',
            'name': 'detectionState',
            'value': kwargs.get('value', "DETECTED"),
            'timeOfSample': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.00Z"),
            'uncertaintyInMilliseconds': kwargs.get('uncertainty_in_milliseconds', 500)
        }

        return change

    def create_payload_endpoint_capability(self, **kwargs):
        capability = {
            'type': kwargs.get('type', 'AlexaInterface'),
            'interface': kwargs.get('interface', 'Alexa'),
            'version': kwargs.get('version', '3')
        }
        supported = kwargs.get('supported', None)
        if supported:
            capability['properties'] = {}
            capability['properties']['supported'] = supported
            capability['properties']['proactivelyReported'] = kwargs.get('proactively_reported', True)
            capability['properties']['retrievable'] = kwargs.get('retrievable', True)
        return capability

    def get(self, remove_empty=True):
        response = {
            'context': self.context,
            'event': self.event
        }

        if len(self.context_properties) > 0:
            response['context']['properties'] = self.context_properties

        if len(self.payload_endpoints) > 0:
            response['event']['payload']['endpoints'] = self.payload_endpoints

        if len(self.payload_change) > 0:
            response['event']['payload']['change'] = self.payload_change

        if remove_empty:
            if len(response['context']) < 1:
                response.pop('context')

        return response

    def set_payload(self, payload):
        self.event['payload'] = payload

    def set_payload_endpoint(self, payload_endpoints):
        self.payload_endpoints = payload_endpoints

    def set_payload_endpoints(self, payload_endpoints):
        if 'endpoints' not in self.event['payload']:
            self.event['payload']['endpoints'] = []
        self.event['payload']['endpoints'] = payload_endpoints

def lambda_handler(request, context):
    # Check if there is an Alexa directive
    if 'directive' not in request:
        rsp = AlexaResponse(
            name='ErrorResponse',
            payload={'type': 'INVALID_DIRECTIVE',
                     'message': 'Missing key: directive.'})
        return rsp.get()

    # Check payload version
    payload_version = request['directive']['header']['payloadVersion']
    if payload_version != '3':
        rsp = AlexaResponse(
            name='ErrorResponse',
            payload={'type': 'INTERNAL_ERROR',
                     'message': 'Make sure you are using Smart Home API version 3'})
        return rsp.get()

    # Handle request
    namespace = request['directive']['header']['namespace']
    if namespace == 'Alexa.Authorization':
        name = request['directive']['header']['name']
        if name == 'AcceptGrant':
            grant_code = request['directive']['payload']['grant']['code']
            grantee_token = request['directive']['payload']['grantee']['token']
            rsp = AlexaResponse(namespace='Alexa.Authorization', name='AcceptGrant.Response')
            return rsp.get()

    if namespace == 'Alexa.Discovery':
        name = request['directive']['header']['name']
        if name == 'Discover':
            rsp = AlexaResponse(namespace='Alexa.Discovery', name='Discover.Response')
            capability_alexa = rsp.create_payload_endpoint_capability()
            capability_alexa_powercontroller = rsp.create_payload_endpoint_capability(
                interface='Alexa.ContactSensor',
                supported=[{'name': 'detectionState'}])
            rsp.add_payload_endpoint(
                friendly_name='Virtual Routine',
                endpoint_id='virtual-routine-01',
                capabilities=[capability_alexa, capability_alexa_powercontroller])
            return rsp.get()

    if namespace == 'Alexa':
        name = request['directive']['header']['name']
        if name == 'ReportState':
            endpoint_id = request['directive']['endpoint']['endpointId']
            correlation_token = request['directive']['header']['correlationToken']
            token = request['directive']['endpoint']['scope']['token']
            state_value = "NOT_DETECTED" # TODO: To be fixed

            rsp = AlexaResponse(name='ChangeReport', correlation_token=correlation_token, endpoint_id=endpoint_id, token=token)
            rsp.add_context_property(namespace='Alexa.ContactSensor', name='detectionState', value=state_value)
            rsp.add_context_property()
            rsp.add_payload_change(value=state_value)

            return rsp.get()
