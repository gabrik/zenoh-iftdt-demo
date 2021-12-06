import json
from zenoh_flow import Inputs, Operator, Outputs
import struct

ON=1.0
OFF=0.0

KNOWN_PEOPLE = ["SomeOne"]

class SensorFusion(Operator):
    def initialize(self, configuration):
         return None

    def finalize(self, state):
        return None

    def input_rule(self, _ctx, _state, tokens):
        people_token = tokens.get('People')
        lum_token = tokens.get('Luminosity')
        if people_token.is_pending():
            lum_token.set_action_drop()
            return False
        if lum_token.is_pending():
            return False

        return True


    def output_rule(self, _ctx, state, outputs, _deadline_miss):
        return outputs

    def run(self, _ctx, state, inputs):
        # Getting the inputs
        people = json.loads(inputs.get('People').data)['detected_people']
        luminosity = struct.unpack('f',inputs.get('Luminosity').data)[0]

        # print(f'People {people}, Luminosity {luminosity}')
        output_value = {}
        for person in KNOWN_PEOPLE:
            if person in people and luminosity < 0.5:
                output_value[person]=ON
            else:
                output_value[person]=OFF


        return {'Lamps':bytes(json.dumps(output_value), 'utf-8')}

def register():
    return SensorFusion
