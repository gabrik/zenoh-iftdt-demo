import json
from zenoh_flow.interfaces import Operator
import struct

ON=1.0
OFF=0.0

class SensorFusionState:
    def __init__(self, configuration):
        self.know_people = configuration.get('people', [])

        self.outfile = "/tmp/fusion.csv"
        if configuration['outfile'] is not None:
            self.outfile = configuration['outfile']
        self.file = open(self.outfile, "w+")
        self.file.write("node,time_in,time_out,kind")
        self.file.flush()

class SensorFusion(Operator):
    def initialize(self, configuration):
        return SensorFusionState(configuration)

    def finalize(self, state):
        state.file.close()
        return None

    # def input_rule(self, _ctx, _state, tokens):
    #     people_token = tokens.get('People')
    #     lum_token = tokens.get('Luminosity')
    #     if people_token.is_pending():
    #         lum_token.set_action_drop()
    #         return False
    #     if lum_token.is_pending():
    #         return False

    #     return True

    def input_rule(self, _ctx, _state, tokens):
        people_token = tokens.get('People')
        lum_token = tokens.get('Luminosity')
        if lum_token.is_pending() and people_token.is_ready():
            people_token.set_action_drop()
            return False
        if people_token.is_pending() and lum_token.is_ready():
            lum_token.set_action_keep()
            return False

        lum_token.set_action_consume()
        people_token.set_action_consume()
        return True

    def output_rule(self, _ctx, state, outputs, _deadline_miss = None):
        return outputs

    def run(self, _ctx, state, inputs):
        intime = time.time_ns()

        # Getting the inputs
        people = json.loads(inputs.get('People').get_data())['detected_people']
        luminosity = struct.unpack('f',inputs.get('Luminosity').get_data())[0]

        # print(f'People {people}, Luminosity {luminosity}')
        output_value = {}
        for person in state.know_people:
            if person in people and luminosity < 0.3:
                output_value[person]=ON
            else:
                output_value[person]=OFF
        values = {'Lamps':bytes(json.dumps(output_value), 'utf-8')}

        outtime = time.time_ns()
        state.file.write(f'fusion,{intime},{outtime},operator')
        state.file.flush()

        return values

def register():
    return SensorFusion
