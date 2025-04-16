import logging
import json


class JsonLogger:
    def __init__(self, log_file="chattykg_json_log.json"):
        self.logger = logging.getLogger("JsonLogger")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.log_data, self.cost = {}, {}
        self.count = 0
        self.reset_logger()


    def reset_logger(self):
        self.log_data = {}
        cost_obj = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        self.cost = {'Understanding': cost_obj.copy(), 'Linking': cost_obj.copy(), 'Filtration': cost_obj.copy()}
        self.count = 0

    def set(self, key, value):
        if key in self.log_data:
            key = key + str(self.count)
            self.count = self.count + 1

        if isinstance(value, str):
            try:
                parsed_value = json.loads(value)
                self.log_data[key] = parsed_value
            except json.JSONDecodeError:
                self.log_data[key] = value
        else:
            self.log_data[key] = value

    def add_cost(self, key, cost_obj):
        for k, v in cost_obj.items():
            if k in self.cost[key]:
                self.cost[key][k] += v


    def log(self):
        self.log_data['Cost'] = self.cost
        json_message = json.dumps(self.log_data)
        self.logger.info(json_message)
        self.reset_logger()


