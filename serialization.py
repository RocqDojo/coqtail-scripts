from json import dump, JSONEncoder
from dataclasses import asdict, dataclass, field, is_dataclass


class EnhancedJSONEncoder(JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default


def json_dump(obj, filename: str):
    dump(obj, fp=open(filename, "w"), cls=EnhancedJSONEncoder)
