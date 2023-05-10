from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset
from .surgical_dataset import Surgical

def get_trainval_datasets(tag, resize):
    if tag == 'aircraft':
        return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', resize=resize), DogDataset(phase='val', resize=resize)
    elif tag == 'sur':
        return Surgical(phase='train', resize=resize), Surgical(phase='val', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))
    
def get_trainval_datasets2(tag, resize):
    if tag == 'aircraft':
        return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', resize=resize), DogDataset(phase='val', resize=resize)
    elif tag == 'sur':
        return Surgical(phase='test', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))