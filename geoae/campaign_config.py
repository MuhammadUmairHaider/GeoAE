"""Original DPC training schema for the matched mid/final-layer AE campaign."""
from copy import copy
from dataclasses import dataclass, field, fields, make_dataclass

from geoae.config import Config, TrainConfig


OTHER_EXPERIMENT_FIELDS = frozenset({"lloyd_rows", "lloyd_iters", "freeze_centroids"})
_campaign_fields = []
for _definition in fields(TrainConfig):
    if _definition.name in OTHER_EXPERIMENT_FIELDS:
        continue
    _default = copy(_definition)
    if _definition.name == "centroid_init":
        _default.default = "dpc"
    _campaign_fields.append((_definition.name, _definition.type, _default))

CampaignTrainConfig = make_dataclass("CampaignTrainConfig", _campaign_fields)
CampaignTrainConfig.__module__ = __name__


@dataclass
class CampaignConfig(Config):
    train: CampaignTrainConfig = field(default_factory=CampaignTrainConfig)

    def validate(self):
        super().validate()
        if self.train.centroid_init != "dpc":
            raise ValueError("The matched AE campaign uses the original DPC initializer")
        return self
