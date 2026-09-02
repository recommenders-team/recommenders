from typing import List
from pydantic import BaseModel, Field


########################
# Recommendation Models
########################

class CandidateGenerator(BaseModel):
    model: str = Field(..., description="Name/identifier of the recall model")
    data_needed: str = Field(..., description="Data required to train or score this model")
    batch_cadence: str = Field(..., description="Frequency of batch generation (e.g., hourly, daily)")


class ReRanker(BaseModel):
    architecture: str
    loss: str
    label_definition: str
    latency_target: str


class TrainingStack(BaseModel):
    etl: str
    framework: str
    hpo_strategy: str
    hardware: str


class ServingPath(BaseModel):
    storage: str
    ann_layer: str
    model_runtime: str
    p99_latency: str


class MetricsRollout(BaseModel):
    offline_metrics: List[str]
    online_kpis: List[str]
    rollout_strategy: str
    testing: str


class DocsIac(BaseModel):
    artifacts: List[str]
    iac_spec: str
    monitoring: str


class RecommendationDeploymentPlan(BaseModel):
    candidate_generators: List[CandidateGenerator]
    reranker: ReRanker
    training_stack: TrainingStack
    serving_path: ServingPath
    metrics_rollout: MetricsRollout
    docs_iac: DocsIac


########################
# Decision Engine Models
########################

class RankedAlgo(BaseModel):
    """Single algorithm entry with its selection rationale."""
    name: str
    why: str


class RankedAlgosResponse(BaseModel):
    """Top-N ranked algorithms returned by the decision engine."""
    ranked_algos: List[RankedAlgo]
