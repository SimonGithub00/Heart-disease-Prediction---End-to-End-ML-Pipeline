from feast import Entity, Field, FileSource, FeatureView
from feast.types import Float32, Int32

patient = Entity(name="patient", join_keys=["patient_id"])
cp = Field(name="cp", dtype=Float32)
thalach = Field(name="thalach", dtype=Int32)
ca = Field(name="ca", dtype=Int32)
thal = Field(name="thal", dtype=Int32)

data_source = FileSource(
    path="heart_disease.parquet",
    event_timestamp_column="timestamp",
    created_timestamp_column="created",
)

heart_disease_fv = FeatureView(
    name="heart_disease",
    entities=[patient],
    schema=[cp, thalach, ca, thal],
    source=data_source,
)
# from feast import FeatureStore; store = FeatureStore(repo_path="."); store.apply([patient, heart_disease_fv])
