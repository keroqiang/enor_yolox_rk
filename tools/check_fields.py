import fiftyone as fo
import fiftyone.zoo as foz

ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Cat", "Dog"],
    dataset_name="test_fields",
)
schema = ds.get_field_schema()
print("字段列表:", list(schema.keys()))
sample = ds.first()
for fname in schema.keys():
    val = sample.get_field(fname)
    print(f"  {fname}: {type(val).__name__}")
