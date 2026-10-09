import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "education": ["School", "Bachelor", "Master", "PhD", "Bachelor"]
})

encoder = OrdinalEncoder(
    categories=[["School", "Bachelor", "Master", "PhD"]]
)

encoded = encoder.fit_transform(df[["education"]])

print(encoded)