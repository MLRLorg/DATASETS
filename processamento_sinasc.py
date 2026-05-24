import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sinasc_dataset = []

def filtrar_neonatal(df):
    df = df.copy()

    df["IDADE"] = (
        df["IDADE"]
        .astype(str)
        .str.strip()
    )

    df["IDADE_NUM"] = pd.to_numeric(
        df["IDADE"],
        errors="coerce"
    )

    df_neonatal = df[
        (df["IDADE_NUM"] < 229) &
        (df["IDADE_NUM"] > 0) &
        (df["IDADE_NUM"].notna()) 
    ].copy()

    del df

    df_neonatal = df_neonatal.drop(columns=["IDADE_NUM"])

    return df_neonatal

def filtrar_peso(df, coluna_peso="PESO", peso_min=350, peso_max=6500):
    df = df.copy()

    df[coluna_peso] = df[coluna_peso].astype(str).str.strip()
    df["PESO_NUM"] = pd.to_numeric(df[coluna_peso], errors="coerce")

    n_antes = df.shape[0]

    df = df[
        (df["PESO_NUM"] >= peso_min) &
        (df["PESO_NUM"] <= peso_max)
    ].copy()

    n_depois = df.shape[0]

    print(f"Registros antes: {n_antes}")
    print(f"Registros depois: {n_depois}")
    print(f"Registros removidos: {n_antes - n_depois}")

    df = df.drop(columns=["PESO_NUM"])

    return df


anos = range(15, 25)
for ano in anos:
    print(f"Lendo SINASC/{ano}.csv...")
    df = pd.read_csv(f'./SINASC/{ano}.csv', sep=";", dtype=str, encoding='latin-1')
    print(f"  → {df.shape[0]:,} registros, {df.shape[1]} colunas")
    sinasc_dataset.append(df)
    del df

sinasc_dataset = pd.concat(sinasc_dataset, ignore_index=True)

print(sinasc_dataset.shape)

sinasc_dataset = filtrar_peso(sinasc_dataset)

sinasc_dataset["TPROBSON_NUM"] = pd.to_numeric(
    sinasc_dataset["TPROBSON"].astype(str).str.strip(),
    errors="coerce"
)

grupos = [1, 2, 3, 4, 6]

sinasc_dataset["catTPROBSON"] = np.where(
    sinasc_dataset["TPROBSON_NUM"].isin(grupos),
    0,
    1
)

del grupos

sinasc_dataset = sinasc_dataset.drop(columns=["TPROBSON_NUM"])

peso = pd.to_numeric(
    sinasc_dataset["PESO"].astype(str).str.strip(),
    errors="coerce"
)

sinasc_dataset["catPeso"] = np.where(
    peso > 2500,
    1,
    0
)

del peso

sinasc_dataset["SEMAGESTAC_NUM"] = pd.to_numeric(
    sinasc_dataset["SEMAGESTAC"].astype(str).str.strip(),
    errors="coerce"
)

sinasc_dataset["catSEMAGESTAC"] = np.select(
    [
        sinasc_dataset["SEMAGESTAC_NUM"] < 37,
        (sinasc_dataset["SEMAGESTAC_NUM"] >= 37) & (sinasc_dataset["SEMAGESTAC_NUM"] < 42),
        sinasc_dataset["SEMAGESTAC_NUM"] >= 42
    ],
    [
        "pre_term",
        "term",
        "pos_term"
    ],
    default=None
)

sinasc_dataset.to_parquet('dataset_sinasc.parquet', compression='snappy', index=False)
del sim_dataset