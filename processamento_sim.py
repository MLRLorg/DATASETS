import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sim_dataset = []

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


anos = range(15, 26)
for ano in anos:
    print(f"Lendo SIM/{ano}.csv...")
    df = pd.read_csv(f'./SIM/{ano}.csv', sep=";", dtype=str, encoding='latin-1')
    print(f"  → {df.shape[0]:,} registros, {df.shape[1]} colunas")
    sim_dataset.append(df)
    del df

sim_dataset = pd.concat(sim_dataset, ignore_index=True)

print(sim_dataset.shape)
sim_dataset = filtrar_neonatal(sim_dataset)

print(sim_dataset.shape)

sim_dataset = filtrar_peso(sim_dataset)

sim_dataset.to_parquet('dataset_sim.parquet', compression='snappy', index=False)
del sim_dataset