import pandas as pd

def save_solution(result, filepath="solution.csv"):
    """
    Salva o resultado completo da simulação em CSV.
    """
    detalhes = result["detalhes"]
    df = pd.DataFrame(detalhes)
    df.to_csv(filepath, index=False, sep=",", encoding="utf-8")
    print(f"✅ Solução salva em {filepath}")
