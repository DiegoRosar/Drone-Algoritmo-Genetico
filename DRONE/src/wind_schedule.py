import pandas as pd

class WindSchedule:
    """
    Lê o cronograma de ventos e permite consultar a velocidade/direção
    com base no dia e hora.
    """
    def __init__(self, filepath="data/wind_schedule.csv"):
        self.df = pd.read_csv(filepath)
        self.df = self.df.sort_values(by=["day", "hour"])

    def get_wind(self, day: int, hour: int):
        """
        Retorna o vento mais próximo do horário informado.
        """
        subset = self.df[(self.df["day"] == day) & (self.df["hour"] == hour)]
        if subset.empty:
            # se não houver dado exato, pega o mais próximo
            diffs = abs(self.df["hour"] - hour) + abs(self.df["day"] - day) * 24
            idx = diffs.idxmin()
            subset = self.df.iloc[[idx]]
        row = subset.iloc[0]
        return {
            "speed_kmh": float(row["wind_speed_kmh"]),
            "dir_deg_from": float(row["wind_dir_deg_from"])
        }
