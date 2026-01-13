import pandas as pd
from src.drone import Drone
from src.wind_schedule import WindSchedule

class Simulator:
    """
    Simula o voo do drone entre uma lista de CEPs.
    Considera vento, autonomia, recargas e custo.
    """
    def __init__(self, ceps_df: pd.DataFrame, wind_schedule: WindSchedule):
        self.ceps_df = ceps_df
        self.wind_schedule = wind_schedule

    def simulate_route(self, route, start_day=1, start_hour=6):
        """
        Recebe uma lista de CEPs e retorna estatísticas da rota:
        - tempo total
        - custo total
        - lista detalhada com cada trecho
        """
        drone = Drone()
        total_time_h = 0.0
        total_cost = 0.0
        results = []

        for i in range(len(route) - 1):
            cep1 = route[i]
            cep2 = route[i + 1]
            p1 = self.ceps_df[self.ceps_df["cep"] == cep1].iloc[0]
            p2 = self.ceps_df[self.ceps_df["cep"] == cep2].iloc[0]

            day = start_day
            hour = start_hour + total_time_h

            wind = self.wind_schedule.get_wind(day, int(hour % 24))
            time_h, cost, recharge = drone.fly(
                p1["latitude"], p1["longitude"],
                p2["latitude"], p2["longitude"],
                wind["speed_kmh"], wind["dir_deg_from"]
            )

            total_time_h += time_h
            total_cost += cost

            results.append({
                "cep_inicial": cep1,
                "lat_inicial": p1["latitude"],
                "lon_inicial": p1["longitude"],
                "dia": day,
                "hora_inicio": round(hour, 2),
                "velocidade_kmh": round(drone.base_speed_kmh, 2),
                "cep_final": cep2,
                "lat_final": p2["latitude"],
                "lon_final": p2["longitude"],
                "pouso": recharge,
                "hora_final": round(hour + time_h, 2)
            })

        return {
            "tempo_total_h": total_time_h,
            "custo_total": total_cost,
            "detalhes": results
        }
  