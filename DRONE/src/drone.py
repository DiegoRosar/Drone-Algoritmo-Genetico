from src.geometry import haversine, bearing
from src.wind import effective_speed

class Drone:
    def __init__(self, autonomy_km=30.0, base_speed_kmh=54.0, cost_per_recharge=80.0):
        self.autonomy_km = autonomy_km * 0.93  # ajuste matrícula começa com 2
        self.base_speed_kmh = base_speed_kmh
        self.cost_per_recharge = cost_per_recharge
        self.battery_km = self.autonomy_km
        self.total_cost = 0.0
        self.total_time_h = 0.0

    def fly(self, lat1, lon1, lat2, lon2, wind_speed_kmh, wind_dir_deg_from):
        """
        Simula o voo entre dois pontos, retornando:
        (tempo_horas, custo, pouso_para_recarga)
        """
        distance = haversine(lat1, lon1, lat2, lon2)
        direction = bearing(lat1, lon1, lat2, lon2)
        effective = effective_speed(self.base_speed_kmh, wind_speed_kmh, direction, wind_dir_deg_from)

        # Tempo de voo
        time_h = distance / effective
        self.total_time_h += time_h

        # Verifica autonomia
        if distance > self.battery_km:
            # Drone precisou pousar e recarregar
            self.total_cost += self.cost_per_recharge
            self.battery_km = self.autonomy_km - distance
            return time_h, self.cost_per_recharge, True
        else:
            self.battery_km -= distance
            return time_h, 0.0, False
