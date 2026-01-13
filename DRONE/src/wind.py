import math

def effective_speed(drone_speed_kmh, wind_speed_kmh, drone_bearing_deg, wind_dir_deg_from):
    """
    Calcula a velocidade efetiva do drone considerando o vento.

    - drone_speed_kmh: velocidade nominal do drone (sem vento)
    - wind_speed_kmh: intensidade do vento
    - drone_bearing_deg: direção do voo (graus)
    - wind_dir_deg_from: direção de onde o vento vem (graus)
    """
    # Converte direções em radianos
    drone_rad = math.radians(drone_bearing_deg)
    wind_rad = math.radians(wind_dir_deg_from)

    # Diferença angular entre o vento e o voo
    diff = drone_rad - wind_rad

    # Projeção do vento na direção do voo
    effective = drone_speed_kmh + wind_speed_kmh * math.cos(diff)

    # Limita para não ficar menor que 10 m/s (36 km/h)
    return max(effective, 36.0)
