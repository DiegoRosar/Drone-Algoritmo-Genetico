import math

# Raio médio da Terra (em km)
R = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância entre dois pontos (latitude/longitude) em quilômetros.
    Usa a fórmula de Haversine.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def bearing(lat1, lon1, lat2, lon2):
    """
    Calcula o ângulo (bearing) em graus entre dois pontos geográficos.
    0° = Norte, 90° = Leste, 180° = Sul, 270° = Oeste.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(x, y)
    return (math.degrees(brng) + 360) % 360


def destination_point(lat, lon, distance_km, bearing_deg):
    """
    Dada uma posição inicial (lat/lon), distância e direção,
    retorna o ponto final (lat2, lon2).
    """
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)
