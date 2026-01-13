#!/usr/bin/env python3
"""
run.py - Versão Completa (detalhada) com execução paralela por tempo (no-mode).
- Mantém regras do enunciado (autonomia 0.93, janelas, vento, recargas, penalidade +R$80 após 17:00).
- Fitness penalizado (NUNCA INF) para permitir evolução e métricas.
- Modo 'local' (1 GA) e 'parallel' (N agentes com limite por tempo).
- Checkpoints e estatísticas por agente salvos em out/.
- Inclui diagnóstico (opção C): grava out/diagnostic_agent_X.json e out/diagnostic_summary.json
- Recomendações: usar venv ativado antes de rodar.
"""

from __future__ import annotations
import os
import csv
import json
import time
import math
import random
import shutil
import signal
import argparse
from math import ceil
from datetime import datetime, timedelta, date, time as dt_time
from multiprocessing import Process, cpu_count
from typing import Dict, List, Tuple, Any
from collections import Counter

import pandas as pd

# ----------------------------
# DEFAULT CONFIG (mude conforme necessário)
# ----------------------------
BASE_CEP = "82821020"

# autonomia (segundos) e ajuste por matrícula começando com 2
AUTONOMY_NOMINAL_S = 5000.0
AUTONOMY_FACTOR = 0.93
AUTONOMY_S = AUTONOMY_NOMINAL_S * AUTONOMY_FACTOR

TIME_PER_STOP_S = 72
V_REF_KMH = 36.0
MIN_SPEED_KMH = 36
MAX_SPEED_KMH = 96
SPEED_STEP = 4

FLY_START_HOUR = 6
FLY_END_HOUR = 19
DAYS_LIMIT = 7

RECHARGE_COST = 80.0
ADDITIONAL_AFTER_17_COST = 80.0

# Fitness weights & penalty constants (ajustáveis)
WEIGHT_MONEY_AS_SECONDS = 60.0  # converte R$ em "segundos-penalidade"
PENALTY_EXCEED_AUTONOMY_PER_KM = 5000.0
PENALTY_AFTER_19_PER_HOUR = 20000.0
PENALTY_VELOCITY_TOO_LOW = 8000.0
PENALTY_ILLEGAL_LANDING = 20000.0
PENALTY_GENERAL = 50000.0

# Defaults for GA per agent
DEFAULT_POP_SIZE = 10
DEFAULT_MUTATION_RATE = 0.2
DEFAULT_GENERATIONS = 1000000000  # practically infinite; we'll stop by time
TOURNAMENT_K = 3
ELITISM = True

# I/O defaults
DEFAULT_CEPS_PATH = "data/ceps.csv"
DEFAULT_WIND_PATH = "data/wind_schedule.csv"
UPLOADED_COORDS = "/mnt/data/coordenadas.csv"  # path from earlier upload (script will copy if needed)

# ----------------------------
# Diagnostic globals
# ----------------------------
# Each agent will maintain its own diagnostics list and save it to out/diagnostic_agent_{id}.json
# Diagnostic record format: {"individual": int, "ceps_valid": int, "fail_cep": str or None}

# ----------------------------
# Utility: Geodesy and wind
# ----------------------------
R_EARTH = 6371000.0

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R_EARTH * c

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_m(lat1, lon1, lat2, lon2) / 1000.0

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r = math.radians(lat1); lat2r = math.radians(lat2)
    dlonr = math.radians(lon2 - lon1)
    y = math.sin(dlonr) * math.cos(lat2r)
    x = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlonr)
    br = math.degrees(math.atan2(y, x))
    return (br + 360) % 360

def wind_vector_kmh(wind_speed_kmh: float, wind_dir_deg_from: float) -> Tuple[float,float]:
    # convert meteorological 'from' degrees to vector pointing 'to' in math coordinates
    to_deg = (wind_dir_deg_from + 180) % 360
    rad = math.radians(to_deg)
    wx = wind_speed_kmh * math.cos(rad)
    wy = wind_speed_kmh * math.sin(rad)
    return wx, wy

def wind_along_heading_kmh(wind_speed_kmh: float, wind_dir_deg_from: float, heading_deg: float) -> float:
    wx, wy = wind_vector_kmh(wind_speed_kmh, wind_dir_deg_from)
    rad = math.radians(heading_deg)
    ux = math.cos(rad); uy = math.sin(rad)
    return wx * ux + wy * uy

def effective_ground_speed_kmh(v_set_kmh: float, heading_deg: float, wind_speed_kmh: float, wind_dir_deg_from: float) -> float:
    comp = wind_along_heading_kmh(wind_speed_kmh, wind_dir_deg_from, heading_deg)
    v_eff = v_set_kmh + comp
    if v_eff < MIN_SPEED_KMH:
        v_eff = MIN_SPEED_KMH
    return v_eff

def battery_needed_seconds(time_seconds: float, v_effective_kmh: float) -> float:
    cf = (v_effective_kmh / V_REF_KMH) ** 2
    return time_seconds * cf + TIME_PER_STOP_S

# ----------------------------
# Wind schedule parsing & lookup
# ----------------------------
def parse_wind_csv(path: str) -> Dict[Tuple[int,int], Dict[str,float]]:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if 'hour' not in df.columns:
        raise Exception("wind_schedule.csv precisa conter a coluna 'hour'")
    def hr(x):
        s = str(x).strip().replace("h","").replace(":","")
        return int(float(s))
    df['hour'] = df['hour'].apply(hr)
    if 'wind_speed_kmh' not in df.columns and 'wind_speed' in df.columns:
        df['wind_speed_kmh'] = df['wind_speed']
    if 'wind_dir_deg_from' not in df.columns and 'wind_dir' in df.columns:
        mapping = {"N":0,"NNE":22,"NE":45,"ENE":70,"E":90,"ESE":110,"SE":135,"SSE":160,
                   "S":180,"SSW":200,"SW":225,"WSW":250,"W":270,"WNW":295,"NW":315,"NNW":337}
        df['wind_dir_deg_from'] = df['wind_dir'].map(lambda x: mapping.get(str(x).strip(), 90))
    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)
    df['wind_speed_kmh'] = df['wind_speed_kmh'].astype(float)
    df['wind_dir_deg_from'] = df['wind_dir_deg_from'].astype(float)
    wind = {}
    for _, r in df.iterrows():
        wind[(int(r['day']), int(r['hour']))] = {'speed_kmh': float(r['wind_speed_kmh']), 'dir_deg_from': float(r['wind_dir_deg_from'])}
    return wind

def get_wind_for_day_hour(wind_dict: Dict[Tuple[int,int],Dict[str,float]], day: int, hour: int) -> Dict[str,float]:
    slots = sorted([h for (d,h) in wind_dict.keys() if d == day])
    if not slots:
        slots = sorted(set([h for (d,h) in wind_dict.keys()]))
    if not slots:
        return {'speed_kmh':0.0, 'dir_deg_from':90.0}
    chosen = max([s for s in slots if s <= hour], default=slots[0])
    return wind_dict.get((day, chosen), {'speed_kmh':0.0, 'dir_deg_from':90.0})

# ----------------------------
# Diagnostic helper: simulate until first HARD failure
# ----------------------------
def simulate_until_failure(route_ceps: List[str], speeds_kmh: List[int], ceps_map: Dict[str,Dict[str,float]], wind_schedule: Dict[Tuple[int,int],Dict[str,float]]) -> Tuple[int, Any]:
    """Retorna (ceps_valid, fail_cep_or_None)
    Ceil the number of CEPs successfully completed (legs) before hitting a hard failure.
    Condições de falha consideradas:
      - need_battery > AUTONOMY_S * 1.0001 (impraticável mesmo com bateria cheia)
      - v_eff <= 0 (velocidade efetiva inválida)
      - day_index > DAYS_LIMIT (extrapolou o número de dias permitido)
    """
    battery = AUTONOMY_S
    start_dt = datetime.combine(date.today(), dt_time(FLY_START_HOUR, 0, 0))
    current_dt = start_dt
    ceps_valid = 0
    for i in range(len(route_ceps)-1):
        cep_a = route_ceps[i]; cep_b = route_ceps[i+1]
        p1 = ceps_map[cep_a]; p2 = ceps_map[cep_b]
        lat1, lon1 = p1['latitude'], p1['longitude']
        lat2, lon2 = p2['latitude'], p2['longitude']
        heading = bearing_deg(lat1, lon1, lat2, lon2)
        distance_km = haversine_km(lat1, lon1, lat2, lon2)
        day_index = (current_dt.date() - start_dt.date()).days + 1
        if day_index > DAYS_LIMIT:
            return ceps_valid, cep_a
        wind = get_wind_for_day_hour(wind_schedule, day_index, current_dt.hour)
        v_set = speeds_kmh[i]
        if (v_set % SPEED_STEP != 0) or v_set < MIN_SPEED_KMH or v_set > MAX_SPEED_KMH:
            v_set = MIN_SPEED_KMH
        v_eff = effective_ground_speed_kmh(v_set, heading, wind['speed_kmh'], wind['dir_deg_from'])
        if v_eff <= 0:
            return ceps_valid, cep_a
        time_hours = distance_km / v_eff if v_eff > 0 else 9999.0
        time_seconds = ceil(time_hours * 3600)
        need_battery = battery_needed_seconds(time_seconds, v_eff)
        if need_battery > AUTONOMY_S * 1.0001:
            return ceps_valid, cep_a
        # if need battery, simulate recharge
        if need_battery > battery:
            current_dt += timedelta(seconds=TIME_PER_STOP_S)
            battery = AUTONOMY_S
            if current_dt.time() > dt_time(FLY_END_HOUR,0,0):
                # roll to next day
                current_dt = datetime.combine(current_dt.date() + timedelta(days=1), dt_time(FLY_START_HOUR,0,0))
                if (current_dt.date() - start_dt.date()).days + 1 > DAYS_LIMIT:
                    return ceps_valid, cep_a
        # perform flight
        battery -= min(need_battery, battery)
        arrival_time = current_dt + timedelta(seconds=time_seconds)
        current_dt = arrival_time + timedelta(seconds=TIME_PER_STOP_S)
        # if arrival exceeds end of day, move to next day
        if current_dt.time() > dt_time(FLY_END_HOUR,0,0):
            current_dt = datetime.combine(current_dt.date() + timedelta(days=1), dt_time(FLY_START_HOUR,0,0))
            if (current_dt.date() - start_dt.date()).days + 1 > DAYS_LIMIT:
                return ceps_valid+1, cep_b
        ceps_valid += 1
    return ceps_valid, None

# ----------------------------
# Simulation: penalized (returns numeric score always)
# ----------------------------
def simulate_full_route_penalized(route_ceps: List[str], speeds_kmh: List[int],
                                  ceps_map: Dict[str,Dict[str,float]],
                                  wind_schedule: Dict[Tuple[int,int],Dict[str,float]],
                                  verbose: bool=False) -> Dict[str,Any]:
    # ... (identical to original implementation) -- keep same body for penalized sim
    battery = AUTONOMY_S
    rows = []
    total_time_s = 0.0
    total_cost = 0.0
    total_penalty = 0.0

    start_dt = datetime.combine(date.today(), dt_time(FLY_START_HOUR, 0, 0))
    current_dt = start_dt

    # iterate legs
    for i in range(len(route_ceps) - 1):
        cep_a = route_ceps[i]; cep_b = route_ceps[i+1]
        p1 = ceps_map[cep_a]; p2 = ceps_map[cep_b]
        lat1, lon1 = p1['latitude'], p1['longitude']
        lat2, lon2 = p2['latitude'], p2['longitude']

        heading = bearing_deg(lat1, lon1, lat2, lon2)
        distance_km = haversine_km(lat1, lon1, lat2, lon2)

        day_index = (current_dt.date() - start_dt.date()).days + 1
        if day_index < 1:
            day_index = 1
        if day_index > DAYS_LIMIT:
            # penalize but continue simulation with day clamped
            total_penalty += PENALTY_GENERAL * (day_index - DAYS_LIMIT)
            # set time to last day end to continue
            current_dt = datetime.combine(start_dt.date() + timedelta(days=DAYS_LIMIT-1), dt_time(FLY_END_HOUR,0,0))

        hour_now = current_dt.hour
        wind = get_wind_for_day_hour(wind_schedule, day_index, hour_now)

        v_set = speeds_kmh[i]
        # enforce discrete speeds: if invalid, penalize and fix to minimum safe speed
        if (v_set % SPEED_STEP != 0) or v_set < MIN_SPEED_KMH or v_set > MAX_SPEED_KMH:
            total_penalty += PENALTY_GENERAL
            v_set = MIN_SPEED_KMH

        v_eff = effective_ground_speed_kmh(v_set, heading, wind['speed_kmh'], wind['dir_deg_from'])
        if v_eff <= 0:
            total_penalty += PENALTY_VELOCITY_TOO_LOW
            v_eff = MIN_SPEED_KMH

        time_hours = distance_km / v_eff if v_eff > 0 else 9999.0
        time_seconds = ceil(time_hours * 3600)
        need_battery = battery_needed_seconds(time_seconds, v_eff)

        # if battery insufficient -> perform recharge (pouso) BEFORE the leg
        if need_battery > battery:
            total_time_s += TIME_PER_STOP_S
            current_dt += timedelta(seconds=TIME_PER_STOP_S)
            total_cost += RECHARGE_COST
            # landing after 17h increases operational cost and receives penalty
            if current_dt.time() >= dt_time(17,0,0):
                total_cost += ADDITIONAL_AFTER_17_COST
                # also add numeric penalty scaled by monetary weight
                total_penalty += ADDITIONAL_AFTER_17_COST * WEIGHT_MONEY_AS_SECONDS
            battery = AUTONOMY_S

            # if recharge pushed beyond flight window -> penalize and roll to next day
            if current_dt.time() > dt_time(FLY_END_HOUR,0,0):
                # penalize by hours exceeded
                hours_exceeded = (current_dt.hour - FLY_END_HOUR) if current_dt.hour > FLY_END_HOUR else 1
                total_penalty += PENALTY_AFTER_19_PER_HOUR * hours_exceeded
                # resume next day at start hour
                current_dt = datetime.combine(current_dt.date() + timedelta(days=1), dt_time(FLY_START_HOUR, 0, 0))

            # recompute wind & v_eff/time after recharge
            day_index = (current_dt.date() - start_dt.date()).days + 1
            if day_index > DAYS_LIMIT:
                total_penalty += PENALTY_GENERAL * (day_index - DAYS_LIMIT)
            wind = get_wind_for_day_hour(wind_schedule, day_index, current_dt.hour)
            v_eff = effective_ground_speed_kmh(v_set, heading, wind['speed_kmh'], wind['dir_deg_from'])
            time_hours = distance_km / v_eff if v_eff > 0 else 9999.0
            time_seconds = ceil(time_hours * 3600)
            need_battery = battery_needed_seconds(time_seconds, v_eff)
            if need_battery > AUTONOMY_S * 1.0001:
                # impraticável mesmo com full battery -> large penalty but continue
                total_penalty += PENALTY_GENERAL * (need_battery / AUTONOMY_S)

        # perform flight (use up battery)
        battery -= min(need_battery, battery)
        depart_time = current_dt
        arrival_time = current_dt + timedelta(seconds=time_seconds)
        arrival_with_stop = arrival_time + timedelta(seconds=TIME_PER_STOP_S)

        total_time_s += time_seconds + TIME_PER_STOP_S
        current_dt = arrival_with_stop

        # if arrival crosses end-of-day window -> penalize and resume next day at start hour
        if arrival_with_stop.time() > dt_time(FLY_END_HOUR,0,0):
            hours_exceeded = (arrival_with_stop.hour - FLY_END_HOUR) if arrival_with_stop.hour > FLY_END_HOUR else 1
            total_penalty += PENALTY_AFTER_19_PER_HOUR * hours_exceeded
            current_dt = datetime.combine(current_dt.date() + timedelta(days=1), dt_time(FLY_START_HOUR,0,0))

        pouso_flag = 'SIM' if battery < AUTONOMY_S*0.5 else 'NAO'

        rows.append({
            'cep_inicial': cep_a,
            'lat_inicial': lat1,
            'lon_inicial': lon1,
            'dia': (depart_time.date() - start_dt.date()).days + 1,
            'hora_inicio': depart_time.time().isoformat(),
            'velocidade_kmh': v_set,
            'cep_final': cep_b,
            'lat_final': lat2,
            'lon_final': lon2,
            'pouso': pouso_flag,
            'hora_final': arrival_time.time().isoformat()
        })

    # final day check
    final_day_idx = (current_dt.date() - start_dt.date()).days + 1
    if final_day_idx > DAYS_LIMIT:
        total_penalty += PENALTY_GENERAL * (final_day_idx - DAYS_LIMIT)

    fitness = total_time_s + WEIGHT_MONEY_AS_SECONDS * total_cost + total_penalty
    valid = (total_penalty == 0 and final_day_idx <= DAYS_LIMIT)
    return {'valid': valid, 'fitness': fitness, 'total_time_s': total_time_s, 'total_cost': total_cost, 'penalty': total_penalty, 'rows': rows}

# ----------------------------
# GA: operators and helpers
# ----------------------------
def random_speed() -> int:
    return random.choice(list(range(MIN_SPEED_KMH, MAX_SPEED_KMH+1, SPEED_STEP)))

def make_individual(all_middle_ceps: List[str]) -> Tuple[List[str], List[int]]:
    perm = all_middle_ceps[:]
    random.shuffle(perm)
    speeds = [random_speed() for _ in range(len(perm)+1)]
    return (perm, speeds)

def decode_individual(ind: Tuple[List[str], List[int]]) -> Tuple[List[str], List[int]]:
    perm, speeds = ind
    return [BASE_CEP] + perm + [BASE_CEP], speeds

def fitness_individual_penalized(ind: Tuple[List[str], List[int]],
                                 ceps_map: Dict[str,Dict[str,float]],
                                 wind_schedule: Dict[Tuple[int,int],Dict[str,float]]) -> Tuple[float, Dict[str,Any]]:
    route, speeds = decode_individual(ind)
    sim = simulate_full_route_penalized(route, speeds, ceps_map, wind_schedule)
    return sim['fitness'], sim

def tournament_selection(pop: List[Tuple[List[str],List[int]]], scores: List[float], k: int=TOURNAMENT_K):
    items = list(zip(pop, scores))
    sampled = random.sample(items, min(k, len(items)))
    sampled.sort(key=lambda x: x[1])
    return sampled[0][0]

def safe_ordered_crossover(parent1: List[str], parent2: List[str]) -> List[str]:
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    missing = [x for x in parent2 if x not in child]
    mi = 0
    for i in range(size):
        if child[i] is None:
            child[i] = missing[mi]; mi += 1
    return child

def crossover(ind1: Tuple[List[str],List[int]], ind2: Tuple[List[str],List[int]]) -> Tuple[List[str],List[int]]:
    p1, s1 = ind1; p2, s2 = ind2
    child_perm = safe_ordered_crossover(p1, p2)
    cut = random.randint(1, len(s1)-1)
    child_speeds = s1[:cut] + s2[cut:]
    return (child_perm, child_speeds)

def mutate(ind: Tuple[List[str],List[int]], mutation_rate: float=DEFAULT_MUTATION_RATE) -> Tuple[List[str],List[int]]:
    perm, speeds = ind
    if random.random() < mutation_rate and len(perm) > 1:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
    for idx in range(len(speeds)):
        if random.random() < mutation_rate:
            speeds[idx] = random_speed()
    return (perm, speeds)

# ----------------------------
# IO: load ceps & wind, save solution
# ----------------------------
def load_ceps(path: str = DEFAULT_CEPS_PATH) -> Dict[str,Dict[str,float]]:
    if not os.path.exists(path):
        # try to fallback to uploaded file path if present
        if os.path.exists(UPLOADED_COORDS):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy(UPLOADED_COORDS, path)
            print(f"[INFO] arquivo {path} não encontrado — copiando upload disponível {UPLOADED_COORDS}")
        else:
            raise FileNotFoundError(f"Arquivo de CEPs não encontrado: {path}")
    df = pd.read_csv(path, dtype={'cep':str})
    cols = [c.strip().lower() for c in df.columns]
    # normalize columns if header format is cep,longitude,latitude
    if 'latitude' not in cols or 'longitude' not in cols:
        df.columns = ['cep','longitude','latitude'] + list(df.columns[3:])
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    ceps = {}
    for _, r in df.iterrows():
        ceps[str(r['cep'])] = {'latitude': float(r['latitude']), 'longitude': float(r['longitude'])}
    return ceps

def load_wind(path: str = DEFAULT_WIND_PATH) -> Dict[Tuple[int,int],Dict[str,float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo wind_schedule.csv não encontrado em: {path}")
    return parse_wind_csv(path)

def save_solution_rows(rows: List[Dict[str,Any]], path: str = "solution.csv") -> None:
    fieldnames = ['cep_inicial','lat_inicial','lon_inicial','dia','hora_inicio','velocidade_kmh',
                  'cep_final','lat_final','lon_final','pouso','hora_final']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ----------------------------
# Agent worker: runs until time limit (per-agent)
# ----------------------------
def agent_worker(agent_id: int, time_limit_seconds: int, pop_size: int, generations_limit: int,
                 seed: int, ceps_path: str, wind_path: str, out_dir: str, mutation_rate: float):
    """
    Worker que executa um GA local até atingir time_limit_seconds.
    Salva:
      - out/best_agent_{id}.json  (melhor indivíduo)
      - out/stats_agent_{id}.csv  (métricas da evolução)
      - out/diagnostic_agent_{id}.json  (quantos CEPs cada indivíduo conseguiu percorrer)
    """

    random.seed(seed + agent_id * 7919)
    start_time = time.time()

    os.makedirs(out_dir, exist_ok=True)

    # Carrega CEPs e vento
    ceps_map = load_ceps(ceps_path)
    wind_sched = load_wind(wind_path)

    # Todos os CEPs menos o CEP base
    all_middle = [c for c in ceps_map.keys() if c != BASE_CEP]

    # População inicial
    population = [make_individual(all_middle) for _ in range(pop_size)]
    best = None
    best_score = float('inf')
    gen = 0

    # Lista de DIAGNÓSTICO deste agente
    diagnostics: List[Dict[str, Any]] = []

    # Arquivo de estatísticas
    stats_file = os.path.join(out_dir, f"stats_agent_{agent_id}.csv")
    with open(stats_file, 'w', newline='', encoding='utf-8') as sf:
        writer = csv.writer(sf)
        writer.writerow(['gen', 'best', 'avg', 'valid_percent', 'time_elapsed_s'])

    print(f"[agent {agent_id}] START time_limit={time_limit_seconds}s pop={pop_size} seed={seed}")

    try:
        while True:
            gen += 1
            scores = []
            valid_count = 0

            # --- AVALIAÇÃO DE CADA INDIVÍDUO -----------------------
            for idx, ind in enumerate(population):

                # --- Telemetria: quantos CEPs ele consegue antes de falhar?
                route, speeds = decode_individual(ind)
                ceps_valid, fail_cep = simulate_until_failure(route, speeds, ceps_map, wind_sched)

                diagnostics.append({
                    'individual': idx,
                    'gen': gen,
                    'ceps_valid': int(ceps_valid),
                    'fail_cep': fail_cep
                })

                # --- Fitness penalizado
                score, sim = fitness_individual_penalized(ind, ceps_map, wind_sched)
                scores.append(score)

                if sim.get("valid", False):
                    valid_count += 1

                # Atualiza best
                if score < best_score:
                    best_score = score
                    best = (ind, sim, score, datetime.utcnow().isoformat())

            # --- Estatísticas da geração ---------------------------
            avg_score = sum(scores) / len(scores)
            valid_pct = (valid_count / len(scores)) * 100.0

            with open(stats_file, 'a', newline='', encoding='utf-8') as sf:
                csv.writer(sf).writerow([gen, best_score, avg_score, valid_pct, int(time.time() - start_time)])

            # Salva checkpoint do best
            if best is not None:
                ind, sim, score, ts = best
                route, speeds = decode_individual(ind)
                outpath = os.path.join(out_dir, f"best_agent_{agent_id}.json")
                with open(outpath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'agent_id': agent_id,
                        'timestamp': ts,
                        'score': score,
                        'route': route,
                        'speeds': speeds,
                        'sim': sim
                    }, f, ensure_ascii=False, indent=2)

            # --- Critérios de parada ---------------------------------
            elapsed = time.time() - start_time
            if elapsed >= time_limit_seconds:
                print(f"[agent {agent_id}] reached TIME limit (elapsed={elapsed:.1f}s) best_score={best_score:.2f} gen={gen}")
                break

            if generations_limit and gen >= generations_limit:
                print(f"[agent {agent_id}] reached GENERATION cap (gen={gen}) best_score={best_score:.2f}")
                break

            # --- GERAR NOVA POPULAÇÃO -------------------------------
            new_pop = []

            if ELITISM and best is not None:
                new_pop.append(best[0])

            pool = list(zip(population, scores))

            while len(new_pop) < pop_size:
                candidates = random.sample(pool, k=min(TOURNAMENT_K, len(pool)))
                candidates.sort(key=lambda x: x[1])
                p1 = candidates[0][0]

                candidates = random.sample(pool, k=min(TOURNAMENT_K, len(pool)))
                candidates.sort(key=lambda x: x[1])
                p2 = candidates[0][0]

                child = crossover(p1, p2)
                child = mutate(child, mutation_rate)
                new_pop.append(child)

            population = new_pop

    except Exception as e:
        print(f"[agent {agent_id}] ERROR: {e}")

    finally:
        # Salva telemetria completa do agente
        diag_path = os.path.join(out_dir, f"diagnostic_agent_{agent_id}.json")
        try:
            with open(diag_path, 'w', encoding='utf-8') as df:
                json.dump(diagnostics, df, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[agent {agent_id}] ERROR saving diagnostics: {e}")

        print(f"[agent {agent_id}] EXIT best_score={best_score:.2f} gen={gen}")

# ----------------------------
# Parallel launcher & aggregator
# ----------------------------
def spawn_agents_and_wait(agents: int, hours: float, pop: int, generations: int,
                          seed: int, ceps_path: str, wind_path: str, out_dir: str, mutation_rate: float):
    seconds = int(hours * 3600)
    procs: List[Process] = []
    os.makedirs(out_dir, exist_ok=True)

    print(f"[MASTER] Spawning {agents} agents, time_limit={seconds}s, pop={pop}, seed={seed}")
    for i in range(agents):
        p = Process(target=agent_worker, args=(i, seconds, pop, generations, seed + i*17, ceps_path, wind_path, out_dir, mutation_rate))
        p.start()
        procs.append(p)
        time.sleep(0.2)  # small stagger

    print(f"[MASTER] All agents started. Waiting for completion...")
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("[MASTER] Interrupted by user. Terminating agents...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()

    # aggregate best results
    best = None
    for fname in os.listdir(out_dir):
        if fname.startswith("best_agent_") and fname.endswith(".json"):
            path = os.path.join(out_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    d = json.load(fh)
                score = d.get('score', float('inf'))
                if best is None or score < best[0]:
                    best = (score, d, path)
            except Exception:
                continue

    if best:
        score, data, path = best
        print(f"[MASTER] Best aggregated found: {path} score: {score}")
        with open(os.path.join(out_dir, "final_best.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # write solution.csv if possible
        rows = data.get('sim', {}).get('rows', [])
        if rows:
            save_solution_rows(rows, "solution.csv")
            print("[MASTER] solution.csv written from best agent.")
        else:
            print("[MASTER] Best sim has no rows to write.")
    else:
        print("[MASTER] No best agent outputs found in out dir.")

    # AGGREGATE DIAGNOSTICS into summary
    summary = {}
    for fname in os.listdir(out_dir):
        if fname.startswith("diagnostic_agent_") and fname.endswith('.json'):
            path = os.path.join(out_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    recs = json.load(fh)
                if not recs:
                    continue
                agent_id = int(fname.split('_')[-1].split('.')[0])
                best_valid = max(r['ceps_valid'] for r in recs)
                avg_valid = sum(r['ceps_valid'] for r in recs) / len(recs)
                fail_ceps = [r['fail_cep'] for r in recs if r.get('fail_cep')]
                most_common = None
                if fail_ceps:
                    cnt = Counter(fail_ceps).most_common(1)[0]
                    most_common = [cnt[0], int(cnt[1])]
                summary[agent_id] = {
                    'best_valid_route': int(best_valid),
                    'average_valid_ceps': float(avg_valid),
                    'most_problematic_cep': most_common
                }
            except Exception as e:
                print(f"[MASTER] error reading diagnostic {path}: {e}")
    try:
        with open(os.path.join(out_dir, 'diagnostic_summary.json'), 'w', encoding='utf-8') as sf:
            json.dump(summary, sf, ensure_ascii=False, indent=2)
        print('[MASTER] diagnostic_summary.json written in out/')
    except Exception as e:
        print(f"[MASTER] error writing diagnostic_summary: {e}")

# ----------------------------
# Local single-run mode (useful for debugging)
# ----------------------------
def run_local(pop: int, generations: int, ceps_path: str, wind_path: str, out_dir: str, mutation_rate: float, seed: int):
    ceps_map = load_ceps(ceps_path)
    wind_sched = load_wind(wind_path)
    all_middle = [c for c in ceps_map.keys() if c != BASE_CEP]
    population = [make_individual(all_middle) for _ in range(pop)]

    best = None
    best_score = float('inf')
    for gen in range(1, generations+1):
        scores = []
        details = []
        valid_count = 0
        for ind in population:
            score, sim = fitness_individual_penalized(ind, ceps_map, wind_sched)
            scores.append(score)
            details.append((ind, sim))
            if sim.get('valid', False):
                valid_count += 1
            if score < best_score:
                best_score = score
                best = (ind, sim, score, datetime.utcnow().isoformat())
        avg = sum(scores) / len(scores)
        valid_pct = (valid_count / len(scores)) * 100.0
        print(f"[LOCAL] Gen {gen}/{generations} best={best_score:.2f} avg={avg:.2f} valid%={valid_pct:.1f}")

        # produce next gen
        new_pop = []
        if ELITISM and best is not None:
            new_pop.append(best[0])
        pool = list(zip([i for i,_ in details], [s for s,_ in zip(scores, details)]))
        while len(new_pop) < pop:
            p1 = tournament_selection([i for i,_ in pool], [s for _,s in pool])
            p2 = tournament_selection([i for i,_ in pool], [s for _,s in pool])
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop

    if best:
        ind, sim, score, ts = best
        route, speeds = decode_individual(ind)
        with open(os.path.join(out_dir, "final_best_local.json"), 'w', encoding='utf-8') as f:
            json.dump({'agent_id':0,'timestamp':ts,'score':score,'route':route,'speeds':speeds,'sim':sim}, f, ensure_ascii=False, indent=2)
        if sim.get('rows'):
            save_solution_rows(sim['rows'], "solution.csv")
            print("[LOCAL] solution.csv written.")
        else:
            print("[LOCAL] best sim had no rows.")
    else:
        print("[LOCAL] no best found.")

# ----------------------------
# CLI / Entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Drone Surveyor - GA parallel runner (detailed)")
    parser.add_argument("--mode", choices=["local","parallel"], default="parallel", help="Modo: local (1 GA) ou parallel (N agentes)")
    parser.add_argument("--ceps", default=DEFAULT_CEPS_PATH, help="Caminho para ceps.csv")
    parser.add_argument("--wind", default=DEFAULT_WIND_PATH, help="Caminho para wind_schedule.csv")
    parser.add_argument("--agents", type=int, default=min(8, cpu_count()), help="Nº agentes (parallel mode)")
    parser.add_argument("--hours", type=float, default=4.0, help="Tempo em horas para executar (parallel mode)")
    parser.add_argument("--pop", type=int, default=DEFAULT_POP_SIZE, help="População por agente")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="Limite de gerações (só como fallback)")
    parser.add_argument("--seed", type=int, default=42, help="Semente base")
    parser.add_argument("--out", default="out", help="Diretório para outputs/checkpoints")
    parser.add_argument("--mutation", type=float, default=DEFAULT_MUTATION_RATE, help="Taxa de mutação")
    args = parser.parse_args()

    # copy uploaded coords if ceps file missing
    if not os.path.exists(args.ceps):
        if os.path.exists(UPLOADED_COORDS):
            os.makedirs(os.path.dirname(args.ceps), exist_ok=True)
            shutil.copy(UPLOADED_COORDS, args.ceps)
            print(f"[MAIN] {args.ceps} não encontrado; copiando upload {UPLOADED_COORDS}")
        else:
            raise FileNotFoundError(f"Arquivo de ceps não encontrado: {args.ceps}")

    if not os.path.exists(args.wind):
        raise FileNotFoundError(f"Arquivo de vento não encontrado: {args.wind}")

    os.makedirs(args.out, exist_ok=True)

    if args.mode == "local":
        print("[MAIN] Modo local selecionado")
        run_local(args.pop, int(args.generations), args.ceps, args.wind, args.out, args.mutation, args.seed)
    else:
        # parallel
        agents = max(1, args.agents)
        hours = float(args.hours)
        pop = max(2, args.pop)
        gens = int(args.generations) if args.generations > 0 else 0
        spawn_agents_and_wait(agents, hours, pop, gens, args.seed, args.ceps, args.wind, args.out, args.mutation)

if __name__ == "__main__":
    main()
